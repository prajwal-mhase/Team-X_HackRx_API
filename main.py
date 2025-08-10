import os
import json
import re
import requests
from io import BytesIO
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import mimetypes
import hashlib
import logging
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import PyPDF2
import google.generativeai as genai

# DOCX & EML support
try:
    import docx
except ImportError:
    docx = None

import email
from email import policy

# OCR support
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# FAISS & embeddings (optional)
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    faiss = None
    embed_model = None
    EMBEDDINGS_AVAILABLE = False

# ------------------------
# Config
# ------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set.")
genai.configure(api_key=GEMINI_API_KEY)

TEAM_AUTH_TOKEN = os.getenv("TEAM_AUTH_TOKEN", "54a8273bcceff8860cca909e4772b16cebfdda5f80d3a6ef557478979c84eb0d")

logger = logging.getLogger("uvicorn.error")

# ------------------------
# Auth Check
# ------------------------
def verify_auth_token(auth_header: Optional[str]):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = auth_header.split("Bearer ")[1].strip()
    if token != TEAM_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token.")

# ------------------------
# Document download with retry
# ------------------------
def download_with_retry(url, retries=3, backoff=5):
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content, resp.headers.get("Content-Type")
        except Exception as e:
            logger.error(f"Download attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                sleep_time = backoff * (2 ** attempt)  # exponential backoff
                time.sleep(sleep_time)
            else:
                raise

# ------------------------
# File Parsing
# ------------------------
def extract_pdf_text(uploaded_file: bytes) -> str:
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip() and OCR_AVAILABLE:
            images = convert_from_bytes(uploaded_file)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    return text.strip()

def extract_docx_text(uploaded_file: bytes) -> str:
    if not docx:
        raise HTTPException(status_code=500, detail="DOCX support not installed.")
    try:
        doc = docx.Document(BytesIO(uploaded_file))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_eml_text(uploaded_file: bytes) -> str:
    try:
        msg = email.message_from_bytes(uploaded_file, policy=policy.default)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_content()
        else:
            return msg.get_content()
    except Exception as e:
        logger.error(f"Error reading EML: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading EML: {str(e)}")
    return ""

def extract_text_from_file(file_bytes: bytes, url: str, content_type: Optional[str] = None) -> str:
    path = urlparse(url).path.lower()

    if path.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    elif path.endswith(".docx"):
        return extract_docx_text(file_bytes)
    elif path.endswith(".eml"):
        return extract_eml_text(file_bytes)

    if content_type:
        ct = content_type.lower()
        if "pdf" in ct:
            return extract_pdf_text(file_bytes)
        elif "word" in ct or "officedocument" in ct:
            return extract_docx_text(file_bytes)
        elif "message/rfc822" in ct or "eml" in ct:
            return extract_eml_text(file_bytes)

    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        if "pdf" in guessed:
            return extract_pdf_text(file_bytes)
        elif "word" in guessed:
            return extract_docx_text(file_bytes)
        elif "message/rfc822" in guessed:
            return extract_eml_text(file_bytes)

    raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: PDF, DOCX, EML")

# ------------------------
# Chunking & Embeddings
# ------------------------
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks: List[str]):
    if not EMBEDDINGS_AVAILABLE:
        return None, None
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity with normalized vectors
    index.add(embeddings)
    return index, embeddings

def save_faiss_index(index, embeddings, meta, path_prefix: str):
    faiss.write_index(index, path_prefix + ".faiss")
    np.save(path_prefix + "_embeddings.npy", embeddings)
    with open(path_prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_faiss_index(path_prefix: str):
    if not (os.path.exists(path_prefix + ".faiss") and
            os.path.exists(path_prefix + "_embeddings.npy") and
            os.path.exists(path_prefix + "_meta.json")):
        return None, None, None
    index = faiss.read_index(path_prefix + ".faiss")
    embeddings = np.load(path_prefix + "_embeddings.npy")
    with open(path_prefix + "_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, embeddings, meta

# ------------------------
# LLM Parser for structured query
# ------------------------
class ParsedQuery(BaseModel):
    intent: Optional[str]
    entities: Optional[Dict[str, Any]] = Field(default_factory=dict)
    subqueries: Optional[List[str]] = Field(default_factory=list)

def parse_with_llm(raw_question: str) -> ParsedQuery:
    prompt = f"""
You are an AI assistant that extracts structured information from user questions about insurance policies.

Extract the following as JSON keys:
- "intent": one-word string summarizing question intent (e.g., "coverage", "exclusion", "limit_check")
- "entities": key-value pairs of important terms or values mentioned (e.g., "procedure", "date", "hospital_type")
- "subqueries": list of simpler focused questions derived from the main question (optional)

User question:
\"\"\"{raw_question}\"\"\" 

Return ONLY valid JSON with those keys. Example:
{{
  "intent": "coverage",
  "entities": {{"procedure": "appendectomy", "hospital_type": "AYUSH"}},
  "subqueries": ["Is appendectomy covered?", "Are AYUSH hospitals covered for surgeries?"]
}}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})

    try:
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        parsed_json = json.loads(match.group(0))
        return ParsedQuery(**parsed_json)
    except Exception as e:
        logger.warning(f"Failed to parse LLM JSON: {e}. Falling back.")
        return ParsedQuery(intent=None, entities={}, subqueries=[raw_question])

# ------------------------
# Clause Matching (combined scoring)
# ------------------------
def score_clause(chunk: str, query_embedding: 'np.ndarray', chunk_embedding: 'np.ndarray', entities: dict) -> float:
    vec_score = float(np.dot(query_embedding, chunk_embedding))  # cosine similarity (dot product)
    token_score = 0
    for k, v in entities.items():
        if str(v).lower() in chunk.lower():
            token_score += 1
    token_score_norm = token_score / max(1, len(entities))
    combined = 0.7 * vec_score + 0.3 * token_score_norm
    return combined

def retrieve_relevant_clauses(parsed_query: ParsedQuery, chunks: List[str], index, embeddings, top_k=5) -> List[Dict]:
    if not EMBEDDINGS_AVAILABLE or index is None or embeddings is None:
        candidates = []
        queries = parsed_query.subqueries if parsed_query.subqueries else [""]
        for q in queries:
            q_lower = q.lower()
            scored = [(chunk.lower().count(q_lower), i, chunk) for i, chunk in enumerate(chunks)]
            scored = [x for x in scored if x[0] > 0]
            scored.sort(reverse=True)
            for count, i, chunk in scored[:top_k]:
                candidates.append({"id": i, "text": chunk, "score": float(count)})
        unique = {c['id']: c for c in candidates}
        return sorted(unique.values(), key=lambda x: x['score'], reverse=True)[:top_k]

    query_text = parsed_query.subqueries[0] if parsed_query.subqueries else ""
    if not query_text:
        query_text = ""

    q_emb = embed_model.encode([query_text], normalize_embeddings=True)[0]

    scored_chunks = []
    for i, (chunk, chunk_emb) in enumerate(zip(chunks, embeddings)):
        score = score_clause(chunk, q_emb, chunk_emb, parsed_query.entities or {})
        scored_chunks.append({"id": i, "text": chunk, "score": score})

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]

# ------------------------
# Logic Evaluation (deterministic + LLM)
# ------------------------
def evaluate_logic(parsed_query: ParsedQuery, clauses: List[Dict], original_question: str) -> Dict:
    for c in clauses:
        txt = c["text"].lower()
        if any(kw in txt for kw in ("excluded", "not covered", "exclusion")):
            return {
                "outcome": "denied",
                "confidence": 0.95,
                "reasons": ["Policy contains exclusion clause relevant to your question."],
                "supporting_clauses": clauses,
                "rationale": f"Found exclusion clause: {c['text']}"
            }

    # Limit total text length of clauses for token efficiency
    clauses_summary = "\n".join(c["text"][:1000] for c in clauses)

    prompt = f"""
You are a claims policy analyst. Using the following clauses and the user's question, provide a JSON with keys:

- outcome: one of "approved", "denied", "needs_follow_up", "partial"
- confidence: a float 0-1 representing certainty
- reasons: list of string explanations
- supporting_clause_ids: list of clause ids used
- rationale: detailed explanation in plain English

User question:
\"\"\"{original_question}\"\"\"

Clauses:
\"\"\"{clauses_summary}\"\"\"

Parsed query:
\"\"\"{parsed_query.json(indent=2)}\"\"\"

Return only valid JSON.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    try:
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        return json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Failed to parse LLM decision JSON: {e}")
        return {
            "outcome": "needs_follow_up",
            "confidence": 0.5,
            "reasons": ["Unable to confidently determine outcome."],
            "supporting_clause_ids": [c["id"] for c in clauses],
            "rationale": response.text.strip()
        }

# ------------------------
# Generate concise answer text from LLM and clauses
# ------------------------
def generate_answer_text(question: str, clauses: List[Dict]) -> str:
    # Compose a prompt to produce a short, clear answer for the user
    clauses_summary = "\n".join(c["text"][:1000] for c in clauses)
    prompt = f"""
You are a helpful assistant that reads insurance policy clauses and answers user questions clearly and concisely.

User question:
\"\"\"{question}\"\"\"

Relevant clauses:
\"\"\"{clauses_summary}\"\"\"

Provide a brief answer in 1-3 sentences, without references or JSON.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})

    return response.text.strip()

# ------------------------
# LLM with Retrieval and concise answer
# ------------------------
def ask_llm_with_context(document_chunks: List[str], question: str, index, embeddings) -> dict:
    parsed_query = parse_with_llm(question)
    matched_clauses = retrieve_relevant_clauses(parsed_query, document_chunks, index, embeddings, top_k=5)
    decision = evaluate_logic(parsed_query, matched_clauses, question)
    answer_text = generate_answer_text(question, matched_clauses)
    return {
        "question": question,
        "parsed_query": parsed_query.dict(),
        "decision": decision,
        "supporting_clauses": matched_clauses,
        "answer": answer_text
    }

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI(
    title="Intelligent Query–Retrieval API",
    description="Processes policy/legal/HR/compliance documents with semantic retrieval & contextual LLM reasoning.",
    version="3.0.0"
)

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_analysis(payload: QuestionRequest, authorization: Optional[str] = Header(None)):
    verify_auth_token(authorization)

    try:
        file_bytes, content_type = download_with_retry(payload.documents)
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(status_code=400, detail="Could not download document.")

    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 15MB).")

    doc_text = extract_text_from_file(file_bytes, payload.documents, content_type)
    if not doc_text:
        raise HTTPException(status_code=400, detail="No readable text found in document.")

    chunks = chunk_text(doc_text, chunk_size=500)

    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    index_path_prefix = f"data/faiss_index_{doc_hash}"

    index, embeddings, meta = None, None, None
    if EMBEDDINGS_AVAILABLE:
        index, embeddings, meta = load_faiss_index(index_path_prefix)
        if index is None or embeddings is None or meta is None:
            index, embeddings = build_faiss_index(chunks)
            meta = [{"chunk_id": i} for i in range(len(chunks))]
            os.makedirs("data", exist_ok=True)
            save_faiss_index(index, embeddings, meta, index_path_prefix)

    answers = []
    for question in payload.questions:
        result = ask_llm_with_context(chunks, question, index, embeddings)
        answers.append(result["answer"])

    return {
        "answers": answers
    }
