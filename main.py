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
import email
from email import policy

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import PyPDF2
import google.generativeai as genai

# ------------------------
# Optional DOCX support
# ------------------------
try:
    import docx
except ImportError:
    docx = None

# ------------------------
# Optional OCR support
# ------------------------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ------------------------
# Config
# ------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set.")

genai.configure(api_key=GEMINI_API_KEY)

TEAM_AUTH_TOKEN = os.getenv(
    "TEAM_AUTH_TOKEN",
    "54a8273bcceff8860cca909e4772b16cebfdda5f80d3a6ef557478979c84eb0d"
)

logger = logging.getLogger("uvicorn.error")

# ------------------------
# Auth
# ------------------------
def verify_auth_token(auth_header: Optional[str]):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = auth_header.split("Bearer ")[1].strip()
    if token != TEAM_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token.")

# ------------------------
# Download with retry
# ------------------------
def download_with_retry(url: str, retries: int = 3, backoff: int = 5):
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content, resp.headers.get("Content-Type")
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise

# ------------------------
# File parsing
# ------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip() and OCR_AVAILABLE:
            images = convert_from_bytes(file_bytes)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF read error: {e}")
    return text.strip()

def extract_docx_text(file_bytes: bytes) -> str:
    if not docx:
        raise HTTPException(status_code=500, detail="DOCX support not installed.")
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_eml_text(file_bytes: bytes) -> str:
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_content()
    return msg.get_content()

def extract_text_from_file(file_bytes: bytes, url: str, content_type: Optional[str]) -> str:
    path = urlparse(url).path.lower()

    if path.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if path.endswith(".docx"):
        return extract_docx_text(file_bytes)
    if path.endswith(".eml"):
        return extract_eml_text(file_bytes)

    if content_type:
        ct = content_type.lower()
        if "pdf" in ct:
            return extract_pdf_text(file_bytes)
        if "word" in ct:
            return extract_docx_text(file_bytes)
        if "message/rfc822" in ct:
            return extract_eml_text(file_bytes)

    raise HTTPException(status_code=400, detail="Unsupported file type")

# ------------------------
# Chunking
# ------------------------
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# ------------------------
# Parsed query schema
# ------------------------
class ParsedQuery(BaseModel):
    intent: Optional[str]
    entities: Dict[str, Any] = Field(default_factory=dict)
    subqueries: List[str] = Field(default_factory=list)

# ------------------------
# LLM parsing
# ------------------------
def parse_with_llm(question: str) -> ParsedQuery:
    prompt = f"""
Extract structured info as JSON with keys:
intent, entities, subqueries.

Question:
\"\"\"{question}\"\"\"
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    try:
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        return ParsedQuery(**json.loads(match.group(0)))
    except Exception:
        return ParsedQuery(subqueries=[question])

# ------------------------
# Keyword clause matching
# ------------------------
def retrieve_relevant_clauses(
    parsed: ParsedQuery,
    chunks: List[str],
    top_k: int = 5
) -> List[Dict]:

    queries = parsed.subqueries or [""]
    scored = []

    for i, chunk in enumerate(chunks):
        score = 0
        text_lower = chunk.lower()
        for q in queries:
            score += text_lower.count(q.lower())
        for v in parsed.entities.values():
            if str(v).lower() in text_lower:
                score += 1
        if score > 0:
            scored.append({"id": i, "text": chunk, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ------------------------
# Decision logic
# ------------------------
def evaluate_logic(parsed: ParsedQuery, clauses: List[Dict], question: str) -> Dict:
    for c in clauses:
        if any(k in c["text"].lower() for k in ["excluded", "not covered", "exclusion"]):
            return {
                "outcome": "denied",
                "confidence": 0.95,
                "reasons": ["Relevant exclusion found"],
                "supporting_clause_ids": [c["id"]],
                "rationale": c["text"]
            }

    prompt = f"""
Answer as JSON with outcome, confidence, reasons.

Question:
{question}

Clauses:
{chr(10).join(c["text"][:800] for c in clauses)}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    try:
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        return json.loads(match.group(0))
    except Exception:
        return {
            "outcome": "needs_follow_up",
            "confidence": 0.5,
            "reasons": ["Insufficient clarity"],
            "supporting_clause_ids": [],
            "rationale": response.text
        }

# ------------------------
# Short answer
# ------------------------
def generate_answer(question: str, clauses: List[Dict]) -> str:
    prompt = f"""
Answer briefly (1–3 sentences).

Question:
{question}

Clauses:
{chr(10).join(c["text"][:800] for c in clauses)}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()

# ------------------------
# API
# ------------------------
app = FastAPI(
    title="Intelligent Policy QA API",
    version="3.0.0"
)

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run(payload: QuestionRequest, authorization: Optional[str] = Header(None)):
    verify_auth_token(authorization)

    file_bytes, content_type = download_with_retry(payload.documents)

    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    text = extract_text_from_file(file_bytes, payload.documents, content_type)
    chunks = chunk_text(text)

    answers = []
    for q in payload.questions:
        parsed = parse_with_llm(q)
        clauses = retrieve_relevant_clauses(parsed, chunks)
        answers.append(generate_answer(q, clauses))

    return {"answers": answers}

# ------------------------
# Local run
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
