# api_app.py
import os
import requests
import PyPDF2
from io import BytesIO
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

API_KEY = "54a8273bcceff8860cca909e4772b16cebfdda5f80d3a6ef557478979c84eb0d"

app = FastAPI(
    title="HackRx Claim Analyzer",
    description="Answers insurance policy questions from a PDF document.",
    version="1.0.0"
)


class HackRxRequest(BaseModel):
    documents: str  
    questions: list[str]  

class HackRxResponse(BaseModel):
    answers: list[str]

async def verify_auth(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing Bearer token")
    token = auth_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

def extract_pdf_text_from_url(pdf_url: str) -> str:
    """Download PDF from URL and extract text."""
    try:
        response = requests.get(pdf_url, timeout=20)
        response.raise_for_status()
        pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise ValueError("No readable text found in PDF.")
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def ask_llm(question: str, document_text: str) -> str:
    """Ask Gemini model a question about the document."""
    prompt = f"""
You are an AI insurance analyst. 
Based on the policy document below, answer the user's question concisely.

Policy Document:
{document_text}

Question:
{question}

Answer in plain text, no formatting.
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

# Endpoint
@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(payload: HackRxRequest, auth=Depends(verify_auth)):

    document_text = extract_pdf_text_from_url(payload.documents)

    answers = []
    for q in payload.questions:
        answer = ask_llm(q, document_text)
        answers.append(answer)

    return HackRxResponse(answers=answers)