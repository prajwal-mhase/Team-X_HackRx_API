# api_app.py

import PyPDF2
import json
import re
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# ✅ Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# ------------------------
# Utility Functions
# ------------------------

def extract_pdf_text(uploaded_file):
    """Extract text from PDF bytes."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_json_from_text(text):
    """Extract the first JSON object from text."""
    try:
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("No JSON object found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

def ask_llm(query, document_text):
    """Send prompt to Gemini model and get structured JSON."""
    prompt = f"""
You are an AI insurance analyst.

User Query:
{query}

Policy Document:
{document_text}

Your task:
1. Determine if the claim should be approved or rejected.
2. Specify the claim amount (if applicable).
3. Justify the decision.

Return ONLY a JSON response in the following format (no markdown, no explanation):

{{
  "decision": "approved or rejected",
  "amount": "amount in INR or null",
  "justification": "your reasoning"
}}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text.strip()

# ------------------------
# FastAPI Setup
# ------------------------

app = FastAPI(
    title="Insurance Claim Analyzer API",
    description="API endpoint for insurance claim analysis",
    version="1.0.0"
)

# Response Model
class ClaimAnalysisResponse(BaseModel):
    decision: str
    amount: str | None
    justification: str

@app.post("/api/v1/hackrx/run", response_model=ClaimAnalysisResponse)
async def analyze_claim(
    pdf_file: UploadFile = File(..., description="Insurance policy PDF file"),
    claim_description: str = Form(..., description="Natural language claim description")
):
    try:
        # Read PDF
        pdf_bytes = await pdf_file.read()
        doc_text = extract_pdf_text(pdf_bytes)

        if not doc_text:
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        # Ask LLM
        raw = ask_llm(claim_description, doc_text)
        result = extract_json_from_text(raw)

        return ClaimAnalysisResponse(
            decision=result.get("decision", ""),
            amount=result.get("amount", None),
            justification=result.get("justification", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

