```markdown
# 📑 Insurance Claim Analyzer – HackRx 6.0

An AI-powered backend service to **analyze and validate insurance claims** using uploaded policy documents (PDF) and natural language claim descriptions.

Built for **HackRx 6.0** by Bajaj Finserv Health.

---

## 🚀 Features

- 📄 Upload insurance policy PDFs
- 💬 Describe claims in natural language
- 🤖 Google Gemini 2.0 Flash analyzes the policy & claim
- ✅ Outputs:
  - Decision: `approved` or `rejected`
  - Estimated Amount (if applicable)
  - Justification for the decision

---

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com)
- **LLM**: Google [Gemini 2.0 Flash](https://ai.google.dev)
- **PDF Parsing**: [PyPDF2](https://pypi.org/project/PyPDF2/)
- **Language**: Python 3.x
- **Hosting**: [Render](https://render.com)

---

## 🌐 Live API Endpoint

Once deployed on Render, your API will be available here:

```

[https://your-service-name.onrender.com/api/v1/hackrx/run](https://your-service-name.onrender.com/api/v1/hackrx/run)

````

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

---

## ▶️ How to Run Locally

1. **Set your environment variable**:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"  # Mac/Linux
   set GEMINI_API_KEY="your_api_key_here"     # Windows (PowerShell)
   ```
2. **Run FastAPI with Uvicorn**:

   ```bash
   uvicorn api_app:app --host 0.0.0.0 --port 8000
   ```
3. Open Swagger API docs at:

   ```
   http://localhost:8000/docs
   ```

---

## 🌐 Deploy on Render

1. Push this project to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Add environment variable:

   * `GEMINI_API_KEY` = your Gemini API key
4. Build command:

   ```bash
   pip install -r requirements.txt
   ```
5. Start command:

   ```bash
   bash start.sh
   ```
6. After deployment, your public API will be live at:

   ```
   https://your-service-name.onrender.com/api/v1/hackrx/run
   ```

---

## 📥 API Input Format

**Method**: `POST`
**Content Type**: `multipart/form-data`

| Field               | Type   | Description                    |
| ------------------- | ------ | ------------------------------ |
| `pdf_file`          | file   | Insurance policy PDF           |
| `claim_description` | string | Claim details in plain English |

---

## 📤 API Output Format

**Response**: JSON

```json
{
  "decision": "approved",
  "amount": "50000",
  "justification": "Coverage includes accidental damage..."
}
```

---

## 🧪 Example Request

```bash
curl -X POST "https://your-service-name.onrender.com/api/v1/hackrx/run" \
  -F "pdf_file=@policy.pdf" \
  -F "claim_description=Car accident with front damage"
```

---

## 👥 Team

* Prajwal Mhase
* Ram Darekar
* Payal Lanke
* Gauri Lanke

---

## 🤝 Acknowledgments

* HackRx 6.0 – Bajaj Finserv Health
* Google Gemini AI
* FastAPI Community
* Render Hosting Platform

---

```
