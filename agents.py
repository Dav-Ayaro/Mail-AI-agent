"""
Extended AI Email Agent
- Adds SMTP sending (auto-reply)
- Adds simple confidence scoring based on similarity values

Dependencies:
  pip install fastapi uvicorn python-multipart requests pydantic scikit-learn sentence-transformers
  pip install aiosmtplib

Configure SMTP via environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os
import json
import uuid
import datetime
import asyncio

# Try to import sentence-transformers. If not available, fallback TF-IDF.
USE_SBERT = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    USE_SBERT = True
except Exception:
    USE_SBERT = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SMTP
import aiosmtplib

# OPTIONAL: OpenAI wrapper (only used if OPENAI_API_KEY env var is set)
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
USE_OPENAI = bool(OPENAI_KEY)
if USE_OPENAI:
    try:
        import requests
    except Exception:
        USE_OPENAI = False

DATA_DIR = Path("./email_agent_data")
DATA_DIR.mkdir(exist_ok=True)
EXAMPLES_FILE = DATA_DIR / "examples.json"
INDEX_FILE = DATA_DIR / "index.json"

if not EXAMPLES_FILE.exists():
    EXAMPLES_FILE.write_text(json.dumps([]))

# SMTP config
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "noreply@example.com")

# Pydantic models
class TrainExample(BaseModel):
    email: str
    reply: str
    metadata: Optional[dict] = None

class EmailIncoming(BaseModel):
    from_addr: str
    to_addr: Optional[str]
    subject: Optional[str]
    body: str
    metadata: Optional[dict] = None

# VectorStore (simple)
class SimpleVectorStore:
    def __init__(self):
        self.docs: List[str] = []
        self.ids: List[str] = []
        self.embeddings = None
        self.vectorizer = None
        if USE_SBERT:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None

    def add(self, doc: str, id: Optional[str] = None):
        self.docs.append(doc)
        self.ids.append(id or str(uuid.uuid4()))

    def build(self):
        if USE_SBERT and self.model is not None:
            embs = self.model.encode(self.docs, show_progress_bar=False)
            self.embeddings = embs
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.embeddings = self.vectorizer.fit_transform(self.docs)

    def search(self, query: str, k: int = 3):
        if len(self.docs) == 0:
            return []
        if USE_SBERT and self.model is not None:
            q_emb = self.model.encode([query])[0]
            sims = cosine_similarity([q_emb], self.embeddings)[0]
        else:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.embeddings)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = [(self.ids[i], self.docs[i], float(sims[i])) for i in top_idx]
        return results

# LLM wrapper
class LLM:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.use_openai = bool(self.openai_key)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if self.use_openai:
            return self._openai_generate(prompt, max_tokens)
        return self._local_generate(prompt)

    def _openai_generate(self, prompt: str, max_tokens: int = 256) -> str:
        url = 'https://api.openai.com/v1/completions'
        headers = {'Authorization': f'Bearer {self.openai_key}', 'Content-Type': 'application/json'}
        payload = {'model': 'text-davinci-003', 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.2}
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data['choices'][0]['text'].strip()

    def _local_generate(self, prompt: str) -> str:
        return ("Hello,\n\nThanks for your message. Based on our context, we will follow up shortly.\n\n"
                "Best regards,\nDots AI")

# App
from fastapi import FastAPI
app = FastAPI(title="Email AI Agent (extended)")
store = SimpleVectorStore()
llm = LLM()

# helpers

def load_examples():
    return json.loads(EXAMPLES_FILE.read_text())

def save_example(example: dict):
    arr = load_examples()
    arr.append(example)
    EXAMPLES_FILE.write_text(json.dumps(arr, indent=2))

async def send_smtp(to_addr: str, subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        return {"status": "skipped", "reason": "SMTP not configured"}
    message = (f"From: {SMTP_FROM}\r\n"
               f"To: {to_addr}\r\n"
               f"Subject: {subject}\r\n"
               f"\r\n{body}")
    await aiosmtplib.send(
        message,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        username=SMTP_USER,
        password=SMTP_PASS,
        start_tls=True,
    )
    return {"status": "sent"}

@app.post('/train')
async def train(example: TrainExample):
    now = datetime.datetime.utcnow().isoformat()
    record = {'id': str(uuid.uuid4()), 'email': example.email, 'reply': example.reply, 'metadata': example.metadata or {}, 'created_at': now}
    save_example(record)
    return {'status': 'ok', 'id': record['id']}

@app.post('/reindex')
async def reindex():
    examples = load_examples()
    store.docs = []
    store.ids = []
    for ex in examples:
        doc = f"EXAMPLE_REPLY:\n{ex['reply']}\nEXAMPLE_EMAIL:\n{ex['email']}"
        store.add(doc, ex['id'])
    store.build()
    INDEX_FILE.write_text(json.dumps({'count': len(store.docs)}))
    return {'status': 'reindexed', 'count': len(store.docs)}

@app.post('/webhook/email')
async def webhook_email(email: EmailIncoming):
    docs = store.search(email.body, k=3)
    snippets = '\n'.join([f"- {d[1]}" for d in docs])
    confidences = [d[2] for d in docs]
    confidence_score = max(confidences) if confidences else 0.0

    system = ("You are Dots Support Assistant. Reply politely and concisely (3-6 sentences). "
              "Do not invent sensitive info.\n\n")
    incoming = f"Incoming Email:\n{email.body}"
    prompt = f"{system}Relevant Context:\n{snippets}\n\n{incoming}\n\nCompose a friendly reply."

    reply = llm.generate(prompt)

    auto_reply_status = None
    if confidence_score > 0.4:  # threshold for auto-reply
        subj = f"Re: {email.subject or 'Your inquiry'}"
        auto_reply_status = await send_smtp(email.from_addr, subj, reply)

    return {'status': 'generated', 'reply': reply, 'confidence': confidence_score, 'auto_reply': auto_reply_status}

@app.get('/examples')
async def examples():
    return load_examples()

@app.get('/')
async def root():
    return {"message": "Email AI Agent with SMTP + confidence scoring"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
