from fastapi import FastAPI
from models import EmailMessage, TrainingExample
from trainer import add_training_example, reindex
from agents import process_incoming_email

app = FastAPI()

@app.post("/train")
def train(example: TrainingExample):
    add_training_example(example)
    return {"status": "ok", "message": "Training example added."}

@app.post("/reindex")
def reindex_data():
    reindex()
    return {"status": "ok", "message": "Reindexed successfully."}

@app.post("/webhook/email")
def webhook_email(email: EmailMessage):
    result = process_incoming_email(email)
    return result
