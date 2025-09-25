import pytest
from agent import process_incoming_email
from models import EmailMessage
from trainer import add_training_example, reindex

def setup_module(module):
    add_training_example({"email": "My transfer failed", "reply": "We are sorry. Please share your ID"})
    reindex()

def test_agent_reply():
    email = EmailMessage(from_addr="test@example.com", subject="Issue", body="transfer failed")
    result = process_incoming_email(email)
    assert "reply" in result
    assert "confidence" in result
