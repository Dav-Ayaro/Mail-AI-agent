from models import EmailMessage
from vector_store import find_best_match
from smtp_utils import send_email
from config import OPENAI_API_KEY
import openai
import os

CONFIDENCE_THRESHOLD = 0.75

def generate_openai_reply(email: EmailMessage, context: str) -> str:
    """
    Uses OpenAI to generate a reply based on the email and context.
    """
    openai.api_key = OPENAI_API_KEY
    prompt = f"""You are a helpful AI assistant for DotPay, a money transfer service.
Customer email: {email.body}
Business context: {context}
Write a professional, human-like reply:"""

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error generating reply with OpenAI: {str(e)}"


def process_incoming_email(email: EmailMessage):
    """
    Process incoming email and generate an auto-reply.
    Uses training examples first, and falls back to OpenAI if API key is set.
    """
    match, confidence = find_best_match(email.body)

    if not match:
        reply_text = "Sorry, we couldn’t understand your request. A human will follow up."
    else:
        reply_text = match.reply

    # If OpenAI key is available, enhance/fallback
    if OPENAI_API_KEY:
        reply_text = generate_openai_reply(email, match.reply if match else "General support")

    # Auto-send only if confidence is high enough
    if confidence >= CONFIDENCE_THRESHOLD:
        send_email(
            to_addr=email.from_addr,
            subject=f"Re: {email.subject}",
            body=reply_text
        )
        return {"status": "reply sent", "reply": reply_text, "confidence": float(confidence)}
    else:
        return {"status": "manual review", "reply": reply_text, "confidence": float(confidence)}
