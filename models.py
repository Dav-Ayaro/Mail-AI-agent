from pydantic import BaseModel
from typing import Optional, Dict

class TrainingExample(BaseModel):
    email: str
    reply: str
    metadata: Optional[Dict] = None

class EmailMessage(BaseModel):
    from_addr: str
    subject: str
    body: str
