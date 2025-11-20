from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import answer_question

app = FastAPI()

class Ask(BaseModel):
    question: str

@app.post("/api/ask-faq")
def ask_faq(req: Ask):
    ans, follow = answer_question(req.question)
    return {
        "answer": ans,
        "followup_used": follow
    }
