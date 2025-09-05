from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from chat import get_bot_response
from db import init_db

app = FastAPI()

# Initialize database on startup
init_db()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(req: ChatRequest, gemini_api_key: str = Header(...)):
    if not gemini_api_key:
        raise HTTPException(status_code=401, detail="Gemini API key required")

    answer = get_bot_response(req.query, gemini_api_key)
    return {"answer": answer}