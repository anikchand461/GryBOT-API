from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import asyncio
import os
from db import init_db, save_chat
from chat import get_bot_response_with_key

# Initialize DB
init_db()

app = FastAPI(title="GryBOT-API")

# ===== Models =====
class ChatRequest(BaseModel):
    query: str

db_lock = asyncio.Lock()

# ===== Routes =====
@app.post("/chat")
async def chat(
    req: ChatRequest,
    gemini_key: str | None = Header(default=None)  # optional user key
):
    # Fallback to internal key if no key provided
    if not gemini_key:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise HTTPException(status_code=500, detail="No Gemini API key available")

    # Run chatbot response safely in a separate thread
    answer = await asyncio.to_thread(get_bot_response_with_key, req.query, gemini_key)

    # Save chat to DB
    async with db_lock:
        await asyncio.to_thread(save_chat, req.query, answer)

    return {"answer": answer}