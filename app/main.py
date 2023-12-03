import sys
from fastapi import FastAPI
from pydantic import BaseModel

from app.llama_cpp import run_summerize_text, run_format_leetcode_data_into_blog

version = f"{sys.version_info.major}.{sys.version_info.minor}"
app = FastAPI()

class Prompt(BaseModel):
    context: str

class LeetcodePayload(BaseModel):
    question: str
    notes: str
    solution: str

@app.post("/summarize-text")
async def summerize_text(prompt: Prompt):
    context = prompt.context
    result = run_summerize_text(context)
    return {"result": result}

@app.post("/leetcode-blog")
async def leetcode_blog(payload: LeetcodePayload):
    result = run_format_leetcode_data_into_blog(payload)
    return {"result": result}
