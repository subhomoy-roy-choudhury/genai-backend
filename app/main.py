import sys
from fastapi import FastAPI
from pydantic import BaseModel

from app.llama_cpp import run_llama_cpp

version = f"{sys.version_info.major}.{sys.version_info.minor}"
app = FastAPI()

class Prompt(BaseModel):
    context: str

@app.post("/summarize-text")
async def read_root(prompt: Prompt):
    context = prompt.context
    result = run_llama_cpp(context)
    return {"result": result}
