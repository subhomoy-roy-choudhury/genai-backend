import sys
from fastapi import FastAPI
from pydantic import BaseModel

from app.llama_cpp import run_summerize_text, run_format_leetcode_data_into_blog, run_summerize_docs

version = f"{sys.version_info.major}.{sys.version_info.minor}"
app = FastAPI()

import google.generativeai as palm
api_key = 'AIzaSyBMpA_LWZ0aOBdvlw3e2LTbLvblQYznuwU' # put your API key here
palm.configure(api_key=api_key)

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

@app.post("/summarize-docs")
async def summerize_docs(prompt: Prompt):
    url = prompt.context
    result = run_summerize_docs(url)
    return {"result": result}

@app.post("/leetcode-blog")
async def leetcode_blog(payload: LeetcodePayload):
    result = run_format_leetcode_data_into_blog(payload)
    return {"result": result}
