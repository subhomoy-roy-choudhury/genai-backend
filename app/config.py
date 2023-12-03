import os

DEBUG = int(os.getenv("DEBUG", "0"))
MODEL_PATH=os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_S.gguf")