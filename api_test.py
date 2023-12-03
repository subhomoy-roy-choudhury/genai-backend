import requests
import json

def read_file(filepath):
    with open(filepath, "r") as file:
        data = file.read()
    return data

data = {
    "question": read_file("samples/README.md"),
    "notes": read_file("samples/NOTES.md"),
    "solution": read_file("samples/code.py"),
}


print(data)
response = requests.post("http://localhost:8000/leetcode-blog", data=json.dumps(data))