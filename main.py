from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import ChatBot

app = FastAPI()
chatbot = ChatBot()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "HackSprint ChatBot API is running!"}

@app.post("/chat")
def chat(query: Query):
    answer = chatbot.get_response(query.question)
    return {"answer": answer}
