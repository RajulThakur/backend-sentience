from os import getenv
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv() 
app = FastAPI()

class Emotion(BaseModel):
    emotion: str
    percent: int
    words: list[str]
class Analysis(BaseModel):
    emotions: list[Emotion]
    percent: list[int]
    confidence: int
    accuracy: int

class SentimentRequest(BaseModel):
    text: str

@app.get("/api/test")
async def root():
    return {"message": "Hello World"}

@app.post("/api/sentiment-analysis")
async def sentiment_analysis(request: SentimentRequest):
    GEMINI_API = getenv("GEMINI_API")
    if not GEMINI_API:
        return {"error": "GEMINI_API environment variable not set"}
    print(GEMINI_API)
    client = genai.Client(api_key=GEMINI_API)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=request.text,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Analysis],
        },
    )
    print(response.text)
    return {"response": response.parsed}