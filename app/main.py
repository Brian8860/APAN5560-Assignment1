import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

nlp = spacy.load("en_core_web_lg")


class EmbeddingRequest(BaseModel):
    word: str


def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


@app.get("/")
def read_root():
    return {"message": "FastAPI word embedding API is running"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)

    return {
        "word": request.word,
        "embedding": embedding.tolist()
    }