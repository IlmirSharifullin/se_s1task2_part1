from fastapi import FastAPI
from pydantic import BaseModel
from sentiment import sentiment_analyzer


class Item(BaseModel):
    text: str


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return sentiment_analyzer(item.text)[0]
