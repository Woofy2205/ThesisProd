from typing import Annotated, List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title= "Simple RAG",
    description= "A simple RAG app",
    version= "0.1",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_pdf(documents: UploadFile = File(...)):
    return {"filename": documents.filename}