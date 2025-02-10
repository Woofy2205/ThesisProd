import os

from fastapi import UploadFile
from model import Movie

UPLOAD_FOLDER = "./static"

async def upload_pdf(documents: UploadFile):
    # Save documents
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    documents_path = os.path.join(UPLOAD_FOLDER, documents.filename)
    with open(documents_path, "wb") as buffer:
        buffer.write(await documents.read())
    
    return documents
