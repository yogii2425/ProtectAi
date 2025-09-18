# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil, os
from backend import fingerprint, db

# Initialize FastAPI app
app = FastAPI(title="ProtectAI - Agentic Security Assistant")

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize DB
db.init_db()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate fingerprint
        hash_value = fingerprint.fingerprint_image(file_path)

        # Save to DB
        db.save_fingerprint(file.filename, hash_value)

        return {
            "filename": file.filename,
            "fingerprint": hash_value,
            "message": "Image uploaded and fingerprint generated successfully!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
