# backend/main.py

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from backend.fingerprint import generate_fingerprint
from backend.search import search_image_in_dataset
from backend.deepfake_detector import detect_deepfake
from backend.alerts import generate_alert, generate_takedown_request

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="ProtectAI", description="Agentic AI Security Assistant")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.get("/")
def home():
    return {"message": "Welcome to ProtectAI Security Assistant 🚀"}


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Step 1: Generate fingerprint + search in dataset
        fingerprint = generate_fingerprint(file_location)
        search_result = search_image_in_dataset(file_location, dataset_folder="dataset")

        # Step 2: Run deepfake detection
        detection_result = detect_deepfake(file_location)

        if "error" in detection_result:
            return JSONResponse(status_code=500, content=detection_result)

        # Step 3: Generate alert & takedown request
        matches = search_result.get("matches", [])
        alert_path = generate_alert(
            file.filename,
            detection_result["prediction"],
            detection_result["confidence"],
            matches
        )
        takedown_path = generate_takedown_request(
            file.filename,
            detection_result["prediction"],
            matches
        )

        return {
            "file": file.filename,
            "fingerprint": fingerprint,
            "search_result": search_result,
            "deepfake_result": detection_result,
            "alert_file": alert_path,
            "takedown_request": takedown_path
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
