# backend/search.py
from backend import db
from imagehash import hex_to_hash
import os
from backend.fingerprint import generate_fingerprint

def search_image_in_dataset(image_path: str, dataset_folder: str = "dataset") -> dict:
    """
    Search for similar images in the dataset using fingerprints.
    Returns a dict with matches.
    """
    query_fingerprint = generate_fingerprint(image_path)
    results = []

    if "Error" in query_fingerprint:
        return {"error": query_fingerprint}

    if not os.path.exists(dataset_folder):
        return {"error": f"Dataset folder '{dataset_folder}' not found."}

    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path):
            dataset_fingerprint = generate_fingerprint(file_path)
            if dataset_fingerprint == query_fingerprint:
                results.append(file)

    return {
        "query": os.path.basename(image_path),
        "matches": results if results else ["No matches found"]
    }
def search_similar(hash_value: str, threshold: int = 5):
    """
    Search for similar fingerprints in the database.
    Uses Hamming distance between hashes.
    
    threshold = max distance allowed (lower = stricter match)
    """
    results = []
    all_fps = db.get_fingerprints()
    
    target_hash = hex_to_hash(hash_value)

    for fp in all_fps:
        db_hash = hex_to_hash(fp.hash_value)
        distance = target_hash - db_hash  # Hamming distance
        if distance <= threshold:
            results.append({
                "filename": fp.filename,
                "hash": fp.hash_value,
                "distance": distance
            })
    
    return results
