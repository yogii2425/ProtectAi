# backend/fingerprint.py

import hashlib
from PIL import Image
import numpy as np

def generate_fingerprint(image_path: str) -> str:
    """
    Generate a unique fingerprint (hash) of an image using SHA256 on pixel values.
    """
    try:
        img = Image.open(image_path).convert("L")  # grayscale
        img = img.resize((128, 128))              # normalize size
        pixels = np.array(img).flatten()
        pixel_bytes = pixels.tobytes()

        # Hash the image pixels
        fingerprint = hashlib.sha256(pixel_bytes).hexdigest()
        return fingerprint
    except Exception as e:
        return f"Error generating fingerprint: {str(e)}"
