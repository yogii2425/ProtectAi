# backend/alerts.py

import os
from datetime import datetime

ALERTS_FOLDER = "alerts"
if not os.path.exists(ALERTS_FOLDER):
    os.makedirs(ALERTS_FOLDER)


def generate_alert(image_file: str, prediction: str, confidence: float, matches: list) -> str:
    """
    Generate a user-friendly alert report.
    Saves report to alerts/ folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_filename = f"alert_{timestamp}.txt"
    alert_path = os.path.join(ALERTS_FOLDER, alert_filename)

    with open(alert_path, "w") as f:
        f.write("==== ProtectAI Alert ====\n")
        f.write(f"Image: {image_file}\n")
        f.write(f"Deepfake Prediction: {prediction} (confidence: {confidence:.2f})\n")
        f.write(f"Matches Found: {', '.join(matches) if matches else 'None'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=========================\n")

    return alert_path


def generate_takedown_request(image_file: str, prediction: str, matches: list) -> str:
    """
    Generate a takedown request template for reporting misuse.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    takedown_filename = f"takedown_{timestamp}.txt"
    takedown_path = os.path.join(ALERTS_FOLDER, takedown_filename)

    with open(takedown_path, "w") as f:
        f.write("To: Abuse/Privacy Team\n")
        f.write("Subject: Takedown Request - Misuse of Personal Image\n\n")
        f.write("Dear Team,\n\n")
        f.write(f"I am writing to request the immediate removal of content that misuses my personal image ({image_file}).\n")
        f.write(f"Detection Result: {prediction}\n")
        if matches:
            f.write(f"Matched Files: {', '.join(matches)}\n")
        f.write("\nThis violates my privacy rights. Please take urgent action.\n\n")
        f.write("Sincerely,\nUser\n")
        f.write(f"Timestamp: {timestamp}\n")

    return takedown_path
