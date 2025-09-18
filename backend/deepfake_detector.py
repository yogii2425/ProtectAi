# backend/deepfake_detector.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# -------------------------
# Define CNN Model (same as in train.py)
# -------------------------
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # assuming input resized to 256x256
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------
# Load model
# -------------------------
MODEL_PATH = "models/deepfake_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeCNN().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print(f"[Warning] Model file not found at {MODEL_PATH}. Run train.py first.")


# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def detect_deepfake(image_path: str) -> dict:
    """
    Run deepfake detection on an image.
    Returns prediction and confidence.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, dim=0)

        label = "FAKE" if predicted_class.item() == 1 else "REAL"
        return {
            "file": os.path.basename(image_path),
            "prediction": label,
            "confidence": float(confidence.item())
        }

    except Exception as e:
        return {"error": str(e)}
