# setup_sample_dataset.py
import os
import requests

# Create folders
folders = ["dataset/train/real", "dataset/train/fake", "dataset/val/real", "dataset/val/fake"]
for f in folders:
    os.makedirs(f, exist_ok=True)

# Some sample images (replace with your own if you want bigger dataset)
real_images = [
    "https://upload.wikimedia.org/wikipedia/commons/3/37/Albert_Einstein_Head.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/99/Barack_Obama.jpg"
]

fake_images = [
    "https://thispersondoesnotexist.com/image",  # AI-generated face
    "https://thispersondoesnotexist.com/image"   # another fake face
]

def download(url, path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded {path}")
        else:
            print(f"⚠️ Failed {url}")
    except Exception as e:
        print(f"Error: {e}")

# Download sample images
for i, url in enumerate(real_images):
    download(url, f"dataset/train/real/real_{i}.jpg")
    download(url, f"dataset/val/real/real_val_{i}.jpg")

for i, url in enumerate(fake_images):
    download(url, f"dataset/train/fake/fake_{i}.jpg")
    download(url, f"dataset/val/fake/fake_val_{i}.jpg")
