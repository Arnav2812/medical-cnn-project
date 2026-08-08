# test_endpoint.py
import os
import requests
from src.explainability import generate_heatmap

# =====================================================================
# 🛠️ CHANGE THIS PATH TO TEST DIFFERENT IMAGES
IMAGE_TO_TEST = "data/raw/Testing/no_tumor/image(2).jpg"
# =====================================================================

URL = "http://127.0.0.1:8000/predict"
absolute_image_path = os.path.join(os.getcwd(), IMAGE_TO_TEST)

if not os.path.exists(absolute_image_path):
    print(f"❌ File not found at: {absolute_image_path}")
    exit(1)

print(f"\n🚀 1. Sending MRI image to Docker API: {absolute_image_path}")

# 1. Ping the Docker API for the JSON Prediction
with open(absolute_image_path, "rb") as img_file:
    files = {"file": (os.path.basename(absolute_image_path), img_file, "image/jpeg")}
    try:
        response = requests.post(URL, files=files)
        if response.status_code == 200:
            print("✅ API Prediction Successful:")
            print(response.json())
        else:
            print(f"❌ API Request failed with status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the Docker container running?")

# 2. Generate the visual heatmap locally
print(f"\n🧠 2. Generating Explainability Heatmap...")
generate_heatmap(absolute_image_path)
print("Done!\n")