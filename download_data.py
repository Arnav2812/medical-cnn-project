# download_data.py
import os
import shutil
import kagglehub

print("Downloading dataset from Kaggle...")
# Downloads the Sartaj Bhuvaji Brain Tumor Classification Dataset
path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")

print(f"Downloaded to temporary cache: {path}")

# Target destination
target_dir = os.path.join(os.getcwd(), "data", "raw")

# Move or copy contents into our project data/raw directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Dataset successfully placed in: {target_dir}")