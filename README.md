# 🧠 Brain Tumor MRI Diagnostic API | MLOps & XAI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Observability-blueviolet)

An end-to-end, production-ready machine learning pipeline for classifying brain tumors from MRI scans. This repository demonstrates full-stack ML engineering capabilities, focusing heavily on **Responsible AI (Explainability)** and **MLOps (Observability)**.

## 🎯 Project Objectives
Built to showcase enterprise-grade AI integration, this project bridges the gap between raw medical imaging data and a deployable diagnostic web service. 
* **High-Accuracy Classification:** Utilizes Transfer Learning (VGG16 & ResNet50) for robust feature extraction.
* **Clinical Transparency:** Implements Grad-CAM to generate visual heatmaps, ensuring AI decisions are explainable and trustworthy.
* **Operational Health:** Wraps the inference engine in an asynchronous FastAPI server, instrumented with OpenTelemetry for real-time latency and trace monitoring.

---

## 📂 Repository Structure

```text
medical_cnn_project/
├── data/
│   ├── raw/                  # Downloaded Kaggle MRI Dataset (Ignored in Git)
│   └── processed/            # Generated Grad-CAM Visualizations
├── models/                   # Compiled .h5 Model Weights (Ignored in Git)
├── src/
│   ├── app.py                # FastAPI Web Server & OpenTelemetry Tracing
│   ├── data_pipeline.py      # tf.data.Dataset Pipeline (Caching & Prefetching)
│   ├── explainability.py     # Grad-CAM Heatmap Generation logic
│   └── train.py              # Transfer Learning & Model Tuning Scripts
├── download_data.py          # Kagglehub API Fetcher
├── requirements.txt          # Python Dependencies
└── README.md                 # Project Documentation
```

## ⚙️ Installation & Setup

1. Clone the repository and initialize the environment:

```bash
git clone <your-github-repo-url>
cd medical_cnn_project
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

2. Acquire the Dataset:

This project utilizes the Sartaj Bhuvaji Brain Tumor Classification MRI dataset. Fetch it directly via the Kaggle API:

```bash
python download_data.py
```

3. Train the Models:

Initiate the transfer learning pipeline to generate custom VGG16 and ResNet50 weights:

```bash
python src/train.py
```

4. Generate Explainability Reports (Optional):

Test the Grad-CAM visualization on a sample image:

```bash
python src/explainability.py
```

## 🚀 Deployment (FastAPI + OpenTelemetry)

Launch the ASGI web server to expose the inference engine:

```bash
uvicorn src.app:app --reload --port 8000
```

* **Interactive API Docs:** Navigate to `http://127.0.0.1:8000/docs` to use the Swagger UI.
* **Health Check:** `GET /health` endpoint for Kubernetes/Docker orchestrator monitoring.
* **Inference Engine:** `POST /predict` accepts an image upload and returns class probabilities alongside real-time OpenTelemetry span tracking in the server console.
