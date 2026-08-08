# 🧠 Medical Imaging Diagnostic API | CNN, MLOps, Docker & Kubernetes

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326ce5)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Observability-blueviolet)

An end-to-end, production-ready machine learning pipeline for classifying brain tumors from MRI scans. This repository demonstrates full-stack ML engineering capabilities, focusing on **Advanced Fine-Tuning**, **Responsible AI (Explainability)**, and **Scalable Deployment (Docker/Kubernetes)**.

## 🎯 Project Highlights & Enterprise Features
* **Advanced Deep Learning Architecture:** Upgraded from basic transfer learning to a fine-tuned VGG16/ResNet50 model utilizing nested Data Augmentation and unfreezed convolutional blocks for high-accuracy brain tissue feature extraction.
* **Decoupled Architecture (Version Drift Immunity):** Model structures are defined explicitly in Python, injecting only raw mathematical tensors (`.weights.h5`) to ensure the API is 100% immune to Keras/TensorFlow schema drift across environments.
* **Explainable AI (XAI):** Implements a custom `GradientTape` Grad-CAM script to generate heatmaps, proving exactly which pixels the AI utilized for its diagnosis, solving the "Black Box" problem.
* **Latency & Metric Optimization:** Includes automated evaluation scripts to measure precise batch latency trade-offs between VGG16 and ResNet50, alongside comprehensive Precision/Recall/F1-Score reports.
* **Containerized MLOps & Observability:** The inference engine is wrapped in an asynchronous FastAPI server, instrumented with OpenTelemetry to trace execution times, and fully containerized via Docker and Kubernetes for scalable cloud deployment.

---

## 📂 Repository Structure
```text
medical-cnn-project/
├── data/
│   ├── raw/                  # Downloaded Kaggle MRI Dataset (Ignored in Git)
│   └── processed/            # Generated Grad-CAM XAI Visualizations
├── k8s/
│   ├── deployment.yaml       # Kubernetes replicas and scaling config
│   └── service.yaml          # Kubernetes load balancing configuration
├── models/                   # Decoupled Tensor Weights (e.g., vgg16_advanced.weights.h5)
├── src/
│   ├── app.py                # FastAPI Web Server, Model Builder & OpenTelemetry Tracing
│   ├── data_pipeline.py      # tf.data.Dataset Pipeline (Caching & Prefetching)
│   ├── evaluate.py           # Model latency and classification report benchmarking
│   ├── explainability.py     # Advanced Nested Grad-CAM Heatmap Generation
│   └── train.py              # Transfer Learning, Fine-Tuning & Augmentation Scripts
├── Dockerfile                # Production Docker Image Configuration
├── download_data.py          # Kagglehub API Fetcher
├── requirements.txt          # Python Dependencies
├── test_endpoint.py          # Automated API Payload Testing & Heatmap Trigger
└── README.md                 # Project Documentation
```

## ⚙️ Local Installation & Setup
Clone the repository and initialize the environment:

```bash
git clone <your-github-repo-url>
cd medical-cnn-project
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Acquire the Dataset:

```bash
python download_data.py
```

Train & Evaluate Models:
Run the training script (optimized for GPU) to generate the decoupled .weights.h5 files, then evaluate their latency and F1-scores.

```bash
python src/train.py
python src/evaluate.py
```

## 🐳 Docker Deployment
The model is designed to be deployed as a completely isolated microservice.

Build the Docker Image:

```bash
docker build -t cnn-mri-api:v1.0 .
```

Run the Containerized Inference Engine:

```bash
docker run -d -p 8000:8000 --name cnn-mri-container cnn-mri-api:v1.0
```

Monitor Container Observability:
Watch the OpenTelemetry traces log in real-time as the server receives requests:

```bash
docker logs -f cnn-mri-container
```

## 🩺 Testing & Clinical Explainability (XAI)
Once the Docker container is running on port 8000, you can interact with the API in multiple ways:

**1. Visual Web UI (Swagger)**
Navigate to `http://localhost:8000/docs` to visually upload MRI scans and receive instant JSON diagnostic predictions.

**2. Automated Pipeline Testing & XAI Generation**
Use the unified testing script to send a REST payload to the Docker API and automatically generate a clinical Grad-CAM heatmap overlay.

```bash
python test_endpoint.py
```

* **Output 1:** The JSON prediction is returned from the Docker API (e.g., Glioma Tumor: 96.42%).
* **Output 2:** A visual heatmap is saved to `data/processed/heatmap_X.jpg` highlighting the exact tissue anomalies the model focused on.
