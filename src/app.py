# src/app.py
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model

# OpenTelemetry Instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize OpenTelemetry Tracing to Console
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("medical-cnn-observability")

# Initialize FastAPI App
app = FastAPI(
    title="Medical Imaging Diagnostic API",
    description="Production-ready CNN API with OpenTelemetry Observability and Explainable AI.",
    version="1.0.0"
)

# Instrument FastAPI App with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Global Configuration
MODEL_PATH = os.path.join(os.getcwd(), "models", "vgg16_tumor_net.h5")
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMAGE_SIZE = (224, 224)

model = None


@app.on_event("startup")
def load_trained_model():
    """Load trained model weights into memory when application starts."""
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"[INIT] Model successfully loaded from: {MODEL_PATH}")
    else:
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")


@app.get("/health", tags=["Operational Health"])
def health_check():
    """Health check endpoint for container orchestrators (K8s/Docker)."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "telemetry_active": True
    }


@app.post("/predict", tags=["Inference Engine"])
async def predict_mri(file: UploadFile = File(...)):
    """Receives an MRI scan, runs VGG16 prediction, and logs telemetry spans."""
    if model is None:
        raise HTTPException(status_code=500, detail="CNN Model is not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Custom OpenTelemetry Span for Inference Engine Tracking
    with tracer.start_as_current_span("mri_preprocessing_and_inference"):
        # 1. Preprocess Image File
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # 2. Model Inference
        predictions = model.predict(img_batch)
        predicted_index = int(np.argmax(predictions[0]))
        confidence_score = float(predictions[0][predicted_index])
        predicted_label = CLASS_NAMES[predicted_index]

    return {
        "filename": file.filename,
        "diagnosis": predicted_label,
        "confidence": f"{round(confidence_score * 100, 2)}%",
        "class_probabilities": {
            class_name: f"{round(float(prob) * 100, 2)}%"
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
    }