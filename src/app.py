import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import deque
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Input
from tensorflow.keras.models import Model, Sequential

# OpenTelemetry Instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

import logging

# Configure a file handler for drift alerts
logging.basicConfig(level=logging.INFO)
drift_logger = logging.getLogger("drift_monitor")
os.makedirs("logs", exist_ok=True)
drift_handler = logging.FileHandler("logs/drift_alerts.log")
drift_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
drift_logger.addHandler(drift_handler)

# Initialize OpenTelemetry Tracing
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

FastAPIInstrumentor.instrument_app(app)

# Configuration & Constants
WEIGHTS_PATH = os.path.join(os.getcwd(), "models", "vgg16_advanced.weights.h5")
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMAGE_SIZE = (224, 224)

# In-Memory Observability State
CONFIDENCE_HISTORY = deque(maxlen=100)
DRIFT_THRESHOLD = 0.70  # Triggers a warning if rolling mean drops below 70%

model = None

def build_advanced_vgg16(num_classes=4):
    """Explicitly defines the Advanced VGG16 architecture structure."""
    base_model = VGG16(weights=None, include_top=False, input_shape=(*IMAGE_SIZE, 3))
    
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=inputs, outputs=outputs)

@app.on_event("startup")
def load_trained_model():
    """Instantiates architecture and loads weight tensors into memory."""
    global model
    try:
        model = build_advanced_vgg16(len(CLASS_NAMES))
        if os.path.exists(WEIGHTS_PATH):
            model.load_weights(WEIGHTS_PATH)
            print(f"[INIT] Advanced Model loaded successfully from: {WEIGHTS_PATH}")
        else:
            print(f"[ERROR] Weights file not found at: {WEIGHTS_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")

@app.get("/health", tags=["Operational Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "telemetry_active": True,
        "rolling_samples_tracked": len(CONFIDENCE_HISTORY)
    }

@app.get("/drift", tags=["Operational Health"])
def get_drift_status():
    """Returns the rolling confidence window metrics and drift alert status."""
    sample_count = len(CONFIDENCE_HISTORY)
    rolling_avg = (sum(CONFIDENCE_HISTORY) / sample_count) if sample_count > 0 else 1.0
    is_drifted = (rolling_avg < DRIFT_THRESHOLD) if sample_count >= 20 else False

    return {
        "drift_detected": is_drifted,
        "current_rolling_avg": round(float(rolling_avg), 4),
        "drift_threshold": DRIFT_THRESHOLD,
        "samples_collected": sample_count,
        "status": "ALERT: Performance Drift Detected" if is_drifted else "Normal"
    }

@app.post("/predict", tags=["Inference Engine"])
async def predict_mri(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="CNN Model is not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    with tracer.start_as_current_span("mri_preprocessing_and_inference") as span:
        # Preprocess input image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # Execute prediction
        predictions = model.predict(img_batch)
        predicted_index = int(np.argmax(predictions[0]))
        confidence_score = float(predictions[0][predicted_index])
        predicted_label = CLASS_NAMES[predicted_index]

        # Instrument OpenTelemetry attributes
        span.set_attribute("model.prediction.class", predicted_label)
        span.set_attribute("model.prediction.confidence", confidence_score)

        # Performance drift tracking
        CONFIDENCE_HISTORY.append(confidence_score)

        if len(CONFIDENCE_HISTORY) >= 20:
            rolling_avg = sum(CONFIDENCE_HISTORY) / len(CONFIDENCE_HISTORY)
            span.set_attribute("model.rolling_confidence_avg", rolling_avg)

            if rolling_avg < DRIFT_THRESHOLD:
                # 1. Log to file
                drift_logger.warning(
                    f"Performance Drift Alert! Rolling Avg: {rolling_avg:.4f} (Threshold: {DRIFT_THRESHOLD})"
                )
                
                # 2. Add OpenTelemetry span event
                span.add_event(
                    name="performance_drift_warning",
                    attributes={
                        "message": "Potential data/performance drift detected: low mean confidence",
                        "current_rolling_avg": float(rolling_avg),
                        "window_size": len(CONFIDENCE_HISTORY)
                    }
                )

    return {
        "filename": file.filename,
        "diagnosis": predicted_label,
        "confidence": f"{round(confidence_score * 100, 2)}%",
        "class_probabilities": {
            class_name: f"{round(float(prob) * 100, 2)}%"
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
    }