# src/app.py
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Input
from tensorflow.keras.models import Model, Sequential

# OpenTelemetry Instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

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

# Configuration - Updated for Advanced Model
WEIGHTS_PATH = os.path.join(os.getcwd(), "models", "vgg16_advanced.weights.h5")
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMAGE_SIZE = (224, 224)

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
        "telemetry_active": True
    }

@app.post("/predict", tags=["Inference Engine"])
async def predict_mri(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="CNN Model is not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    with tracer.start_as_current_span("mri_preprocessing_and_inference"):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

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