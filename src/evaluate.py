# src/evaluate.py
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_pipeline import load_datasets

CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def evaluate_model_performance(model_path, val_ds):
    """Evaluates accuracy, precision, recall, F1-score, and latency metrics."""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"\n================ Evaluating: {os.path.basename(model_path)} ================")
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    latencies = []

    for images, labels in val_ds:
        start_time = time.time()
        preds = model.predict(images, verbose=0)
        end_time = time.time()

        # Record latency per batch (in milliseconds)
        batch_latency = (end_time - start_time) * 1000
        latencies.append(batch_latency)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    avg_latency = np.mean(latencies)
    print(f"⚡ Average Batch Latency: {avg_latency:.2f} ms")
    print(f"📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    _, val_dataset, _ = load_datasets()

    vgg_path = os.path.join(os.getcwd(), "models", "vgg16_tumor_net.keras")
    resnet_path = os.path.join(os.getcwd(), "models", "resnet_tumor_net.keras")

    evaluate_model_performance(vgg_path, val_dataset)
    evaluate_model_performance(resnet_path, val_dataset)