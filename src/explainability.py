# src/explainability.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap highlighting regions influencing prediction."""
    # Build a model that outputs the last conv layer and final prediction
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output class w.r.t the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive activations and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), pred_index.numpy()


def overlay_and_save_gradcam(image_path, heatmap, output_path, alpha=0.4):
    """Overlays the heatmap on top of the original MRI scan and saves the output."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_arr = img_to_array(img) / 255.0

    # Rescale heatmap to 0-255
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMAGE_SIZE).numpy().squeeze()
    
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr)
    plt.title("Original MRI Scan")
    plt.axis("off")

    # Grad-CAM Heatmap Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr)
    plt.imshow(heatmap_resized, cmap="jet", alpha=alpha)
    plt.title("Grad-CAM Explainability Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Explainability visual saved to: {output_path}")


def explain_mri_scan(image_path, model_path, last_conv_layer_name="block5_conv3"):
    """Runs prediction and generates explainability report for a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    print(f"\nAnalyzing scan: {image_path}")
    model = load_model(model_path)

    # Preprocess image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)  # Batch dimension

    # Generate Heatmap
    heatmap, pred_idx = get_gradcam_heatmap(img_batch, model, last_conv_layer_name)
    predicted_class = CLASS_NAMES[pred_idx]

    print(f"Model Diagnosis: {predicted_class.upper()}")

    # Save output visualization
    output_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradcam_explanation.png")

    overlay_and_save_gradcam(image_path, heatmap, output_path)
    return predicted_class, output_path


if __name__ == "__main__":
    # Test on a sample glioma image from the raw dataset
    sample_dir = os.path.join(os.getcwd(), "data", "raw", "Training", "glioma_tumor")
    
    if os.path.exists(sample_dir):
        sample_image = os.path.join(sample_dir, os.listdir(sample_dir)[0])
        model_file = os.path.join(os.getcwd(), "models", "vgg16_tumor_net.h5")

        explain_mri_scan(sample_image, model_file)
    else:
        print("Sample image directory not found. Please verify data setup.")