# src/explainability.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Input
from tensorflow.keras.models import Model, Sequential

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


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


def get_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3"):
    """Advanced Grad-CAM: Threads through a nested model architecture."""
    # 1. Extract the layers by index
    data_aug_layer = model.layers[1]
    vgg_model = model.layers[2]
    gap_layer = model.layers[3]
    dense_1 = model.layers[4]
    dropout_layer = model.layers[5]
    dense_out = model.layers[6]

    # 2. Build a micro-model just for the nested VGG convolutions
    inner_grad_model = tf.keras.models.Model(
        inputs=vgg_model.inputs, 
        outputs=[vgg_model.get_layer(last_conv_layer_name).output, vgg_model.output]
    )

    # 3. Manually trace the forward pass to capture gradients
    with tf.GradientTape() as tape:
        # Pass input (training=False skips augmentation changes)
        x = data_aug_layer(img_array, training=False)
        
        # Get VGG output and target convolution map
        conv_outputs, vgg_outputs = inner_grad_model(x)
        tape.watch(conv_outputs)
        
        # Pass through the classification head
        x = gap_layer(vgg_outputs)
        x = dense_1(x)
        x = dropout_layer(x, training=False)
        preds = dense_out(x)
        
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate Gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy(), pred_index.numpy()


def generate_heatmap(image_path):
    """Loads weights, generates heatmap, and saves it."""
    weights_path = os.path.join(os.getcwd(), "models", "vgg16_advanced.weights.h5")
    
    model = build_advanced_vgg16(len(CLASS_NAMES))
    model.load_weights(weights_path)

    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    heatmap, pred_idx = get_gradcam_heatmap(img_batch, model)

    output_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"heatmap_{os.path.basename(image_path)}")

    img_arr = img_array / 255.0
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMAGE_SIZE).numpy().squeeze()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr)
    plt.title("Original Scan")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_arr)
    plt.imshow(heatmap_resized, cmap="jet", alpha=0.4)
    plt.title(f"Grad-CAM ({CLASS_NAMES[pred_idx].upper()})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"📸 Explainability Heatmap saved to: {output_path}")