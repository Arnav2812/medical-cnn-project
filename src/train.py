# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_pipeline import load_datasets, IMAGE_SIZE, BATCH_SIZE

# ==================== GPU CONFIGURATION ====================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Dynamically allocate GPU memory as needed instead of allocating all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"⚡ GPU Detected & Configured: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU Config Error: {e}")
else:
    print("⚠️ No physical GPU detected by TensorFlow. Running on CPU.")
# ===========================================================

# Training Configurations
EPOCHS = 10  # Increased for GPU training
LEARNING_RATE = 0.0001


def build_vgg16_model(num_classes):
    """Builds transfer learning model using VGG16 base."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    
    # Freeze pre-trained base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="VGG16_Brain_Tumor")
    return model


def build_resnet50_model(num_classes):
    """Builds transfer learning model using ResNet50 base."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    
    # Freeze pre-trained base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="ResNet50_Brain_Tumor")
    return model


def train_and_save_model(model, train_ds, val_ds, save_name):
    """Compiles, trains, and exports model weights in modern .keras format."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    save_path = os.path.join(os.getcwd(), "models", save_name)
    callbacks = [
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    print(f"\n================ Starting Training: {model.name} ================")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"Model saved successfully to: {save_path}\n")
    return history


if __name__ == "__main__":
    # 1. Load Data (Dataset yields pre-batched data based on BATCH_SIZE)
    train_dataset, val_dataset, class_names = load_datasets()
    num_classes = len(class_names)

    os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)

    # 2. Train VGG16
    vgg_model = build_vgg16_model(num_classes)
    train_and_save_model(vgg_model, train_dataset, val_dataset, "vgg16_tumor_net.keras")

    # 3. Train ResNet50
    resnet_model = build_resnet50_model(num_classes)
    train_and_save_model(resnet_model, train_dataset, val_dataset, "resnet_tumor_net.keras")