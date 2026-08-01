# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_pipeline import load_datasets, IMAGE_SIZE, BATCH_SIZE

# Training Configurations
EPOCHS = 3  # Set to 3 for fast execution on CPU (increase if using GPU)
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
    """Compiles, trains, and exports model weights."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    save_path = os.path.join(os.getcwd(), "models", save_name)
    callbacks = [
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
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
    # 1. Load Data
    train_dataset, val_dataset, class_names = load_datasets()
    num_classes = len(class_names)

    # Ensure models directory exists
    os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)

    # 2. Train VGG16
    vgg_model = build_vgg16_model(num_classes)
    train_and_save_model(vgg_model, train_dataset, val_dataset, "vgg16_tumor_net.h5")

    # 3. Train ResNet50
    resnet_model = build_resnet50_model(num_classes)
    train_and_save_model(resnet_model, train_dataset, val_dataset, "resnet_tumor_net.h5")