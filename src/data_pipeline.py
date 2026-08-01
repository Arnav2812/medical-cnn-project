# src/data_pipeline.py
import os
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def load_datasets(data_dir=None, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    """
    Loads raw MRI images from folder structure and returns split train/val datasets.
    """
    if data_dir is None:
        # Default to data/raw/Training if available from Kaggle structure
        base_path = os.path.join(os.getcwd(), "data", "raw")
        if os.path.exists(os.path.join(base_path, "Training")):
            data_dir = os.path.join(base_path, "Training")
        else:
            data_dir = base_path

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}. Please run download_data.py first.")

    print(f"Loading training data from: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"Detected Classes ({len(class_names)}): {class_names}")

    # Optimize pipeline with AUTOTUNE caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    train_dataset, val_dataset, classes = load_datasets()
    print("Data Pipeline Verification Successful!")
    print(f"Number of training batches: {len(train_dataset)}")
    print(f"Number of validation batches: {len(val_dataset)}")