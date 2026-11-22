"""
data_preprocessing.py
----------------------
Loads the HAM10000 metadata, builds stratified train/val/test splits
(grouped by lesion_id so the same lesion never leaks across splits),
computes class weights to counter the severe class imbalance
(melanocytic nevi ~= 67% of the dataset), and defines a tf.data
augmentation pipeline.

Expected folder structure (after downloading from Kaggle):

    data/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    │   └── *.jpg
    └── HAM10000_images_part_2/
        └── *.jpg

Run directly for a sanity check:
    python data_preprocessing.py
"""

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data"
METADATA_CSV = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# The 7 diagnostic classes in HAM10000
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def build_image_path_index(data_dir=DATA_DIR):
    """Map image_id -> absolute file path, searching both image folders
    (and falling back to a recursive glob for other Kaggle download
    layouts, e.g. a single combined 'HAM10000_images' folder)."""
    patterns = [
        os.path.join(data_dir, "HAM10000_images_part_1", "*.jpg"),
        os.path.join(data_dir, "HAM10000_images_part_2", "*.jpg"),
        os.path.join(data_dir, "**", "*.jpg"),
    ]
    path_index = {}
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            image_id = os.path.splitext(os.path.basename(path))[0]
            path_index.setdefault(image_id, path)
    return path_index


def load_metadata(metadata_csv=METADATA_CSV, data_dir=DATA_DIR):
    df = pd.read_csv(metadata_csv)
    path_index = build_image_path_index(data_dir)

    df["path"] = df["image_id"].map(path_index)
    missing = int(df["path"].isna().sum())
    if missing:
        print(f"[warn] {missing} images referenced in metadata were not found on "
              f"disk and will be dropped. Check that both image folders were "
              f"extracted into {data_dir}/.")
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    df["label"] = df["dx"].astype("category")
    df["label_idx"] = df["label"].cat.set_categories(CLASS_NAMES).cat.codes

    print("Class distribution (raw counts):")
    print(df["dx"].value_counts())
    print("\nClass distribution (%):")
    print((df["dx"].value_counts(normalize=True) * 100).round(2))
    return df


def split_dataset(df, test_size=0.15, val_size=0.15, seed=SEED):
    """Stratified split by dx, grouped by lesion_id so multiple images
    of the same lesion don't end up split across train/val/test (which
    would leak information and inflate validation accuracy)."""
    unique_lesions = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    train_lesions, temp_lesions = train_test_split(
        unique_lesions,
        test_size=(test_size + val_size),
        stratify=unique_lesions["dx"],
        random_state=seed,
    )
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=test_size / (test_size + val_size),
        stratify=temp_lesions["dx"],
        random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_lesions["lesion_id"])].reset_index(drop=True)

    print(f"\nSplit sizes -> train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights(train_df):
    """Inverse-frequency class weights to counter the severe imbalance
    (nv is ~67% of the dataset, df and vasc are each <2%). Passed to
    model.fit(class_weight=...) so the loss penalizes mistakes on rare
    classes (melanoma, akiec, etc.) more heavily."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["label_idx"].values,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print("\nComputed class weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>6s}: {class_weight_dict[i]:.3f}")
    return class_weight_dict


def _load_and_preprocess(path, label, training):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    if training:
        # Lesion photos have no canonical orientation, so flips/rotations
        # are safe (unlike e.g. photos of text or faces).
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    label = tf.one_hot(label, depth=len(CLASS_NAMES))
    return image, label


def make_dataset(df, training, batch_size=BATCH_SIZE):
    paths = df["path"].values
    labels = df["label_idx"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    df = load_metadata()
    train_df, val_df, test_df = split_dataset(df)
    class_weights = compute_class_weights(train_df)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    for images, labels in train_ds.take(1):
        print("\nSanity check batch shapes:", images.shape, labels.shape)

    print("\nDatasets built successfully.")
