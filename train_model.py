"""
train_model.py
---------------
Two-phase transfer learning on MobileNetV2 for 7-class HAM10000
classification:
  Phase 1: freeze the MobileNetV2 base, train only the new head.
  Phase 2: unfreeze the top ~40 layers of the base and fine-tune at a
           low learning rate.

MobileNetV2 is used (rather than ResNet50) because the final .h5 is
~10-15MB vs ~100MB+, which matters a lot once you try to git-push the
model to a free hosting platform. Swap in ResNet50 by changing
`tf.keras.applications.MobileNetV2` -> `tf.keras.applications.ResNet50`
(and the matching preprocess_input in data_preprocessing.py / app.py)
if you'd rather trade file size for a bit more accuracy.

Run:
    python train_model.py

Output:
    skin_cancer_model.h5     <- full Keras model (architecture + weights)
    training_history.png     <- accuracy / loss curves
    confusion_matrix.png
    classification_report.txt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import (
    load_metadata, split_dataset, compute_class_weights,
    make_dataset, CLASS_NAMES, IMG_SIZE,
)

NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "skin_cancer_model.h5"
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 10
FINE_TUNE_LAST_N_LAYERS = 40


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Phase 1: fully frozen

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    # training=False keeps BatchNorm in inference mode even after we
    # unfreeze base_model in phase 2 — recommended by the Keras transfer
    # learning guide to avoid destroying the pretrained BN statistics.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history, path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved {path}")


def evaluate(model, test_ds, test_df):
    y_true = test_df["label_idx"].values
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    print("\n" + report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # Because of the imbalance, macro-averaged recall matters more than
    # raw accuracy here — a model that always predicts "nv" would score
    # ~67% accuracy while being medically useless.


def main():
    df = load_metadata()
    train_df, val_df, test_df = split_dataset(df)
    class_weights = compute_class_weights(train_df)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model, base_model = build_model()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    # ---------- Phase 1: train the head, base frozen ----------
    print("\n=== Phase 1: training classification head (base frozen) ===")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS,
        class_weight=class_weights, callbacks=cbs,
    )

    # ---------- Phase 2: fine-tune top layers of the base ----------
    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    freeze_until = len(base_model.layers) - FINE_TUNE_LAST_N_LAYERS
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # much lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS,
        class_weight=class_weights, callbacks=cbs,
    )

    combined_history = {k: history1.history[k] + history2.history[k] for k in history1.history}
    plot_history(combined_history)

    # ModelCheckpoint already saved the best epoch to MODEL_PATH; this
    # final save ensures the fine-tuned model is what's on disk.
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    evaluate(model, test_ds, test_df)


if __name__ == "__main__":
    main()
