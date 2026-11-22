
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# Model paths
MODEL_DIR = "models"
IMAGE_DIR = "demo_images/demo"

# Load models
stage1 = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "stage1_final_best.keras"),
    compile=False
)

stage2_old = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "stage2_old_mobilenetv2.keras"),
    compile=False
)

stage2_focal = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "stage2_exp1_focal_loss.keras"),
    compile=False
)

# Load ensemble weights
with open(os.path.join(MODEL_DIR, "stage2_ensemble_config.json")) as f:
    config = json.load(f)

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

print("\nModels loaded successfully!\n")

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    # Temporary preprocessing assumption:
    # Verify this against the original training pipeline.
    image_array = np.array(image, dtype=np.float32) / 127.5 - 1.0
    image_array = np.expand_dims(image_array, axis=0)

    # Stage 1
    stage1_probability = float(stage1.predict(image_array, verbose=0)[0][0])

    print(f"\nImage: {filename}")
    print(f"Stage 1 lesion probability: {stage1_probability:.4f}")

    if stage1_probability >= 0.5:
        print("Stage 1 result: LESION")

        # Stage 2 weighted ensemble
        pred_old = stage2_old.predict(image_array, verbose=0)[0]
        pred_focal = stage2_focal.predict(image_array, verbose=0)[0]

        final_prediction = (
            config["model_1_weight"] * pred_old
            + config["model_2_weight"] * pred_focal
        )

        predicted_index = int(np.argmax(final_prediction))

        print("Stage 2 category:", class_names[predicted_index])
        print("Stage 2 confidence:", f"{final_prediction[predicted_index] * 100:.2f}%")

    else:
        print("Stage 1 result: NORMAL")
