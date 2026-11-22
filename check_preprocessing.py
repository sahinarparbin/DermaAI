
import os
import tensorflow as tf

model_names = [
    "stage1_final_best.keras",
    "stage2_old_mobilenetv2.keras",
    "stage2_exp1_focal_loss.keras"
]

for name in model_names:
    model = tf.keras.models.load_model(
        os.path.join("models", name),
        compile=False
    )

    print("\nMODEL:", name)

    for layer in model.layers:
        if any(word in layer.__class__.__name__.lower()
               for word in ["rescaling", "normalization", "lambda"]):
            print(layer.name, layer.__class__.__name__)
