import os
import tensorflow as tf

models = [
    "stage1_final_best.keras",
    "stage2_old_mobilenetv2.keras",
    "stage2_exp1_focal_loss.keras"
]

for name in models:
    model = tf.keras.models.load_model(
        os.path.join("models", name),
        compile=False
    )

    print("\nMODEL:", name)
    print("Last 10 layers:")

    for layer in model.layers[-10:]:
        activation = getattr(layer, "activation", None)
        activation_name = getattr(activation, "__name__", "N/A")
        print(layer.name, layer.__class__.__name__, activation_name)
