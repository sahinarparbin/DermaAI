import tensorflow as tf

model = tf.keras.models.load_model(
    "models/stage2_exp1_focal_loss.keras",
    compile=False
)

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name, layer.output.shape)