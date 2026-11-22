
import numpy as np
import tensorflow as tf
import cv2

def generate_gradcam(model, image_batch, class_index):
    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + 1e-8),
        heatmap
    )

    return heatmap.numpy()


def create_overlay(image, heatmap):
    image = image.convert("RGB").resize((224, 224))
    original = np.array(image)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap, cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original, 0.6, colored_heatmap, 0.4, 0
    )

    return overlay
