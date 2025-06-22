import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = MobileNetV2(weights="imagenet")
last_conv_layer_name = "Conv_1"

def get_img_array(img_path, size=(224, 224)):
    img = image.load_img(img_path, target_size=size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array), np.array(img)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, original_img, alpha=0.4, output_path="gradcam_output.jpg"):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    
    # Save the result
    cv2.imwrite(output_path, superimposed_img)
    print(f"✅ Grad-CAM result saved to: {output_path}")

def classify_image(image_path):
    try:
        img_array, original_img = get_img_array(image_path)
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded):
            print(f"{i + 1}: {label} ({score:.2f})")

        top_pred_index = np.argmax(predictions[0])
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=top_pred_index)
        display_gradcam(image_path, heatmap, original_img.astype(np.uint8))

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "kami.png"
    classify_image(image_path)
