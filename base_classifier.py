import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

import numpy as np
import cv2
import os

model = MobileNetV2(weights="imagenet")
LAST_CONV = "Conv_1"
IMG_SIZE  = (224, 224)

def get_img_arrays(img_path, size=IMG_SIZE):
    pil = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(pil)
    batch = np.expand_dims(arr, axis=0)
    return preprocess_input(batch), arr.astype(np.uint8)

def gradcam_heatmap(batch_img, class_idx=None):
    grad_model = Model(inputs=model.input,
                       outputs=[model.get_layer(LAST_CONV).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch_img)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        class_score = preds[:, class_idx]

    grads  = tape.gradient(class_score, conv_out)
    w      = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam    = tf.reduce_sum(tf.multiply(w, conv_out[0]), axis=-1)

    cam    = np.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    cam    = cam.numpy()
    return cam, preds.numpy()

def mask_from_cam(cam, orig_shape, thresh=0.6):
    cam_rs = cv2.resize(cam, (orig_shape[1], orig_shape[0]))
    return cam_rs >= (thresh * cam_rs.max())

def occlude_black(img, mask):
    occl = img.copy()
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    cv2.rectangle(occl, (x, y), (x+w, y+h), (0, 0, 0), thickness=-1)
    return occl

def occlude_blur(img, mask, ksize=51):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    occl = img.copy()
    occl[mask] = blurred[mask]
    return occl

def occlude_noise(img, mask):
    noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
    occl  = img.copy()
    occl[mask] = noise[mask]
    return occl

def save_and_classify(img_arr, fname):
    cv2.imwrite(fname, cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
    batch = preprocess_input(np.expand_dims(img_arr.astype(np.float32), 0))
    preds = model.predict(batch, verbose=0)
    top3  = decode_predictions(preds, top=3)[0]
    print(f"\n🖼  Results for {fname}:")
    for i, (_, lbl, sc) in enumerate(top3, 1):
        print(f"   {i}: {lbl} ({sc:.2f})")

def run_pipeline(img_path):
    print(f"👉 Classifying original image: {img_path}")
    batch, orig = get_img_arrays(img_path)
    cam, preds  = gradcam_heatmap(batch)
    mask        = mask_from_cam(cam, orig.shape)

    # Original predictions
    for i, (_, lbl, sc) in enumerate(decode_predictions(preds, top=3)[0], 1):
        print(f"   {i}: {lbl} ({sc:.2f})")

    # Apply three occlusions guided by the Grad-CAM mask
    occlusions = {
        "occl_black.jpg": occlude_black(orig, mask),
        "occl_blur.jpg" : occlude_blur(orig, mask),
        "occl_noise.jpg": occlude_noise(orig, mask)
    }

    # Iterate, save, and re-classify
    for fname, occl_img in occlusions.items():
        save_and_classify(occl_img, fname)

if __name__ == "__main__":
    IMG_PATH = "kami.png"
    run_pipeline(IMG_PATH)
