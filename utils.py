import numpy as np
import cv2

def load_image_grayscale(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found.")
    return img.astype(np.float32)

def save_image(path: str, img: np.ndarray):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)

