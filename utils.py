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

def blockify(img, block_size=8):
    h, w = img.shape
    blocks = img.reshape(h // block_size, block_size,
                         w // block_size, block_size)
    return blocks.swapaxes(1, 2)

def unblockify(blocks, block_size=8):
    blocks = blocks.swapaxes(1, 2)
    h_blocks, w_blocks, _, _ = blocks.shape
    return blocks.reshape(h_blocks * block_size,
                          w_blocks * block_size)

def threshold_coeffs(coeffs: np.ndarray, threshold: float):
    return np.where(np.abs(coeffs) < threshold, 0, coeffs)

def mse(original: np.ndarray, reconstructed: np.ndarray):
    return np.mean((original - reconstructed) ** 2)

def psnr(original: np.ndarray, reconstructed: np.ndarray):
    m = mse(original, reconstructed)
    if m == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / m)