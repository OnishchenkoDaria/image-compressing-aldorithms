import numpy as np
import cv2

def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0


def load_image_grayscale(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(">>> Error: Image not found.")
        return None

    img = img.astype(np.float32)
    h, w = img.shape

    # check if dimensions are powers of two
    if not is_power_of_two(h) or not is_power_of_two(w):
        print(f">>> Error: Image size must be power of two (e.g., 512x512). Got {h}x{w}.")
        return None

    # check if / by 8 (DCT block size)
    if h % 8 != 0 or w % 8 != 0:
        print(f">>> Error: Image size must be divisible by 8 for DCT blocks. Got {h}x{w}.")
        return None

    return img

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

# adding function for dividing into 8*8 matrix for haar even though uneven default size
def pad_image_to_blocksize(img, block_size=8):
    h, w = img.shape
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size

    padded = np.zeros((new_h, new_w), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w

# restore original size
def unpad_image(img, original_h, original_w):
    return img[:original_h, :original_w]