from utils import load_image_grayscale, save_image
from dct_compressor import DCTCompressor

def main():
    img = load_image_grayscale("sample.png")

    #run dct
    dct = DCTCompressor(block_size=8, keep_ratio=0.5)
    dct_compressed = dct.compress(img)
    dct_reconstructed = dct.decompress(dct_compressed)
    save_image("output_dct.png", dct_reconstructed)