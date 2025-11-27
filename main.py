from utils import load_image_grayscale, save_image
from dct_compressor import DCTCompressor
from dwt_haar_compressor import DWTHaarCompressor
from evaluation import compare_methods

img = load_image_grayscale("sample.png")

if(img):
    # run dct
    dct = DCTCompressor(block_size=8, keep_ratio=0.5)
    dct_compressed = dct.compress(img)
    dct_reconstructed = dct.decompress(dct_compressed)
    save_image("output_dct.png", dct_reconstructed)

    # run dwt
    dwt = DWTHaarCompressor(level=3, threshold=10)
    dwt_compressed = dwt.compress(img)
    dwt_reconstructed = dwt.decompress(dwt_compressed)
    save_image("output_dwt.png", dwt_reconstructed)

    # comparing
    results = compare_methods(img, dct_reconstructed, dwt_reconstructed)

    print("\n=== PSNR RESULTS ===")
    print("DCT PSNR:", results["DCT_PSNR"])
    print("DWT PSNR:", results["DWT_PSNR"])