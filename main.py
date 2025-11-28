import os

from utils import load_image_grayscale, save_image
from dct_compressor import DCTCompressor
from dwt_haar_compressor import DWTHaarCompressor
from evaluation import compare_methods
from plotting import visualize_results, visualize_dct_coefficients, visualize_haar_coefficients

for image in os.listdir("./source_images"):
    print("Image:", image)
    img = load_image_grayscale(f"./source_images/{image}")
    print(img)
    if img is not None:
        # for plotting
        h, w = img.shape

        # run dct
        dct = DCTCompressor(block_size=8, keep_ratio=0.5)
        dct_compressed = dct.compress(img)
        dct_reconstructed = dct.decompress(dct_compressed)
        save_image(f"./results/output_dct_{image}.png", dct_reconstructed)

        # run dwt
        dwt = DWTHaarCompressor(level=3, threshold=10)
        dwt_compressed = dwt.compress(img)
        dwt_reconstructed = dwt.decompress(dwt_compressed)
        save_image(f"./results/output_dwt_{image}.png", dwt_reconstructed)

        # comparing
        results = compare_methods(img, dct_reconstructed, dwt_reconstructed)

        print("\n=== PSNR RESULTS ===")
        print("DCT PSNR:", results["DCT_PSNR"])
        print("DWT PSNR:", results["DWT_PSNR"], "\n")

        # in between process visualising
        visualize_dct_coefficients(dct_compressed, title="DCT Coefficients (after thresholding)")
        visualize_haar_coefficients(dwt_compressed, title="Haar Coefficients (after thresholding)")

        # visualize side by side
        visualize_results(img, dct_reconstructed, dwt_reconstructed, results, image)