import matplotlib.pyplot as plt
import numpy as np

def visualize_results(original, rec_dct, rec_dwt, psnr_results, image_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rec_dct, cmap='gray')
    plt.title(f"DCT\nPSNR: {psnr_results['DCT_PSNR']:.2f} dB")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(rec_dwt, cmap='gray')
    plt.title(f"DWT\nPSNR: {psnr_results['DWT_PSNR']:.2f} dB")
    plt.axis('off')

    plt.suptitle(f"Comparison for {image_name}")
    plt.tight_layout()
    plt.show()

def visualize_dct_coefficients(dct_blocks, block_size=8, title="DCT Coefficients"):
    # unite dct blocks for unite visuals
    h_blocks, w_blocks, _, _ = dct_blocks.shape
    coeff_image = np.zeros((h_blocks * block_size, w_blocks * block_size))
    for i in range(h_blocks):
        for j in range(w_blocks):
            coeff_image[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size] = dct_blocks[i, j]

    plt.figure(figsize=(6,6))
    plt.imshow(coeff_image, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

# haar post thresholding.
def visualize_haar_coefficients(compressed_data, title="Haar DWT Coefficients"):
    cA, cDs = compressed_data
    coeff_image = cA.copy()
    for (cH, cV, cD) in cDs:
        # adding details for showcase
        coeff_image = np.block([
            [coeff_image, np.abs(cH)],
            [np.abs(cV), np.abs(cD)]
        ])
    plt.figure(figsize=(6,6))
    plt.imshow(coeff_image, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()