import matplotlib.pyplot as plt

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