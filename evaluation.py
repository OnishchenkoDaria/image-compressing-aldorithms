from utils import psnr

def compare_methods(original, rec_dct, rec_dwt):
    psnr_dct = psnr(original, rec_dct)
    psnr_dwt = psnr(original, rec_dwt)

    return {
        "DCT_PSNR": psnr_dct,
        "DWT_PSNR": psnr_dwt,
    }