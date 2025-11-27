import pywt
import numpy as np
from utils import threshold_coeffs

class DWTHaarCompressor:
    def __init__(self, level=3, threshold=10):
        self.level = level
        self.threshold = threshold

    def compress(self, img):
        # multi-level Haar DWT
        coeffs = pywt.wavedec2(img, 'haar', level=self.level)

        # thresholding of detail coefficients
        cA = coeffs[0]  # LL block remains unchanged
        cDs = []

        for (cH, cV, cD) in coeffs[1:]:
            cH = threshold_coeffs(cH, self.threshold)
            cV = threshold_coeffs(cV, self.threshold)
            cD = threshold_coeffs(cD, self.threshold)
            cDs.append((cH, cV, cD))

        return (cA, cDs)

    def decompress(self, compressed_data):
        cA, cDs = compressed_data
        coeffs = [cA] + cDs
        return pywt.waverec2(coeffs, 'haar')