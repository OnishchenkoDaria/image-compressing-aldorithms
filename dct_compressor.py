import numpy as np
from scipy.fftpack import dct, idct
from utils import blockify, unblockify

# JPEG-like quantization matrix
Q_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float32)


class DCTCompressor:
    def __init__(self, block_size=8, keep_ratio=0.5):
        self.block_size = block_size
        self.keep_ratio = keep_ratio   # portion of low frequencies kept

    def compress(self, img):
        blocks = blockify(img, self.block_size)
        h, w, _, _ = blocks.shape

        dct_blocks = np.zeros_like(blocks)

        # DCT each block
        for i in range(h):
            for j in range(w):
                B = blocks[i, j]

                # 2D DCT
                C = dct(dct(B.T, norm='ortho').T, norm='ortho')

                # quantization
                Cq = np.round(C / Q_TABLE)

                # keep only low frequencies
                limit = int(self.block_size * self.keep_ratio)
                mask = np.zeros_like(Cq)
                mask[:limit, :limit] = 1
                Cq *= mask

                dct_blocks[i, j] = Cq

        return dct_blocks

    def decompress(self, dct_blocks):
        h, w, _, _ = dct_blocks.shape
        rec_blocks = np.zeros_like(dct_blocks)

        for i in range(h):
            for j in range(w):
                Cq = dct_blocks[i, j]

                # dequantization
                C = Cq * Q_TABLE

                # inverse DCT
                B = idct(idct(C.T, norm='ortho').T, norm='ortho')

                rec_blocks[i, j] = B

        return unblockify(rec_blocks, self.block_size)