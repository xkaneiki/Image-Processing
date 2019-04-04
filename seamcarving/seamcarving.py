import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import convolve as _cov


class SeamCarving:
    def __init__(self):
        # self.img = img
        # self.src = np.array(img, "float32")
        self.dy = [
            [-1, -1],
            [0, -1],
            [1, -1]
        ]
        self.dx = [
            [-1, -1],
            [-1, 0],
            [-1, 1]
        ]
        # sobel
        self.filter_x = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])
        self.filter_y = np.array([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ])

    def convovle(self, src, filter_):
        res = np.zeros(src.shape, dtype=np.float)
        w, h = filter_.shape
        W, H = src.shape
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                tmp = np.zeros([w, h], dtype=np.float)
                src[i-w//2:i+w//2+1, j-h//2:j+h//2+1]
                dw1 = min(w//2+1, W-i)
                dw0 = min(w//2, i)
                dh1 = min(h//2+1, H-j)
                dh0 = min(h//2, j)
                tmp[w//2-dw0:w//2+dw1, h//2-dh0:h//2 +
                    dh1] = src[i-dw0:i+dw1, j-dh0:j+dh1]
                res[i][j] = np.sum(tmp*filter_)
        return res

    def get_M(self, e):
        M = e.copy()
        w, h = e.shape
        v1 = np.zeros(h,np.float)
        v1[0]=float('inf')
        v2 = np.zeros(h,np.float)
        v2[h-1]=float('inf')
        for i in range(1, w):
            v1[1:h]  = M[i-1, 0:h-1]
            v2[0:h-1] = M[i-1, 1: h]
            M[i] = np.min(np.stack([v1, v2, M[i-1]], axis=1), axis=1)+M[i]

        return M

    def get_M_(self, e):
        M = e.copy()
        w, h = e.shape
        dirt = np.zeros((w, h), dtype=np.int)
        for i in range(1, w):
            for j in range(h):
                t = float('inf')
                for _ in range(3):
                    if i+self.dx[_][0] >= 0 and i+self.dx[_][0] < w and j+self.dx[_][1] >= 0 and j+self.dx[_][1] < h:
                        p = M[i+self.dx[_][0]][j+self.dx[_][1]]+e[i][j]
                        if p < t:
                            t = p
                            dirt[i][j] = _
                M[i][j] = t
        return M

    def carve_row(self, src):
        pt = time.time()

        e = self._energy(src)

        nt = time.time()
        print(nt-pt, "s")
        pt = nt

        w, h = e.shape
        M = self.get_M(e)
        # print(M)
        pos = np.argmin(M[w-1])
        # print('pos', pos)

        nt = time.time()
        print(nt-pt, "s")
        pt = nt

        B = np.ones(e.shape, dtype=np.bool)

        # print(dir)
        for i in reversed(range(w)):
            # print('pos', pos)
            B[i][pos] = False
            if i == 0:
                break
            if pos-1 >= 0 and (int)(M[i-1][pos-1]) == int(M[i][pos]-e[i][pos]):
                pos = pos-1
            elif pos+1 < h and (int)(M[i-1][pos+1]) == int(M[i][pos]-e[i][pos]):
                pos = pos+1

        B = np.stack([B]*3, axis=2)

        img = src[B].reshape((w, h-1, 3))

        nt = time.time()
        print(nt-pt, "s")
        pt = nt

        return img

    def carve_col(self, src):
        src1 = np.stack([
            src[:, :, 0].T,
            src[:, :, 1].T,
            src[:, :, 2].T
        ], axis=2)
        img = self.carve_row(src1)
        img = np.stack([
            img[:, :, 0].T,
            img[:, :, 1].T,
            img[:, :, 2].T
        ], axis=2)
        return img

    def carve(self, src, len, Axis):
        img = np.array(src, dtype=np.float)
        # print(img)
        if Axis == 'x':
            for _ in range(len):
                # print(_)
                img = self.carve_row(img)

        elif Axis == 'y':
            for _ in range(len):
                # print(_)
                img = self.carve_col(img)

        return img.astype(np.uint8)

    def energy(self, src):
        W, H, Z = src.shape
        e = np.stack([np.abs(_cov(src[:, :, 0], self.filter_x)) +
                      np.abs(_cov(src[:, :, 0], self.filter_y)),
                      np.abs(_cov(src[:, :, 1], self.filter_x)) +
                      np.abs(_cov(src[:, :, 1], self.filter_y)),
                      np.abs(_cov(src[:, :, 2], self.filter_x)) +
                      np.abs(_cov(src[:, :, 2], self.filter_y))
                      ], axis=2)

        e = np.sum(e, axis=2)
        # self.show(e, "carve")
        return e

    # def show(self, src, title):
    #     cv2.imshow(title, src.astype(np.uint8))

    def _energy(self, src):
        W, H, Z = src.shape
        filter_x = np.stack([self.filter_x]*3, axis=2)
        filter_y = np.stack([self.filter_y]*3, axis=2)

        e = np.absolute(_cov(src, filter_x))+np.absolute(_cov(src, filter_y))
        e = np.sum(e, axis=2)
        return e


if __name__ == "__main__":
    img = cv2.imread(
        "/Users/kaneiki/Desktop/Image_Processing/imgs/carving.png")
    # cv2.imshow("raw", np.array(img))
    # src=np.array(img,np.uint8)
    print(img.shape)

    test = SeamCarving()

    stat = time.time()
    res = test.carve(img, 500, 'x')
    print(time.time()-stat, "s")

    # res = test.carve(img, 10, 'y')
    print(res.shape)
    # cv2.imshow("res", res)

    t = np.hstack([res, img])
    print(t.shape)
    cv2.imshow("windows", t)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
