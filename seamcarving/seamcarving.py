import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


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

    def carve_row(self, src):
        e = self.energy(src)
        M = e.copy()
        w, h = e.shape
        dirt = np.zeros((w, h),dtype=np.int)
        for i in range(1, w):
            for j in range(h):
                t = 987654321
                for _ in range(3):
                    if i+self.dx[_][0] >= 0 and i+self.dx[_][0] < w and j+self.dx[_][1] >= 0 and j+self.dx[_][1] < h:
                        p = M[i+self.dx[_][0]][j+self.dx[_][1]]+e[i][j]
                        if p < t:
                            t = p
                            dirt[i][j] = _
                M[i][j] = t
        pos = np.argmin(M[w-1])
        print('pos',pos)

        B = np.ones(e.shape, dtype=np.bool)
        
        # print(dir)
        for i in reversed(range(w)):
            B[i][pos] = False
            pos = pos+self.dx[dirt[i][pos]][1]

        B = np.stack([B]*3, axis=2)

        img = src[B].reshape((w, h-1, 3))

        return img

    def carve_col(self, src):
        pass

    def carve(self, src, len, Axis):
        img = np.array(src, dtype=np.float)
        print(img)
        if Axis == 'x':
            for _ in range(len):
                img = self.carve_row(img)

        elif Axis == 'y':
            for _ in range(len):
                img = self.carve_row(img)
        
        return img.astype(np.uint8)

    def energy(self, src):
        W, H, Z = src.shape
        e = np.stack([np.abs(self.convovle(src[:, :, 0], self.filter_x)) +
                      np.abs(self.convovle(src[:, :, 0], self.filter_y)),
                      np.abs(self.convovle(src[:, :, 1], self.filter_x)) +
                      np.abs(self.convovle(src[:, :, 1], self.filter_y)),
                      np.abs(self.convovle(src[:, :, 2], self.filter_x)) +
                      np.abs(self.convovle(src[:, :, 2], self.filter_y))
                      ], axis=2)

        e = np.sum(e, axis=2)
        # self.show(e, "carve")
        return e

    # def show(self, src, title):
    #     cv2.imshow(title, src.astype(np.uint8))


if __name__ == "__main__":
    img = cv2.imread(
        "/Users/kaneiki/Desktop/Image_Processing/imgs/carving.png")
    cv2.imshow("raw", img)
    print(img.shape)

    test = SeamCarving()
    res = test.carve(img, 100, 'x')
    print(res.shape)
    cv2.imshow("res", res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
