# 图像去雾
import cv2
import numpy as np


def get_dark(src, r=1):
    t = np.min(src, axis=2)
    w, h = t.shape
    res = t.copy()
    for i in range(r+1):
        res[0:w-i, 0:h -
            i] = np.min(np.stack([res[0:w-i, 0:h-i], t[i:w, i:h]], axis=2), axis=2)
        res[i:w, i:h] = np.min(
            np.stack([res[i:w, i:h], t[0:w-i, 0:h-i]], axis=2), axis=2)
    return res


def dehaze(src, size, w=0.95):
    img = np.array(src, dtype=np.float)
    print("img", img)

    A = np.max(img)
    dark = get_dark(img, size//2)
    print("dark", dark)

    t = 1 - w*dark/A
    t = np.where(t < 0.1, 0.1, t)
    print(t.shape)
    print("t", t)

    res = np.zeros(img.shape)
    res[:, :, 0] = (img[:, :, 0]-A)/t+A
    res[:, :, 1] = (img[:, :, 1]-A)/t+A
    res[:, :, 2] = (img[:, :, 2]-A)/t+A
    print("res", res)
    res = res.astype(np.uint8)

    return res


img = cv2.imread("/Users/kaneiki/Desktop/Image_Processing/imgs/haze1.jpg")
img_ans = dehaze(img, 3, 0.8)

show = np.hstack([img, img_ans])

cv2.imshow("img_ans", show)

cv2.waitKey(0)
cv2.destroyAllWindows()
