import cv2
import numpy as np
from scipy.ndimage.filters import convolve

img = cv2.imread("/Users/kaneiki/Desktop/image_processing/imgs/carving.png")


def get_dark(src,  r=1):
    t = np.min(src, axis=2)
    w, h = t.shape
    res = t.copy()
    for i in range(r+1):
        res[0:w-i, 0:h -
            i] = np.min(np.stack([res[0:w-i, 0:h-i], t[i:w, i:h]], axis=2), axis=2)
        res[i:w, i:h] = np.min(
            np.stack([res[i:w, i:h], t[0:w-i, 0:h-i]], axis=2), axis=2)

    return res


def togray(src):
    img = np.array(src, np.float)
    res = (img[:, :, 0]*0.299+img[:, :, 1]*0.587+img[:, :, 2]*0.144)
    return res.astype(np.uint8)


def mean(src, size):
    filter_ = np.zeros(size)+1.0/(size[0]*size[1])
    return convolve(src, filter_)


def guidefilter(I, P, r, exps):
    w = (2*r+1, 2*r+1)
    meani = mean(I, w)
    meanp = mean(P, w)

    corri = mean(I*I, w)
    corrip = mean(I*P, w)

    vari = corri-meani*meani
    covip = corrip-meani*meanp

    a = covip/(vari+exps)
    b = meanp-a*meani

    meana = mean(a, w)
    meanb = mean(b, w)
    q = meana*I+meanb

    return q


g = togray(img)
shape = (5, 5)
img1 = cv2.blur(g, shape)
img2 = mean(np.array(g, np.float), shape)
dark = get_dark(np.array(img, np.float), 2)

img3 = guidefilter(img2, g, 2, 200)

imgs = np.hstack([img2.astype(np.uint8), img3.astype(np.uint8)])
cv2.imshow("show", imgs)

cv2.waitKey(0)
cv2.destroyAllWindows()
