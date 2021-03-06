# 图像去雾
import cv2
import numpy as np
from scipy.ndimage.filters import convolve


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


def get_dark(src, r=1):
    t = np.min(src, axis=2)
    w, h = t.shape
    v = t.copy()
    for i in range(r+1):
        v[0:w-i, 0:h -
            i] = np.min(np.stack([v[0:w-i, 0:h-i], t[i:w, i:h]], axis=2), axis=2)
        v[i:w, i:h] = np.min(
            np.stack([v[i:w, i:h], t[0:w-i, 0:h-i]], axis=2), axis=2)
    # res = guidefilter(t, v, r, 0.001)
    return v



def dehaze(src, size, w=0.95):
    img = np.array(src, dtype=np.float)
    print("img", img)

    A0 = np.max(img[:, :, 0])
    A1 = np.max(img[:, :, 1])
    A2 = np.max(img[:, :, 2])

    dark = get_dark(img, size//2)
    print("dark", dark)

    g = togray(src)
    # dark = guidefilter(dark, g.astype(np.float)/255.0, 3, 255*255*0.02)
    # cv2.imshow("dark",dark.astype(np.uint8))
    # A=np.max(dark)
    ia = img.copy()
    ia[:, :, 0] = ia[:, :, 0]/A0
    ia[:, :, 1] = ia[:, :, 1]/A1
    ia[:, :, 1] = ia[:, :, 2]/A2

    # tmp=guidefilter(get_dark(ia,size//2),g,3,0.02)
    tmp = get_dark(ia, size//2)
    # tmp=guidefilter(tmp, g.astype(np.float)/255.0, 3, 255*255*0.02)
    t = 1 - w*tmp
    t = np.where(t < 0.1, 0.1, t)
    print(t.shape)
    print("t", t)

    res = np.zeros(img.shape)
    res[:, :, 0] = (img[:, :, 0]-A0)/t+A0
    res[:, :, 1] = (img[:, :, 1]-A1)/t+A1
    res[:, :, 2] = (img[:, :, 2]-A2)/t+A2
    print("res", res)
    res = res.astype(np.uint8)

    return res


img = cv2.imread("/Users/kaneiki/Desktop/Image_Processing/imgs/haze3.png")
img_ans = dehaze(img, 3, 0.8)

show = np.hstack([img, img_ans])

cv2.imshow("img_ans", show)

cv2.waitKey(0)
cv2.destroyAllWindows()
