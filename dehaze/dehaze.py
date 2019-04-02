# 图像去雾
import cv2
import numpy as np


def dehaze(src, size, w=0.95):
    img = np.array(src, dtype=np.float)
    print("img", img)

    A = np.max(img)
    # A = 255
    tmp = np.min(img, axis=2)

    W = img.shape[0]
    H = img.shape[1]
    dark = tmp.copy()

    for x in range(W):
        for y in range(H):
            for i in range(x-(int)(size/2), x+(int)(size/2)+1, 1):
                for j in range(y-(int)(size/2), y+(int)(size/2)+1, 1):
                    if i >= 0 and i < W and j >= 0 and j < H:
                        dark[x][y] = min(dark[x][y], tmp[i][j])
    print('dark', dark)

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


img = cv2.imread("/Users/kaneiki/Desktop/DeepLearning/picture/haze2.png")
img_ans = dehaze(img, 3, 0.8)

show = np.hstack([img, img_ans])

cv2.imshow("img_ans", show)

cv2.waitKey(0)
cv2.destroyAllWindows()
