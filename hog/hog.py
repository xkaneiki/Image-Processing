import numpy as np
import cv2
from scipy.ndimage.filters import convolve
import math
from sklearn import svm
import os
import time
f_path = '/Users/kaneiki/Desktop/Image_Processing/hand_img/f'
g_path = '/Users/kaneiki/Desktop/Image_Processing/hand_img/g/'
w_path = '/Users/kaneiki/Desktop/Image_Processing/hand_img/w/'
l_path = '/Users/kaneiki/Desktop/Image_Processing/hand_img/l/'
others_path = "/Users/kaneiki/Desktop/Image_Processing/hand_img/others/"
test_f = "/Users/kaneiki/Desktop/Image_Processing/hand_img/data/f/"
test_g = "/Users/kaneiki/Desktop/Image_Processing/hand_img/g/"
# test_l = "/Users/kaneiki/Desktop/Image_Processing/hand_img/data/l/"
cn = 20


def Gx(src):
    f = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], np.float)
    return convolve(src, f)


def Gy(src):
    f = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], np.float)
    return convolve(src, f)


def togray(src):
    return src[:, :, 0]*0.299+src[:, :, 1]*0.587+src[:, :, 0]*0.144


def Gamma(src, gamma):
    return np.exp(np.log(src/np.max(src))*gamma)*255


def gaussfilter(src, size, exps):
    PI = math.acos(-1)
    f = np.zeros((size, size), np.float)
    for i in range(size):
        for j in range(size):
            f[i][j] = (i-size//2)*(i-size//2)+(j-size//2)*(j-size//2)
    f = 1./(2.*PI*exps*exps)*np.exp(-f/(2.*exps*exps))
    return convolve(src, f)


def exact(angle, c_size, k_size, bin=9, sigma=0.1):
    h, w = c_size
    H, W = angle.shape
    ht = np.zeros((H//h, W//w, bin), np.float)

    for i in range(H//h):
        for j in range(W//w):
            t, _ = np.histogram(angle[i*h:i*h+h, j*w:j*w+w], bin, (-180, 180))
            ht[i][j] = t

    # print(ht)

    H = H//h
    W = W//w
    h, w = k_size
    l = h*w*bin
    res = np.zeros((H//h, W//w, l), np.float)
    for i in range(H//h):
        for j in range(W//w):
            res[i][j] = normal(np.reshape(
                ht[i*h:i*h+h, j*w:j*w+w], (l)), sigma)

    return np.reshape(res, (1, res.shape[0]*res.shape[1]*res.shape[2]))


def normal(src, sigma):
    v = src/np.sqrt(np.dot(src.T,src)+sigma*sigma)
    return v


def get_hog(img):
    PI = math.acos(-1)
    src = np.array(img, np.float)
    # 1.
    src = Gamma(src, 1)
    src = togray(src)
    src = gaussfilter(src, 7, 1.)
    # cv2.imshow("show", src.astype(np.uint8))

    # 2.
    gx = Gx(src)
    gy = Gy(src)
    angle = 1/PI*180*np.arctan(gy/gx)
    # print(angle)

    hog = exact(angle, (8, 8), (2, 2))
    return hog


def get_data(path, cl):
    c = 0
    X = 0
    for filename in os.listdir(path):
        if c >= cn:
            break
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            hog = get_hog(img)
            if c == 0:
                X = hog
            else:
                X = np.vstack([
                    X, hog
                ])
            c += 1
            # print(c)
    Y = np.ones((cn), np.float)*cl
    return X, Y


if __name__ == "__main__":
    # img = cv2.imread("/Users/kaneiki/Desktop/Image_Processing/hand_img/f/1.jpg")
    # hog = exact(angle, (8, 8), (2, 2))
    # hg=get_hog(img)
    # print(np.sum(hg))
    st = time.time()
    X, Y = get_data(f_path, 0)
    x, y = get_data(g_path, 1)
    X = np.vstack([X, x])
    Y = np.hstack([Y, y])
    # x,y=get_data(w_path,2)
    # X=np.vstack([X,x])
    # Y=np.hstack([Y,y])
    # x,y=get_data(l_path,-1)
    # X=np.vstack([X,x])
    # Y=np.hstack([Y,y])
    # x,y=get_data(w_path,4)
    # X=np.vstack([X,x])
    # Y=np.hstack([Y,y])
    print("X", X.shape)
    print("Y", Y.shape)
    clf = svm.SVC()
    clf.fit(X, Y)

    X_t, Y_t = get_data(test_f, 0)
    
    x, y = get_data(test_g, 1)
    X_t = np.vstack([X_t, x])
    Y_t = np.hstack([Y_t, y])
    
    # x,y=get_data(test_l,-1)
    # X_t=np.vstack([X_t,x])
    # Y_t=np.hstack([Y_t,y])

    Y_ = clf.predict(X_t)
    # print(Y_t)
    # print(Y_)
    t=Y_t-Y_
    t=np.where(t==0,1,0)
    d=1
    for i in t.shape:
        d*=i
    print(100*np.sum(t)/d,"%")
    print(time.time()-st, "s")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
