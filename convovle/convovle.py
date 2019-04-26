import numpy as np
import cv2
from scipy.ndimage.filters import convolve as _conv

def convovle(src, filter_):
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



if __name__ == "__main__":
    # img=cv2.imread("")
    # src=np.array(img,dtype=np.float)
    # res=np.zeros(src.shape)
    filter_ = np.array([
        [1, 2, 1],
        [2, 3, 1],
        [1, 2, 1]
    ], dtype=np.float)
    test = np.array([
        [[1,2], [1,2], [1,2], [1,2]],
        [[1,2], [1,2], [1,2], [1,2]],
        [[1,2], [1,2], [1,2], [1,2]],
        [[1,2], [1,2], [1,2], [1,2]],
    ], dtype=np.float)
    e0=np.abs(convovle(test[:,:,0], filter_))
    e1=np.abs(convovle(test[:,:,1], filter_))
    
    print(e0)
    print(np.abs(_conv(test[:,:,0], filter_)))
    
    # print(np.stack([e0,e1],axis=2))
    # test[0:2,0:2]=np.array([
    #     [2,2],
    #     [2,2],
    # ])
    # print(test)
