import numpy as np
import cv2
from scipy.ndimage.filters import convolve as conv
import time

def convovle(src,conv_filter):
    h,w=conv_filter.shape
    H,W=src.shape
    res=np.zeros((H,W),np.float)
    #注意负数向下取整-3//2==-2 
    for i in range(-(h//2),h//2+1):
        for j in range(-(w//2),w//2+1):
            res[max(0,-i):min(H,H-i),max(0,-j):min(W,W-j)]+=conv_filter[i+h//2][j+w//2]*src[max(0,i):min(H,H+i),max(0,j):min(W,W+j)]
            # if i <=0 and j <=0:
            #     res[-i:H,-j:W]+=conv_filter[i+h//2,j+w//2]*src[0:H+i,0:W+j]
            # elif i >=0 and j<=0:
            #     res[0:H-i,-j:W]+=conv_filter[i+h//2,j+w//2]*src[i:H,0:W+j]
            # elif i<=0 and j>=0:
            #     res[-i:H,0:W-j]+=conv_filter[i+h//2,j+w//2]*src[0:H+i,j:W]
            # else:
            #     res[0:H-i,0:W-j]+=conv_filter[i+h//2,j+w//2]*src[i:H,j:W]
    return res

def convovle_(src,conv_filter):
    h,w=conv_filter.shape
    H,W=src.shape
    res=np.zeros((H,W),np.float)
    for i in range(H):
        for j in range(W):
            res[i][j]=np.sum(
                src[max(0,i-h//2):min(H,i+h//2+1),max(0,j-w//2):min(W,j+w//2+1)]*
                conv_filter[h//2-(i-max(0,i-h//2)):h//2+(min(H,i+h//2+1)-i),w//2-(j-max(0,j-w//2)):w//2+(min(W,j+w//2+1)-j)]
            )
    return res

if __name__=='__main__':
    # src=np.array(
    #     cv2.imread("imgs/haze1.jpg")[:,:,0]
    # ,np.float)
    src=np.array([
        [10,10,10,10,10,10],
        [11,11,11,11,11,11],
        [12,12,12,12,12,12],
        [13,13,13,13,13,13]
    ],np.float)

    f=np.array([
        [1,2,1],
        [1,3,1],
        [1,2,3],
    ],np.float)
    s=time.time()

    t=convovle(src,f)
    print(t)
    s1=time.time()
    print(s1-s)

    t1=conv(src,f)
    s2=time.time()
    print(t1)
    print(s2-s1)

    t2=convovle_(src,f)
    s3=time.time()
    print(t2)
    print(s3-s2)