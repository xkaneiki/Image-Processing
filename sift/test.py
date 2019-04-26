# -*- coding: utf-8 -*-

import cv2
import numpy as np

inputImgPath = '/Users/kaneiki/Desktop/Image_Processing/imgs/boy.jpeg' 
outputImgPath = '/Users/kaneiki/Desktop/Image_Processing/sift/ans.txt' 


# featureSun:计算特征点个数
featureSum = 0
img = cv2.imread(inputImgPath)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

detector = cv2.xfeatures2d.SIFT_create()
# 找到关键点
kps , des = detector.detectAndCompute(gray,None)
# 绘制关键点
img=cv2.drawKeypoints(gray,kps,img)

# 将特征点保存
np.savetxt(outputImgPath ,des ,  fmt='%.2f')
featureSum += len(kps)
cv2.imshow('result',img)
print('kps:' + str(featureSum))

