import cv2
import numpy as np

# 拉普拉斯算子
img = cv2.imread('/Users/kaneiki/Desktop/DeepLearning/picture/hand.jpg')
blur = cv2.GaussianBlur(img, (5, 5), 0)
laplacian = cv2.Laplacian(blur, cv2.CV_16S, ksize=5)
dst = cv2.convertScaleAbs(laplacian)


cv2.imshow('laplacian', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

