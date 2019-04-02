import cv2
import numpy as np

# canny算子
img = cv2.imread('/Users/kaneiki/Desktop/DeepLearning/picture/hand.jpg')
# print(img)
# cv2.imshow("canny",img)
blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
canny = cv2.Canny(blur, 0, 30)  # 50是最小阈值,150是最大阈值
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


