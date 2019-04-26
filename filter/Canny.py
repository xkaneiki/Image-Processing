import cv2
import numpy as np

# canny算子
img = cv2.imread('/Users/kaneiki/Desktop/Image_Processing/imgs/haze.png')
# print(img)
# cv2.imshow("canny",img)
blur = cv2.GaussianBlur(img, (3, 3), 0.1)  

canny = cv2.Canny(blur, 0, 30) 
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


