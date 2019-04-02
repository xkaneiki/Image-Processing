import cv2
import numpy as np

img=cv2.imread("/Users/kaneiki/Desktop/DeepLearning/picture/boy.jpeg")

img=cv2.resize(img,(480,640))

cv2.imshow("raw",img)

img_bila=cv2.bilateralFilter(img,3,75,75)
cv2.imshow("bila",img_bila)

img_median=cv2.medianBlur(img,5)
# cv2.imshow("media",img_median);

img_blur=cv2.blur(img,(5,5))
# cv2.imshow("blur",img_blur)

img_gauss=cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow("Gauess",img_gauss)

img_border=img-img_gauss
cv2.imshow("border",img_border)

img_canny = cv2.Canny(img_gauss, 50, 150)  # 50是最小阈值,150是最大阈值
cv2.imshow("canny",img_canny)

cv2.waitKey(0)
cv2.destroyAllWindows()