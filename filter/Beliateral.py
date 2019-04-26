import numpy as np
import cv2

img=cv2.imread("/Users/kaneiki/Desktop/Image_Processing/imgs/carving.png")
img1=cv2.bilateralFilter(img,5,0.8,0.8)
cv2.imshow("raw",np.hstack([img,img1]))
cv2.waitKey(0)
cv2.destroyAllWindows()