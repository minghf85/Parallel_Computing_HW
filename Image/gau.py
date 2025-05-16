import cv2
import numpy as np

# 读取图像
img = cv2.imread('./img/test.jpg')

# 高斯滤波基本用法
blurred = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite('./img/test_blurred.jpg', blurred)
# 显示结果
cv2.imshow('Original', img)
cv2.imshow('Gaussian Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()