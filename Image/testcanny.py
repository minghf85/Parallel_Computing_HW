import cv2
import numpy as np

# 读取图像（自动转为灰度）
img = cv2.imread('./img/gray.jpg', cv2.IMREAD_GRAYSCALE)  

# 基本Canny检测
edges = cv2.Canny(img, threshold1=20, threshold2=150, L2gradient=True)
cv2.imwrite('./img/canny2.jpg', edges)
# 显示结果
cv2.imshow('Original', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()