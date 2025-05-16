import cv2
import numpy as np

cap = cv2.VideoCapture("test.mp4")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(prev_frame)
hsv[..., 1] = 255  # 初始化HSV图像（用于可视化）

while True:
    ret, frame = cap.read()
    if not ret: break
    
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # 转换为极坐标（幅值和角度）
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # 色调表示方向
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 亮度表示速度
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imshow("Dense Optical Flow", bgr)
    if cv2.waitKey(30) == 27: break
    
    prev_gray = curr_gray.copy()

cap.release()
cv2.destroyAllWindows()