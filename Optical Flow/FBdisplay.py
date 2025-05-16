import cv2
import numpy as np

# 初始化视频捕获
cap = cv2.VideoCapture("data/test.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(prev_frame)
hsv[..., 1] = 255  # 初始化HSV图像（饱和度固定为255）

# 可视化模式：0=HSV, 1=箭头, 2=轨迹, 3=掩码
display_mode = 0
pause = False

while True:
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算密集光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        prev_gray = curr_gray.copy()

        # 转换为极坐标（幅值和角度）
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # 色调表示方向
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 亮度表示速度
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 根据显示模式生成不同可视化结果
    if display_mode == 0:  # HSV颜色编码
        vis = bgr_flow
        title = "HSV Color Encoding (Press 1/2/3/4 to switch)"
    elif display_mode == 1:  # 箭头矢量场
        vis = frame.copy()
        step = 16  # 箭头网格步长
        h, w = flow.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        cv2.polylines(vis, lines, isClosed=False, color=(0, 255, 0), thickness=1)
        for (x1, y1), _ in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        title = "Arrow Vector Field (Press 1/2/3/4 to switch)"
    elif display_mode == 2:  # 运动轨迹叠加
        vis = frame.copy()
        mask = np.zeros_like(frame)
        mask[..., 1] = 255  # 绿色轨迹
        flow_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mask[..., 0] = flow_mag
        mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        vis = cv2.addWeighted(vis, 0.7, mask, 0.3, 0)
        title = "Motion Trails (Press 1/2/3/4 to switch)"
    elif display_mode == 3:  # 二值运动掩码
        _, mask = cv2.threshold(mag, 5, 255, cv2.THRESH_BINARY)  # 速度阈值=5
        mask = mask.astype(np.uint8)
        vis = cv2.bitwise_and(frame, frame, mask=mask)
        title = "Binary Motion Mask (Press 1/2/3/4 to switch)"

    # 显示结果和说明文字
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Optical Flow Visualization", vis)

    # 键盘控制
    key = cv2.waitKey(30 if not pause else 0) & 0xFF
    if key == 27:  # ESC退出
        break
    elif key == ord('1'):  # 切换HSV模式
        display_mode = 0
    elif key == ord('2'):  # 切换箭头模式
        display_mode = 1
    elif key == ord('3'):  # 切换轨迹模式
        display_mode = 2
    elif key == ord('4'):  # 切换掩码模式
        display_mode = 3
    elif key == ord(' '):  # 暂停/继续
        pause = not pause

cap.release()
cv2.destroyAllWindows()