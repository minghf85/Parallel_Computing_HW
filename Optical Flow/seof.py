import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

CONSISTENT_FLOW_WINDOW_SIZE = 5

def serial_optical_flow(frame1, frame2):
    # Lucas-Kanade光流法
    # 无优化版本
    # 确保输入是灰度图
    if len(frame1.shape) > 2:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if len(frame2.shape) > 2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 创建空白图像保存光流结果
    flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
    
    # 计算图像的x和y方向的梯度以及时间方向的梯度
    fx = cv2.Sobel(frame1, cv2.CV_32F, 1, 0)
    fy = cv2.Sobel(frame1, cv2.CV_32F, 0, 1)
    ft = frame2.astype(np.float32) - frame1.astype(np.float32)
    
    h, w = frame1.shape
    half_window = CONSISTENT_FLOW_WINDOW_SIZE // 2
    
    # 对每个像素使用Lucas-Kanade方法计算光流
    for i in range(half_window, h - half_window):
        for j in range(half_window, w - half_window):
            # 获取当前窗口内的梯度
            Ix = fx[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            Iy = fy[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            It = ft[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
            
            # 构建A矩阵和b向量
            A = np.vstack((Ix, Iy)).T
            b = -It
            
            # 使用最小二乘法求解Av=b
            ATA = np.matmul(A.T, A)
            if np.linalg.det(ATA) != 0:
                ATb = np.matmul(A.T, b)
                v = np.linalg.solve(ATA, ATb)
                flow[i, j, 0] = v[0]
                flow[i, j, 1] = v[1]
    
    return flow

def opencv_lucas_kanade(frame1, frame2):
    # 确保输入是灰度图
    if len(frame1.shape) > 2:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
    
    if len(frame2.shape) > 2:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    
    # 创建特征点
    h, w = gray1.shape
    step = 16  # 采样步长，与plot_optical_flow中相同
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(np.float32)
    points = np.vstack((x, y)).T.reshape(-1, 1, 2)
    
    # OpenCV's Lucas-Kanade optical flow
    nextPts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, points, None,
        winSize=(CONSISTENT_FLOW_WINDOW_SIZE, CONSISTENT_FLOW_WINDOW_SIZE),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # 创建稀疏流场
    flow = np.zeros((h, w, 2), dtype=np.float32)
    
    # 提取有效的光流向量
    good_points = points[status == 1].reshape(-1, 2)
    good_next = nextPts[status == 1].reshape(-1, 2)
    
    # 计算位移
    displacements = good_next - good_points
    
    # 根据特征点位置填充流场
    for i, (point, displacement) in enumerate(zip(good_points, displacements)):
        x, y = int(point[0]), int(point[1])
        if 0 <= y < h and 0 <= x < w:
            flow[y, x] = displacement
    
    return flow, good_points, displacements

def plot_optical_flow(flow, step=16):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    # 创建一个新的图像
    plt.figure(figsize=(12, 10))
    plt.title('Optical Flow')
    plt.axis('off')
    
    # 绘制箭头
    plt.quiver(x, y, fx, fy, color='b', angles='xy', scale_units='xy', scale=0.25)
    
    # 添加背景网格
    plt.grid()
    plt.show()

def plot_optical_flow_comparison(custom_flow, opencv_points, opencv_displacements, frame1, step=16):
    h, w = custom_flow.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = custom_flow[y, x].T
    
    plt.figure(figsize=(18, 8))
    
    # 左侧：自定义方法
    plt.subplot(1, 2, 1)
    plt.title('Custom Serial Lucas-Kanade')
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.quiver(x, y, fx, fy, color='b', angles='xy', scale_units='xy', scale=0.25)
    plt.axis('off')
    
    # 右侧：OpenCV方法
    plt.subplot(1, 2, 2)
    plt.title('OpenCV Lucas-Kanade')
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.quiver(opencv_points[:, 0], opencv_points[:, 1], 
               opencv_displacements[:, 0], opencv_displacements[:, 1], 
               color='r', angles='xy', scale_units='xy', scale=0.25)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取两帧图片
    frame1 = cv2.imread("data/f11.jpg")
    frame2 = cv2.imread("data/f12.jpg")
    # 下采样到宽为512
    frame1 = cv2.resize(frame1, (512, int(512 * frame1.shape[0] / frame1.shape[1])))
    frame2 = cv2.resize(frame2, (512, int(512 * frame2.shape[0] / frame2.shape[1])))
    if frame1 is None or frame2 is None:
        print("Error loading images. Please check the file paths.")
    else:
        # 计算自定义串行光流
        start_time = time.time()
        custom_flow = serial_optical_flow(frame1, frame2)
        custom_end_time = time.time()
        custom_time = custom_end_time - start_time
        print(f"Custom serial optical flow calculation time: {custom_time:.4f} seconds")
        
        # 计算OpenCV光流
        start_time = time.time()
        opencv_flow, opencv_points, opencv_displacements = opencv_lucas_kanade(frame1, frame2)
        opencv_end_time = time.time()
        opencv_time = opencv_end_time - start_time
        print(f"OpenCV optical flow calculation time: {opencv_time:.4f} seconds")
        print(f"Speed improvement: {custom_time/opencv_time:.2f}x faster with OpenCV")
        
        # 可视化并比较两种方法的光流结果
        plot_optical_flow_comparison(custom_flow, opencv_points, opencv_displacements, frame1)