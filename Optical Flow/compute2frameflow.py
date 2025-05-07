import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_optical_flow_with_timedelta(img1_path, img2_path, time_delta_sec, output_path=None, method='farneback', grid_step=20, scale=1.0):
    """
    计算考虑时间差的光流矢量场
    
    参数:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        time_delta_sec: 两帧时间差(秒)
        output_path: 输出路径(可选)
        method: 光流算法 ['farneback'|'lucas_kanade']
        grid_step: 网格采样步长(像素)
        scale: 箭头长度缩放系数
    """
    # 读取图片并检查时间差有效性
    if time_delta_sec <= 0:
        raise ValueError("时间差必须为正数")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图片")

    # 计算原始光流
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    elif method == 'lucas_kanade':
        p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)
        h, w = gray1.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)
        for (x0, y0), (x1, y1) in zip(p0[st==1].reshape(-1,2), p1[st==1].reshape(-1,2)):
            flow[int(y0), int(x0)] = [x1-x0, y1-y0]
    else:
        raise ValueError("不支持的method参数")

    # 转换为速度场 (像素/秒)
    velocity_field = flow / time_delta_sec

    # 可视化
    h, w = flow.shape[:2]
    y, x = np.mgrid[grid_step//2:h:grid_step, grid_step//2:w:grid_step]
    velocity_samples = velocity_field[y.astype(int), x.astype(int)]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    
    # 绘制带物理单位的矢量场
    quiver = ax.quiver(
        x, y, 
        velocity_samples[..., 0], -velocity_samples[..., 1],  # 注意y轴反转
        angles='xy',
        scale_units='xy',
        scale=scale,
        color='red',
        width=0.003,
        headwidth=4,
        headlength=5
    )

    # 添加带物理单位的色标
    magnitudes = np.linalg.norm(velocity_samples, axis=2)
    cbar = plt.colorbar(quiver, ax=ax, format='%.1f')
    cbar.set_label('Velocity (pixels/second)', rotation=270, labelpad=20)

    # 添加时间差标注
    ax.set_title(f"Optical Flow Velocity Field\nTime Delta: {time_delta_sec:.3f} seconds", pad=20)
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"结果已保存到: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='带时间差的光流矢量场计算')
    parser.add_argument('img1', help='第一帧图片路径')
    parser.add_argument('img2', help='第二帧图片路径')
    parser.add_argument('--time_delta', type=float, required=True, help='两帧时间差(秒)')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--method', choices=['farneback', 'lucas_kanade'], default='farneback')
    parser.add_argument('--grid', type=int, default=20, help='网格采样步长(默认20)')
    parser.add_argument('--scale', type=float, default=1.0, help='箭头长度缩放系数(默认1.0)')
    
    args = parser.parse_args()
    
    calculate_optical_flow_with_timedelta(
        args.img1, args.img2,
        time_delta_sec=args.time_delta,
        output_path=args.output,
        method=args.method,
        grid_step=args.grid,
        scale=args.scale
    )
    # python compute2frameflow.py.py /data/EP01_2_frame_0.000s.jpg /data/EP01_2_frame_0.200s.jpg --output flow_result.jpg