# dense_optical_flow.py
import cv2
import numpy as np
import argparse


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    # 读取第一帧
    ret, old_frame = cap.read()

    # 创建HSV并使Value为常量
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # 精确方法的预处理
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    index = 0
    while True:
        index += 1
        # 读取下一帧
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # 精确方法的预处理
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # 计算光流
        flow = method(old_frame, new_frame, None, *params)

        # 编码:将算法的输出转换为极坐标
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # 使用色相和饱和度来编码光流
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # 转换HSV图像为BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["farneback", "lucaskanade_dense", "rlof"],
        required=True,
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path", default="duck.mp4", help="Path to the video",
    )

    args = parser.parse_args()
    video_path = args.video_path
    if args.algorithm == "lucaskanade_dense":
        method = cv2.optflow.calcOpticalFlowSparseToDense
        dense_optical_flow(method, video_path, to_gray=True)

    elif args.algorithm == "farneback":
        # OpenCV Farneback算法需要一个单通道的输入图像，因此我们将BRG图像转换为灰度。
        method = cv2.calcOpticalFlowFarneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback的算法参数
        dense_optical_flow(method, video_path, params, to_gray=True)

    elif args.algorithm == "rlof":
        # 与Farneback算法相比，RLOF算法需要3通道图像，所以这里没有预处理。
        method = cv2.optflow.calcOpticalFlowDenseRLOF
        dense_optical_flow(method, video_path)


if __name__ == "__main__":
    main()
# python realtimeof.py --algorithm farneback --video_path Optical\ Flow/data/SVID_20230909_085341_1.mp4