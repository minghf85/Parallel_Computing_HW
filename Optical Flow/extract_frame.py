import cv2
import os
import argparse

def parse_time(time_str):
    """将各种时间格式转换为秒数（支持小数）"""
    try:
        if ':' in time_str:
            parts = list(map(float, time_str.split(':')))
            if len(parts) == 3:  # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:  # MM:SS
                return parts[0] * 60 + parts[1]
            else:
                raise ValueError("时间格式错误")
        else:
            return float(time_str)  # 直接处理秒数（含小数）
    except ValueError:
        raise ValueError(f"无效时间格式: {time_str} (示例: 0.1 / 1:30 / 0:0:1.5)")

def extract_frame(video_path, target_time):
    """提取视频指定时间帧（精确到毫秒）"""
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 {video_path}")
        return

    try:
        target_sec = parse_time(target_time)
    except ValueError as e:
        print(e)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误: 无法打开视频")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if target_sec < 0 or target_sec > duration:
        print(f"错误: 时间 {target_sec:.3f}秒 超出视频范围 (0-{duration:.2f}秒)")
        cap.release()
        return

    # 精确跳转到目标时间（支持小数秒）
    cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
    ret, frame = cap.read()

    if not ret:
        print("错误: 读取帧失败")
        cap.release()
        return

    # 生成输出路径
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(
        video_dir, 
        f"{video_name}_frame_{target_sec:.3f}s.jpg"
    )

    cv2.imwrite(output_path, frame)
    print(f"已保存: {output_path}")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取视频帧（支持小数秒）")
    parser.add_argument("video_path", help="视频路径（如：video.mp4）")
    parser.add_argument("target_time", help="目标时间（如：0.1 / 1:30.5 / 0:0:0.123）")
    args = parser.parse_args()
    extract_frame(args.video_path, args.target_time)
    # python extract_frame.py data/EP01_2.avi 0