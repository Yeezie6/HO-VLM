import cv2
import os
from natsort import natsorted

def images_to_video(image_folder, output_path, fps=30):
    # 获取所有图片文件并排序
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = natsorted(images)
    if not images:
        print("未找到图片文件")
        return

    # 读取第一张图片获取尺寸
    first_img_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_img_path)
    h, w, _ = frame.shape

    # 定义视频编码器和输出对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for idx, img_name in enumerate(images):
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"跳过无法读取的图片: {img_name}")
            continue
        # 在左上角添加帧编号文字（如00005.png）
        text = img_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)  # 白色
        # 黑色描边
        cv2.putText(frame, text, (10, 40), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # 白色文字
        cv2.putText(frame, text, (10, 40), font, font_scale, color, thickness, cv2.LINE_AA)
        out.write(frame)

    out.release()
    print(f"视频已保存到: {output_path}")