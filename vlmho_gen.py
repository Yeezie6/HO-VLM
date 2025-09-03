# 将指定序列文件夹中的RGB图片和深度图进行采样、标注、合并，生成横向和纵向拼接的样本图片，便于多模态模型推理和可视化。
import os
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def colorize_depth(depth_map, ctab='turbo'):
    """
    Colorizes a depth map using a colormap.
    
    Args:
        depth_map (numpy.ndarray): The depth map to colorize.
        ctab (str): The colormap to use. Default is 'turbo'.
        
    Returns:
        numpy.ndarray: The colorized depth map.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Normalize the depth map to [0, 1]
    norm_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    
    # Apply the colormap
    cmap = cm.get_cmap(ctab)
    colorized_depth = cmap(norm_depth)[:, :, :3]  # Get RGB channels
    
    return (colorized_depth * 255).astype(np.uint8)

def read_images_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    img_extensions = {'.jpg', '.jpeg', '.png',}
    img_files = [f for f in files if os.path.splitext(f.lower())[1] in img_extensions]
    sorted_img_files = natsorted(img_files)
    return [os.path.join(folder_path, f) for f in sorted_img_files]

def read_depths_from_file(file_path):
    depths = np.load(file_path, allow_pickle=True)
    return depths

def annotate_image(img : Image.Image, img_path: str):
    # img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    filename = os.path.basename(img_path)
    
    font_size = int(min(img.width, img.height) * 0.07)  # 字体大小为图片尺寸的7%
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # 计算标注位置（左上角5%宽度，12%高度的区域）
    text_position = (int(img.width * 0.05), int(img.height * 0.05))
    
    # 添加文字，白色带黑色边框以增加可读性
    # 添加黑色阴影
    shadow_offset = max(1, int(font_size * 0.05))
    draw.text((text_position[0]+shadow_offset, text_position[1]+shadow_offset), filename, font=font, fill="black")
    # 添加主文字
    draw.text(text_position, filename, font=font, fill="white")
    
    return img

def merge_images_horizontally(image_tuples):
    ret = []
    for tup in image_tuples:
        N = len(tup)
        # 添加标注并获取所有图片
        annotated_images = [annotate_image(img, path) for img, path in tup]
    
        # 找出最大的高度，以便统一
        max_height = max(img.height for img in annotated_images)
        
        # 调整所有图片的高度为最大高度，保持宽高比
        resized_images = []
        for img in annotated_images:
            ratio = max_height / img.height
            new_width = int(img.width * ratio)
            resized_images.append(img.resize((new_width, max_height), Image.LANCZOS))
    
        # 黑色分隔条的宽度
        separator_width = 12
        
        # 计算合并后图片的总宽度（加上分隔条的宽度）
        total_width = sum(img.width for img in resized_images)
        if N > 1:  # 只有当有多张图片时才添加分隔条
            total_width += (N - 1) * separator_width
    
        # 创建新图片
        merged_image = Image.new('RGB', (total_width, max_height))
        
        # 将调整后的图片粘贴到新图片上，并添加黑色分隔条
        x_offset = 0
        for i, img in enumerate(resized_images):
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
            
            # 在图片后面添加黑色分隔条（最后一张图片后不添加）
            if i < len(resized_images) - 1 and N > 1:
                # 黑色分隔条区域保持为黑色（默认背景色）
                x_offset += separator_width
        ret.append(merged_image)
    return ret

def merge_images_vertically(imgs):
    if not imgs:
        return None

    N = len(imgs)
    # 黑色分隔条的高度
    separator_height = 12
    
    # 计算合并后图片的总高度（加上分隔条的高度）
    total_height = sum(img.height for img in imgs)
    if N > 1:
        total_height += (N - 1) * separator_height
        
    max_width = max(img.width for img in imgs)
    
    # 创建新图片
    merged_image = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for i, img in enumerate(imgs):
        # 水平居中粘贴图片
        x_pos = (max_width - img.width) // 2
        merged_image.paste(img, (x_pos, y_offset))
        y_offset += img.height
        
        # 在图片下面添加黑色分隔条（最后一张图片后不添加）
        if i < N - 1:
            y_offset += separator_height
    
    return merged_image

def main():
    import argparse

    parser = argparse.ArgumentParser(description="合并并标注图片")
    parser.add_argument("-i", "--input", help="序列文件夹路径")
    parser.add_argument("-o", "--output", default="merged_output.jpg", help="输出文件路径，默认为 merged_output.jpg")
    parser.add_argument("-k", "--count", type=int, default=4, help="合并图片数量，默认为4")
    parser.add_argument("--interval", type=int, default=1, help="每张图片之间的间隔，单位为秒，默认为1")
    parser.add_argument("-r", "--round", type=int, default=1, help="合并轮次, 默认为1")
    parser.add_argument("--rand-k", action="store_true", help="随机选择k张图片进行合并")
    
    args = parser.parse_args()
    
    img_paths = read_images_from_folder(f"{args.input}/image")
    # generate `round` times different sample
    os.makedirs(args.output, exist_ok=True)
    # 随机采样初始帧
    samples = np.random.choice(
        len(img_paths) - args.count * args.interval + 1, 
        args.round, 
        replace=False
    )
    
    
    depths = read_depths_from_file(os.path.join(args.input, "metric_depth.pkl"))
    rgbs = [Image.open(path) for path in img_paths]
    depths_colorized = [colorize_depth(depths[i]) for i in range(len(depths))]
    
    # generate sample tuples
    K = args.count
    interval = int(args.interval)
    img_tups = []
    dep_tups = []
    for i in range(args.round):
        cur_id = samples[i]
        if args.rand_k:
            selected_indices = np.random.choice(len(img_paths), K, replace=False)
        else:
            selected_indices = [_ for _ in range(cur_id, len(img_paths), interval)]
            selected_indices = selected_indices[:K]
        img_tups.append([(rgbs[i], img_paths[i]) for i in selected_indices])
        dep_tups.append([(Image.fromarray(depths_colorized[i]), img_paths[i]) for i in selected_indices])
    
    # generate merged images from tuples
    merged_rgb = merge_images_horizontally(img_tups)
    merged_depth = merge_images_horizontally(dep_tups)
    
    for i, selected in enumerate(merged_rgb):
        output_path = f"{args.output}/sampleRGB_{i+1}.jpg"
        selected.save(output_path)
        composite = merge_images_vertically([selected, merged_depth[i]])
        composite_output_path = f"{args.output}/sampleRGBD_{i+1}.jpg"
        composite.save(composite_output_path)

if __name__ == "__main__":
    main()


"""
python vlmho_gen.py \
    -i /home/ubuntu/gnaq_release/rsrd/rsrd_nerfgun/build \
    -o /home/ubuntu/gnaq-proj/api/output/rsrd_nerfgun/k3k1r50\
    -k 3 -r 50 --interval 1
"""

"""
this is a horizontally merged sequence of K selected frame from a video, 
in the video, a person is interacting with an articulated object.
there are black bars between each two frames in all of these K frames.
the first row is the RGB frames, and the second row is the depth frames.
the selecting method is using a fixed interval to select K frames from the video, 
with a random starting point.

Please compare these frames and answer question twice, once for left hand and once for right hand.
if there's no left hand or right hand in the video, please answer 'none' for that hand.

starts from the first 2 of K, is the left / right hand performing:
(A) fingertips start touching this articulated object within this interval, 
    and not doing this in the first frame of this interval.
(B) fingertips stop touching this articulated object within this interval,
    and not doing this in the first frame of this interval.
(C) fingertips keep touching this articulated object within this interval.
(D) fingertips not touching this articulated object within this interval.

then, based on your answer for each pair, please output 2 * K-1 answers in the following format:

example of K=4:
{
    00010-00020: Left:C, Right:A,
    00020-00030: Left:B, Right:D,
    00030-00040: Left:A, Right:C,
}

finally, if I uploaded multiple (merged) images, please output the answer for each image in a new line.
"""