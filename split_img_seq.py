from PIL import Image
import os

def split_images_by_width(img_dir, out_dir_left, out_dir_right, split_pixel):
    """
    将img_dir下的所有图片按宽度split_pixel处切割成左右两部分，分别保存到out_dir_left和out_dir_right
    支持不同宽度的图片
    """
    os.makedirs(out_dir_left, exist_ok=True)
    os.makedirs(out_dir_right, exist_ok=True)
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path)
        w, h = img.size
        # 左右切割
        left_img = img.crop((0, 0, split_pixel, h))
        right_img = img.crop((split_pixel, 0, w, h))
        left_img.save(os.path.join(out_dir_left, fname))
        right_img.save(os.path.join(out_dir_right, fname))
        print(f"已切割并保存: {fname}")

# 用法示例
split_images_by_width("/home/ubuntu/gnaq_release/rsrd/ytb_swpress_short/build/image/", 
                      "/home/ubuntu/gnaq_release/rsrd/ytb_swpress_short/build/image_left/", 
                      "/home/ubuntu/gnaq_release/rsrd/ytb_swpress_short/build/image_right/", 512)