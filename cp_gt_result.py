import os
import shutil

# cp gt
# src_root = "/home/ubuntu/gnaq_release/rsrd/"

# cp result
src_root = "./output"
dst_root = "./result"

for folder in os.listdir(src_root):
    src_folder = os.path.join(src_root, folder)
    if not os.path.isdir(src_folder):
        continue
    src_json = os.path.join(src_folder, "ho_contact.json")
    if not os.path.exists(src_json):
        continue
    dst_folder = os.path.join(dst_root, folder)
    os.makedirs(dst_folder, exist_ok=True)
    dst_json = os.path.join(dst_folder, "ho_contact.json")
    shutil.copy2(src_json, dst_json)
    print(f"已复制: {src_json} -> {dst_json}")