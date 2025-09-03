# 标注label
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import json
from datetime import datetime
import argparse
from loguru import logger as loguru
import shutil
# from arrgh import arrgh

# WS_ROOT = "/home/gnaq/dev/DMT-align"
DEVICE = torch.device("cuda")
import sys
# sys.path.append(
def parse_ranges(input_str, img_files):
    idx_set = set()
    if input_str.lower() == 'done':
        return idx_set
    ranges = input_str.split(',')
    img_nums = [int(os.path.splitext(f)[0]) for f in img_files]
    num2idx = {num: idx for idx, num in enumerate(img_nums)}
    for r in ranges:
        if '-' not in r:
            continue
        start_num, end_num = map(int, r.split('-'))
        for num in range(start_num, end_num + 1):
            if num in num2idx:
                idx_set.add(num2idx[num])
    return idx_set



def main():
    parser = argparse.ArgumentParser(description='Generate sequence for DMT-align')
    parser.add_argument('--seq-path', type=str, required=True, 
                        help='Path to the sequence directory')
    parser.add_argument('--out', '--o', type=str, default='/processed/ho_contact.json',
                        help='Output directory for the generated sequence')
    
    args = parser.parse_args()
    out_file = f"{args.seq_path}{args.out}"
    img_dir = os.path.join(args.seq_path, 'build/image')
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    frames_cnt = len(img_files)
    
    appeared = []
    left_appear = input("Did left hand appear? (y/n): ").strip().lower()
    if left_appear == 'y':
        appeared.append('left')
    right_appear = input("Did right hand appear? (y/n): ").strip().lower()
    if right_appear == 'y':
        appeared.append('right')
    
    r_contacts = [True for _ in range(frames_cnt)]
    l_contacts = [True for _ in range(frames_cnt)]
    while True:
        s = input(f"right-hand NON-contact ranges (e.g., 156-160, 170-175) or 'done' to finish: ")
        if s.lower() == 'done':
            break
        idx_set = parse_ranges(s, img_files)
        for i in idx_set:
            r_contacts[i] = False
    
    while True:
        s = input(f"left-hand NON-contact ranges (e.g., 156-160, 170-175) or 'done' to finish: ")
        if s.lower() == 'done':
            break
        idx_set = parse_ranges(s, img_files)
        for i in idx_set:
            l_contacts[i] = False
    loguru.info(f"Final contact ranges: {r_contacts}, {l_contacts}")
    contact_list = []
    for i, (cr, cl) in tqdm(enumerate(zip(r_contacts, l_contacts)), total=frames_cnt):
        frame_num = int(os.path.splitext(img_files[i])[0])  # 提取文件名数字部分
        contact_list.append({
            'frame': frame_num,
            'r_contact': cr,
            'l_contact': cl,
        })
    contact_dict = {
        'frames_cnt': frames_cnt,
        'appeared': appeared,
        'contacts': contact_list,
    }
    with open(out_file, 'w') as f:
        json.dump(contact_dict, f, indent=2)
    loguru.info(f"Contact data saved to {out_file}")

if __name__ == "__main__":
    main()