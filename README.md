# HO-VLM

This directory contains scripts for predicting contact info of each frame in HOI by VLM.

Command line exp:

```bash
python seq_vlmho_gen.py \
    -i /home/ubuntu/gnaq_release/rsrd/rsrd_nerfgun/build \
    -o /home/ubuntu/gnaq-proj/api/output/rsrd_nerfgun/seqk3k1\
    -k 3 --interval 1

python vla_apis.py
```

## Directory Structure

```
datasets/robotwin2/
├── vla_apis.py                     # **Main code** for contact info predicting by Qwen
├── vlmho_gen.py                    # samples, annotates, and merges RGB images and depth maps from a specified sequence folder, generating horizontally and vertically stitched sample images(**random** sample)
├── seq_vlmho_gen.py                # same function as vlmho_gen but sample in **sequence**
├── vr_apis.py                      # **Video reasoning** version for contact info predicting by Qwen
├── manual_label_ho.py              # **label** ho-contact gt for each video
├── ...
├── output/                         # Pre-processed seq_vlmho_gens
│   ├── hoi_img_seq_1
│   ├── hoi_img_seq_2                       
│   └── ...
└── README.md                      # This file
```
