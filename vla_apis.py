from openai import OpenAI
import time
import base64
import requests
import dashscope
import os
from natsort import natsorted
from dashscope import MultiModalConversation
import json
import re
from collections import defaultdict

def encode_image(image_path):
    """
    Encode an image to base64 string.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def load_gt_contacts(json_path):
    with open(json_path, 'r') as f:
        gt = json.load(f)
    gt_dict = {}
    for item in gt["contacts"]:
        gt_dict[str(item["frame"]).zfill(5)] = {
            "Left": item["l_contact"],
            "Right": item["r_contact"]
        }
    return gt_dict

def calc_vlm_accuracy_multi(vlm_result, gt_result):
    total = 0
    correct = 0
    wrong_frames = []
    for frame, preds in vlm_result.items():
        gt = gt_result.get(frame)
        if gt is None:
            continue
        for pred in preds:
            if pred["Left"] == gt["Left"] and pred["Right"] == gt["Right"]:
                correct += 1
            else:
                wrong_frames.append({
                    "frame": frame,
                    "vlm": pred,
                    "gt": gt
                })
            total += 1
    acc = correct / total if total > 0 else 0
    return acc, wrong_frames


def parse_result_str(result_str):
    """
    将模型输出的字符串解析为Python字典
    """
    result = {}
    # 去掉大括号和多余空白
    result_str = result_str.strip().strip('{}').strip()
    # 按行分割
    for line in result_str.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 匹配帧号和左右手
        m = re.match(r'(\d+): Left:(\w+), Right:(\w+)', line)
        if m:
            frame, left, right = m.groups()
            result[frame] = {'Left': left.lower() == 'true', 'Right': right.lower() == 'true'}
    return result



# 假设 qwen_results = {img_name: result_str, ...}
def parse_all_results(qwen_results):
    all_preds = defaultdict(list)
    for img_name, result_str in qwen_results.items():
        parsed = parse_result_str(result_str)
        for frame, pred in parsed.items():
            all_preds[frame].append(pred)
    return all_preds

# 假设你的qwen_results如下
# qwen_results = {...}

class Qwen:
    """
    Qwen/Qwen-VL-Max
    """
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = "sk-266647a45f5c43359ad636d536ea657d"  

    def __init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
    )

    def request_with_image(self, prompt, image_path):
        image_base64 = encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        completion = self.client.chat.completions.create(
            model="qwen-vl-max-latest",  # 可根据实际模型更换
            messages=messages
        )
        return completion.choices[0].message.content
    """
    qwen-vl-max
    qvq-max
    qwen2.5-vl-72b-instruct 不太行
    qwen-vl-max-latest
    """
    


class GPT:
    api_base = "http://rerverseapi.workergpt.cn/v1"
    api_key= "sk-VZX3uKA47hTAePoSF8FcEb91724c4c47A23e6547606583Fb"
    deployment_name = 'gpt-4o'
    completion_tokens = 0
    prompt_tokens = 0
    def __init__(self):
        self.client = OpenAI(
            api_key=self.api_key,  
            base_url=self.api_base
        )
    def request_with_image(self, prompt, image_path):
        image_base64 = encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content":[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail":"low"
                     }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        result = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
        )
        self.completion_tokens += result.usage.completion_tokens
        self.prompt_tokens += result.usage.prompt_tokens
        return result.choices[0].message.content
        


class QwenSingle:
    """
    使用DashScope SDK，支持本地图片base64方式，支持高分辨率参数
    """
    def __init__(self):
        self.api_key = "sk-266647a45f5c43359ad636d536ea657d"
        self.model = "qwen-vl-max-latest"
    

    def request_with_image(self, prompt, image_path):
        base64_image = encode_image(image_path)
        messages = [
            {
                "role": "system",
                "content": [{"text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{base64_image}"},
                    {"text": prompt}
                ]
            }
        ]
        # 调用DashScope多模态接口，设置高分辨率参数
        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            vl_high_resolution_images=True
        )
        # 返回文本内容
        return response.output.choices[0].message.content[0]["text"]
    
    
class QwenMulti:
    """
    使用DashScope SDK，支持多图base64输入，支持高分辨率参数
    """
    def __init__(self):
        self.api_key = "sk-266647a45f5c43359ad636d536ea657d"
        self.model = "qwen-vl-max"

    def request_with_images(self, prompt, image_paths, image_format="jpg"):
        image_contents = []
        for path in image_paths:
            base64_img = encode_image(path)
            image_contents.append({"image": f"data:image/{image_format};base64,{base64_img}"})
            print(path)
        image_contents.append({"text": prompt})

        messages = [
            {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
            {"role": "user", "content": image_contents}
        ]
        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            parameters={"vl_high_resolution_images": True}
        )

        return response.output.choices[0].message.content[0]["text"]
        

prompt = """
    this is a horizontally merged sequence of K selected frame from a video which is from a first-person perspective.
    in the video, a person is interacting with an articulated object.
    there are black bars between each two frames in all of these K frames.
    the first row is the RGB frames, and the second row is the depth frames.
    the selecting method is using a fixed interval to select K frames from the video, 
    with a random starting point.

    Based on the perspective information, please compare these frames and answer question twice, once for left hand of human and once for right hand of human.

    starts from the first 2 of K, determine for the left / right hand in this interval:
    (true) the hand is contacting the object in this frame.
    (false) the hand is not contacting the object in this frame.

    then, based on your answer for each pair, please output 2 * K answers in the following format.
    here is an example of K=4:
        for left hand:
            at frame 00010, the left hand is not contacting the object;
            at frame 00020, the left hand starts contacting the object and keeps contacting it until frame 00030;
            at frame 00040, the left hand is no longer in contact.
        
        for right hand:
            the right hand is in contact with the object from frame 00010 to frame 00030,
            and stops contacting it at frame 00040.

    then the correct format is:

        {
            00010: Left:true, Right:true
            00020: Left:false, Right:true
            00030: Left:false, Right:true
            00040: Left:true, Right:false
        }

    In situations where you are not highly confident about whether contact occurs, you should prefer choice false rather than true.
    please do not output any information other than that format.
"""

# QwenMulti add on
"""
finally, if I uploaded multiple (merged) images, please output the answer for each image in a new line.
    For the contact information of all frames contained in the input images, please check for consistency before outputting the results, to avoid self-contradictory situations where a frame is marked as both false and true. 
    If any inconsistency is found during the check, you need to re-analyze and ensure consistency; when re-analyzing, you should prefer to judge as false.
    please use the global information to reason the images. For example, you should leverage all images in the input to reason about the contact status in the first image.
"""
 
def main():
    
    qwen_results = {}  # 用于存储每张图片的推理结果
     
    
    for i in range(1, 31):
        """
        测试多模态模型读取图片并生成文本
        """
        print(f"i = {i}")
        image_path = f"/home/ubuntu/gnaq-proj/api/output/rsrd_redbox/k3k1r100/sampleRGBD_{i}.jpg"  # 替换为你的图片路径
        

        # print("=== GPT-4o 多模态 ===")
        # gpt = GPT()
        # gpt_response = gpt.request_with_image(prompt, image_path)
        # print("GPT-4o:", gpt_response)

        # print("=== QwenVL 多模态 ===")
        # qwen = Qwen()
        # qwen_response = qwen.request_with_image(prompt, image_path)
        # print("QwenVL:", qwen_response)
        
        print("=== Qwen 多模态 ===")
        qwen = QwenSingle()
        qwen_response = qwen.request_with_image(prompt, image_path)
        print("QwenSingle:", qwen_response)
        
        # 存储到字典，key为图片名，value为模型输出
        qwen_results[os.path.basename(image_path)] = qwen_response
        
    # 合并所有图片的帧为一个vlm_result
    vlm_result = parse_all_results(qwen_results)
    print(vlm_result)
    
    gt_result = load_gt_contacts("/home/ubuntu/gnaq_release/rsrd/rsrd_redbox/processed/ho_contact.json")
    acc, wrong_frames = calc_vlm_accuracy_multi(vlm_result, gt_result)
    
    print("Accuracy: {:.4f}".format(acc))
    print("Wrong frames:", wrong_frames)
        
    # img_dir = f"/home/ubuntu/gnaq-proj/api/output/rsrd_nerfgun/k3k1r30"
    # image_paths = [
    #     os.path.join(img_dir, fname)
    #     for fname in natsorted(os.listdir(img_dir))
    #     if "RGBD" in fname and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    # ][:10]
    
    # print("=== QwenMulti 多模态 ===")
    # qwen = QwenMulti()
    # qwen_response = qwen.request_with_images(prompt, image_paths)
    # print("QwenMulti:", qwen_response)

if __name__ == '__main__':
    main()
    
    