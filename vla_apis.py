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
            "r_contact": item["r_contact"],
            "l_contact": item["l_contact"]
        }
    return gt_dict

def calc_vlm_accuracy_multi(vlm_result, gt_result):
    """
    计算VLM输出与GT的准确率，只要左右手都正确才算该帧正确
    支持vlm_result["contacts"]为list，每个元素包含frame, r_contact, l_contact
    gt_result为dict，key为帧号字符串（如'00031'），value为左右手接触dict
    wrong_frames 只输出 frame, l_contact, r_contact
    """
    total = 0
    correct = 0
    wrong_frames = []
    for contact in vlm_result["contacts"]:
        frame_key = str(contact["frame"]).zfill(5)
        gt = gt_result.get(frame_key)
        if gt is None:
            continue
        if contact.get("r_contact") == gt["r_contact"] and contact.get("l_contact") == gt["l_contact"]:
            correct += 1
        else:
            wrong_frames.append({
                "frame": frame_key,
                "vlm_l_contact": contact.get("l_contact"),
                "vlm_r_contact": contact.get("r_contact"),
                "gt_l_contact": gt["l_contact"],
                "gt_r_contact": gt["r_contact"]
            })
        total += 1
    acc = correct / total if total > 0 else 0
    return acc, wrong_frames


import json
from collections import defaultdict

def parse_all_results(qwen_results):
    """
    合并所有图片的VLM输出为标准结构，支持手指信息
    对于重复帧，contact采用少数服从多数，若true/false一样多则取false，手指信息合并去重
    """
    frame_contacts = defaultdict(list)
    appeared_set = set()
    # 支持 fingers 字段的正则
    contact_pattern = re.compile(
        r'"frame"\s*:\s*"?(\d+)"?,\s*"r_contact"\s*:\s*(true|false),\s*"l_contact"\s*:\s*(true|false)(?:,\s*"r_fingers"\s*:\s*\[([^\]]*)\])?(?:,\s*"l_fingers"\s*:\s*\[([^\]]*)\])?'
    )
    for img_name, result_str in qwen_results.items():
        for m in contact_pattern.finditer(result_str):
            frame = int(m.group(1))
            r_contact = m.group(2) == "true"
            l_contact = m.group(3) == "true"
            r_fingers = re.findall(r'"(thumb|index|middle)"', m.group(4) or "")
            l_fingers = re.findall(r'"(thumb|index|middle)"', m.group(5) or "")
            frame_contacts[frame].append({
                "r_contact": r_contact,
                "l_contact": l_contact,
                "r_fingers": r_fingers,
                "l_fingers": l_fingers
            })
            if r_contact:
                appeared_set.add("right")
            if l_contact:
                appeared_set.add("left")

    # 合并同帧的结果，手指contact采用出现次数最多的组合，平票取手指数少的，仍平票则取最后一个
    contacts = []
    for frame in sorted(frame_contacts.keys()):
        items = frame_contacts[frame]
        r_true = sum(item["r_contact"] for item in items)
        r_false = len(items) - r_true
        l_true = sum(item["l_contact"] for item in items)
        l_false = len(items) - l_true
        # 多数为True，否则False，平票取False
        r_contact = True if r_true > r_false else False
        l_contact = True if l_true > l_false else False

        # 统计所有r_fingers组合出现次数
        r_finger_counts = defaultdict(int)
        r_finger_last_idx = {}
        for idx, item in enumerate(items):
            key = tuple(sorted(item["r_fingers"]))
            r_finger_counts[key] += 1
            r_finger_last_idx[key] = idx
        # 找出现次数最多的组合
        max_count = max(r_finger_counts.values()) if r_finger_counts else 0
        candidates = [k for k, v in r_finger_counts.items() if v == max_count]
        # 若有多个，选手指数少的
        min_len = min(len(k) for k in candidates) if candidates else 0
        candidates2 = [k for k in candidates if len(k) == min_len]
        # 若还有多个，选最后出现的
        if candidates2:
            chosen_r_fingers = max(candidates2, key=lambda k: r_finger_last_idx[k])
        else:
            chosen_r_fingers = ()

        # l_fingers同理
        l_finger_counts = defaultdict(int)
        l_finger_last_idx = {}
        for idx, item in enumerate(items):
            key = tuple(sorted(item["l_fingers"]))
            l_finger_counts[key] += 1
            l_finger_last_idx[key] = idx
        max_count = max(l_finger_counts.values()) if l_finger_counts else 0
        candidates = [k for k, v in l_finger_counts.items() if v == max_count]
        min_len = min(len(k) for k in candidates) if candidates else 0
        candidates2 = [k for k in candidates if len(k) == min_len]
        if candidates2:
            chosen_l_fingers = max(candidates2, key=lambda k: l_finger_last_idx[k])
        else:
            chosen_l_fingers = ()

        contacts.append({
            "frame": frame,
            "r_contact": r_contact,
            "l_contact": l_contact,
            "r_fingers": list(chosen_r_fingers),
            "l_fingers": list(chosen_l_fingers)
        })

    return {
        "frames_cnt": len(contacts),
        "appeared": sorted(list(appeared_set)),
        "contacts": contacts
    }

def get_contact_segments(contacts, hand):
    segments = []
    in_contact = False
    start = None
    fingers_set = set()
    for c in contacts:
        contact = c[f"{hand}_contact"]
        fingers = set(c.get(f"{hand}_fingers", []))
        if contact:
            if not in_contact:
                in_contact = True
                start = c["frame"]
                fingers_set = set(fingers)
            else:
                fingers_set |= fingers
        else:
            if in_contact:
                segments.append({"start": start, "end": c["frame"]-1, "fingers": sorted(list(fingers_set))})
                in_contact = False
                start = None
                fingers_set = set()
    if in_contact:
        segments.append({"start": start, "end": contacts[-1]["frame"], "fingers": sorted(list(fingers_set))})
    return segments


def save_vlm_result_to_json(vlm_result, out_path):
    """
    直接保存标准结构（已解析好的字典）
    """
    with open(out_path, "w") as f:
        json.dump(vlm_result, f, indent=2, ensure_ascii=False)
        

def count_files(directory):
        try:
            return sum(
                1 for entry in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, entry))  # 确保是文件
                and "RGBD" in entry
                and entry.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))  # 匹配图片格式
            )
        except FileNotFoundError:
            print(f"目录 '{directory}' 不存在")
            return 0


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
        

class QwenPerspective:
    """
    判断人称视角，根据所有图片
    """
    def __init__(self):
        self.api_key = "sk-266647a45f5c43359ad636d536ea657d"
        self.model = "qwen-vl-max-latest"

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

        # print("response:", response)
        # if not response or not getattr(response, "output", None):
        #     print("Warning: response or response.output is None")
        #     return ""
        
        return response.output.choices[0].message.content[0]["text"]


class QwenSingle:
    """
    使用DashScope SDK，支持本地图片base64方式，支持高分辨率参数
    """
    def __init__(self):
        self.api_key = "sk-266647a45f5c43359ad636d536ea657d"
        self.model = "qwen-vl-max-latest"
    
    def request_with_image(self, prompt, image_path, max_retries=5):
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
        retry = 0
        wait_time = 10  # 初始等待10秒
        while retry < max_retries:
            try:
                response = dashscope.MultiModalConversation.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    vl_high_resolution_images=True
                )
                # 健壮性检查
                if not response or not getattr(response, "output", None):
                    print("Warning: response or response.output is None")
                    raise Exception("Empty response")
                if not response.output.choices or not response.output.choices[0].message.content:
                    print("Warning: response.output.choices[0].message.content is empty")
                    raise Exception("Empty choices")
                return response.output.choices[0].message.content[0]["text"]
            except Exception as e:
                print(f"API调用异常: {e}，将在{wait_time}秒后重试（第{retry+1}次）")
                time.sleep(wait_time)
                wait_time *= 2  # 指数退避
                retry += 1
        print("API多次重试失败，跳过该图片。")
        return ""
    
    
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
            # print(path)
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
        
        # print("response:", response)
        # if not response or not getattr(response, "output", None):
        #     print("Warning: response or response.output is None")
        #     return ""

        return response.output.choices[0].message.content[0]["text"]
      
      
prompt_perspective = """
    You are given a set of images sampled from a video about a human manipulating an articulated object.
    Please determine whether this video is from a first-person perspective or a third-person perspective.
    A first-person perspective means the video is filmed from the operator's point of view, usually showing the operator's arms or hands extending from the bottom or sides of the frame, and the viewpoint is aligned with the operator's head direction.
    A third-person perspective means the video is filmed from an observer's point of view, usually showing the whole or most of the operator's body, and the viewpoint is not aligned with the operator's head direction.
    Here are some judgment principles:
        1. If only one hand appears, it must be first-person perspective.
        2. If a human face appears, it must be third-person perspective.
        3. In first-person perspective, the hand(s) usually occupy a large area of the image.
    If it is first-person perspective, output only the number 1.
    If it is third-person perspective, output only the number 3.
    Do not output any other text or explanation.
"""


prompt_change = """
Please note that in the given image, the contact status between the hand and the object may change from one frame to another. Do not simply output the same contact status (all true or all false) for all frames unless the visual evidence is truly identical. Carefully observe each frame and make an independent judgment for each one, considering possible changes in hand-object contact across adjacent frames.
"""


prompt = """
    in the video, a person is interacting with an articulated object.
    there are black bars between each two frames in all of these K frames.
    the first row is the RGB frames, and the second row is the depth frames.
    the selecting method is using a fixed interval to select K frames from the video, 
    with a random starting point.

    Based on the perspective information, please firstly accurately distinguish between the left and right hands in each frame. If only one hand appears, please determine whether it is the left hand or the right hand.
    Then please determine whether the left hand and right hand appear in the video frame sequence. if a hand does not appear in any frame, mark it as not appeared. 
    Please compare these frames and answer question twice, once for left hand of human and once for right hand of human. 
    

    starts from the first 2 of K, determine for the left / right hand in this interval:
    (true) the hand is contacting the object in this frame.
    (false) the hand is not contacting the object in this frame.
    
    When you are not sure whether hand-object contact occurs, prefer false.
    
    then, based on your answer for each pair, please output 2 * K answers in the following format.
    here is an example of K=4:
        Both hands appear in the K frames.
        for left hand:
            at frame 00010, the left hand is not contacting the object;
            at frame 00020, the left hand starts contacting the object and keeps contacting it until frame 00030;
            at frame 00040, the left hand is no longer in contact.
        
        for right hand:
            the right hand is in contact with the object from frame 00010 to frame 00030,
            and stops contacting it at frame 00040.

    then the correct format is:

        {
            "frames_cnt": 4,
            "appeared": ["left", "right"],  // or ["left"], ["right"], or []
            "contacts": [
                {"frame": 10, "r_contact": true, "l_contact": false},
                {"frame": 20, "r_contact": true, "l_contact": true},
                {"frame": 30, "r_contact": true, "l_contact": true},
                {"frame": 40, "r_contact": false, "l_contact": false}
            ]
        }
    
    please do not output any information other than that format.
    
    For each frame, in addition to predicting whether the left and right hands are contacting the object, you must also independently determine for each hand which of the following fingers are in contact with the object: the fingertips of the thumb, index finger, and middle finger. 
    Please carefully observe the position, shape, and occlusion of each fingertip. If you are not sure whether a fingertip is in contact, prefer to mark it as not in contact.
    If in frame 10, the left hand has only the middle fingertip in contact with the object, and the right hand has the thumb, index fingertip, and middle fingertip in contact with the object,
    then output the result for each frame in the following JSON format:

    {
        "frame": 10,
        "r_contact": true,
        "l_contact": false,
        "r_fingers": ["thumb", "index", "middle"],  // right hand fingertips in contact
        "l_fingers": ["middle"]                   // left hand fingertips in contact
    }

    Do not ignore the middle fingertip. If the middle fingertip is in contact with the object, be sure to include 'middle' in the list.
    If a fingertip is occluded or unclear, do not include it in the list. Only include a fingertip if you are completely confident it is in contact with the object.
"""

prompt_multi_img = """
    finally, if I uploaded multiple (merged) images, please output the answer for each image in a new line.
    For the contact information of all frames contained in the input images, please check for consistency before outputting the results, to avoid self-contradictory situations where a frame is marked as both false and true. 
    If any inconsistency is found during the check, you need to re-analyze and ensure consistency; when re-analyzing, you should prefer to judge as false.
    please use the global information to reason the images. For example, you should leverage all images in the input to reason about the contact status in the first image.
"""


ds = "rs_scissor"  # type of dataset
fd = "seqk3k1"


def main():
    
    qwen_results = {}  # 用于存储每张图片的推理结果
    folder_path = f"/home/ubuntu/gnaq-proj/api/output/{ds}/{fd}"
    
    
    # 打印seqk3k1文件夹的文件数目
    seqk3k1_path = os.path.join(os.path.dirname(folder_path), "seqk3k1")
    if os.path.exists(seqk3k1_path):
        num_files = len([f for f in os.listdir(seqk3k1_path) if os.path.isfile(os.path.join(seqk3k1_path, f))])
        print(f"RGBD image 数目: {int(num_files / 2)}")
    else:
        print(f"文件夹 {seqk3k1_path} 不存在")

    image_paths = [
        os.path.join(folder_path, fname)
        for fname in natsorted(os.listdir(folder_path))
        if "RGBD" in fname and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ][:10]
    for i in range(1, 2):
        print("=== QwenPerspective 多模态 ===")
        qwenp = QwenPerspective()
        qwen_response_p = qwenp.request_with_images(prompt_perspective, image_paths)
        print("QwenPerspective:", qwen_response_p)

        print()

        # 解析人称视角
        perspective = str(qwen_response_p).strip()
        if perspective not in ["1", "3"]:
            print("Warning: QwenPerspective output not recognized, defaulting to 1st person.")
            perspective = "1"  
        print(perspective)
    
    if perspective == "3":
        hand_hint = (
            "In third-person perspective, the left hand of the person usually appears on the right side of the object, "
            "and the right hand appears on the left side, as seen from the observer’s viewpoint."
            "If there is only one hand in the image, please distinguish the hand with the information above."
        )
    else:
        hand_hint = (
            "In first-person perspective, the left hand of the person appears on the left side of the object, "
            "and the right hand appears on the right side, due to the camera facing outward from the operator’s viewpoint."
            "If there is only one hand in the image, please distinguish the hand with the information above."
        )
        
    prompt_with_perspective = f"""
        this is a horizontally merged sequence of K selected frame from a video which id from a {'third' if perspective == '3' else 'first'}-person perspective.
    {hand_hint}
    {prompt}
    """
    
    # print(prompt_with_perspective)
    
    for i in range(1, count_files(folder_path)+1):
    # test
    # for i in range(104, 107):
        """
        测试多模态模型读取图片并生成文本
        """
        # if (i - 1) % 80 == 0 and i != 1:
        #    print("已处理80张图片，休息1分钟以缓解API压力...")
        #    time.sleep(60)

        
        print(f"i = {i}")
        image_path = f"sampleRGBD_{i}.jpg"  # 替换为你的图片路径
        path = os.path.join(folder_path, image_path)
        

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
        qwen_response = qwen.request_with_image(prompt_with_perspective, path)
        print("QwenSingle:", qwen_response)
        
        # 存储到字典，key为图片名，value为模型输出
        qwen_results[os.path.basename(path)] = qwen_response
        
    # 合并所有图片的帧为一个vlm_result
    vlm_result = parse_all_results(qwen_results)
    vlm_result["contact_segments"] = {
        "left": get_contact_segments(vlm_result["contacts"], "l"),
        "right": get_contact_segments(vlm_result["contacts"], "r")
    }
    
    # print(vlm_result)
    # 如果某只手的appeared是false，则所有帧该手的fingers都置为[]
    appeared = set(vlm_result.get("appeared", []))
    for contact in vlm_result["contacts"]:
        if "left" not in appeared:
            contact["l_fingers"] = []
        if "right" not in appeared:
            contact["r_fingers"] = []
    # contact_segments 也要处理
    for seg in vlm_result["contact_segments"].get("left", []):
        if "left" not in appeared:
            seg["fingers"] = []
    for seg in vlm_result["contact_segments"].get("right", []):
        if "right" not in appeared:
            seg["fingers"] = []
    save_vlm_result_to_json(vlm_result, f"./output/{ds}/ho_contact.json")
    
    gt_result = load_gt_contacts(f"/home/ubuntu/gnaq_release/rsrd/{ds}/processed/ho_contact.json")
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
    
    
"""
改ds # type of dataset
"""