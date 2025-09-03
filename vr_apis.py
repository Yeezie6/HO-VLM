# VLM判断视频HOI contact
from openai import OpenAI
import os
import base64
from syn_img import images_to_video

prompt = """
请你分析我上传的视频，分别判断视频中哪一帧中左手和右手与物体接触或者分离。
"""
#  base 64 编码格式 
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 将xxxx/test.mp4替换为你本地视频的绝对路径
images_to_video(
        image_folder="/home/ubuntu/gnaq_release/rsrd/rsrd_bear/build/image",      # 替换为你的图片文件夹路径
        output_path="/home/ubuntu/gnaq_release/rsrd/rsrd_bear/build/video.mp4",        # 输出视频路径                                 # 帧率，可根据需要调整
    )
base64_video = encode_video("/home/ubuntu/gnaq_release/rsrd/rsrd_bear/build/video.mp4")

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key = "sk-266647a45f5c43359ad636d536ea657d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen-vl-max",  # 此处以 qvq-max 为例，可按需更换模型名称
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    # 需要注意，传入Base64编码时，video/mp4应根据本地视频的格式进行修改
                    "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ],
    stream=True,
    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
)

print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)