# Magi, The Manga Whisperer

![Static Badge](https://img.shields.io/badge/v1-grey) 
[![Static Badge](https://img.shields.io/badge/arXiv-2401.10224-blue)](http://arxiv.org/abs/2401.10224)
[![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fragavsachdeva%2Fmagi%3Fexpand%255B%255D%3Ddownloads%26expand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20Downloads)](https://huggingface.co/ragavsachdeva/magi)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/ragavsachdeva/the-manga-whisperer/)

![Static Badge](https://img.shields.io/badge/v2-grey) 
[![Static Badge](https://img.shields.io/badge/arXiv-2408.00298-blue)](https://arxiv.org/abs/2408.00298)
[![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fragavsachdeva%2Fmagiv2%3Fexpand%255B%255D%3Ddownloads%26expand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=%F0%9F%A4%97%20Downloads)](https://huggingface.co/ragavsachdeva/magiv2)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/ragavsachdeva/Magiv2-Demo)

```bash
sudo apt-get update && sudo apt-get install cbm ffmpeg git-lfs
```

- 人物名称判定不精确的版本（可以考虑使用tag进行改进）
```python
#!/usr/bin/env python
# coding: utf-8

# 安装依赖
#get_ipython().system('pip install torch datasets huggingface_hub transformers scipy einops pulp shapely timm')

# 导入必要的库
from datasets import load_dataset
import os
from PIL import Image
import numpy as np
from transformers import AutoModel
import torch

# 配置输出路径
output_dir = "first_sTitle_output_pages"  # 可配置的输出路径
os.makedirs(output_dir, exist_ok=True)  # 确保路径存在

# 加载《原神》漫画的英文数据集
ds = load_dataset("svjack/Genshin-Impact-Manga-EN-US")

# 加载《原神》角色插图数据集
Genshin_Impact_Illustration_ds = load_dataset("svjack/Genshin-Impact-Illustration")["train"]
ds_size = len(Genshin_Impact_Illustration_ds)
name_image_dict = {}
for i in range(ds_size):
    row_dict = Genshin_Impact_Illustration_ds[i]
    name_image_dict[row_dict["name"]] = row_dict["image"]

# 中英文映射字典
name_mapping = {
    '卡齐娜': 'Kachina',
    '玛拉妮': 'Maranee',
    '那维莱特': 'Navilette',
    '菲米尼': 'Ferminet',
    '娜维娅': 'Navia',
    '阿蕾奇诺': 'Arlecchino',
    '夏沃蕾': 'Chevreuse',
    '克洛琳德': 'Clorinde',
    '林尼': 'Lyney',
    '琳妮特': 'Lynette',
    '希格雯': 'Sigewinne',
    '夏洛蒂': 'Charlotte',
    '芙宁娜': 'Furina',
    '千织': 'Chiori',
    '莱欧斯利': 'Wriothesley',
    '艾梅莉埃': 'Emilie',
    '珐露珊': 'Faruzan',
    '纳西妲': 'Nahida',
    '卡维': 'Kaveh',
    '妮露': 'Nilou',
    '多莉': 'Dori',
    '坎蒂丝': 'Candace',
    '迪希雅': 'Dehya',
    '流浪者': 'Wanderer',
    '提纳里': 'Tighnari',
    '艾尔海森': 'Alhaitham',
    '赛索斯': 'Sethos',
    '赛诺': 'Cyno',
    '柯莱': 'Collei',
    '莱依拉': 'Layla',
    '行秋': 'Xingqiu',
    '申鹤': 'Shenhe',
    '辛焱': 'Xinyan',
    '瑶瑶': 'Yaoyao',
    '重云': 'Chongyun',
    '七七': 'Qiqi',
    '刻晴': 'Keqing',
    '夜兰': 'Yelan',
    '魈': 'Xiao',
    '达达利亚': 'Tartaglia',
    '甘雨': 'Ganyu',
    '云堇': 'Yunjin',
    '香菱': 'Xiangling',
    '嘉明': 'Gaming',
    '烟绯': 'Yanfei',
    '闲云': 'Xianyun',
    '北斗': 'Beidou',
    '白术': 'Baizhu',
    '钟离': 'Zhongli',
    '凝光': 'Ningguang',
    '胡桃': 'Hu Tao',
    '雷电将军': 'Raiden Shogun',
    '五郎': 'Gorou',
    '九条裟罗': 'Kujou Sara',
    '鹿野院平藏': 'Shikanoin Heizou',
    '早柚': 'Sayu',
    '八重神子': 'Yae Miko',
    '久岐忍': 'Kuki Shinobu',
    '绮良良': 'Kirara',
    '珊瑚宫心海': 'Sangonomiya Kokomi',
    '枫原万叶': 'Kaedehara Kazuha',
    '神里绫人': 'Kamisato Ayato',
    '神里绫华': 'Kamisato Ayaka',
    '宵宫': 'Yoimiya',
    '托马': 'Thoma',
    '荒泷一斗': 'Arataki Itto',
    '丽莎': 'Lisa',
    '莫娜': 'Mona',
    '安柏': 'Amber',
    '迪卢克': 'Diluc',
    '可莉': 'Klee',
    '凯亚': 'Kaeya',
    '诺艾尔': 'Noelle',
    '菲谢尔': 'Fischl',
    '罗莎莉亚': 'Rosaria',
    '砂糖': 'Sucrose',
    '优菈': 'Eula',
    '琴': 'Jean',
    '芭芭拉': 'Barbara',
    '米卡': 'Mika',
    '埃洛伊': 'Aloy',
    '班尼特': 'Bennett',
    '阿贝多': 'Albedo',
    '迪奥娜': 'Diona',
    '温迪': 'Venti',
    '雷泽': 'Razor'
}

# 将 name_image_dict 的键替换为英文
name_image_dict_en = {name_mapping[name]: image for name, image in name_image_dict.items()}

# 获取第一章的标题
first_sTitle = ds["train"][0]["sTitle"]
print(f"First chapter title: {first_sTitle}")

# 提取第一章的所有页面图片
chapter_pages_im = []
for i in range(len(ds["train"])):
    if ds["train"][i]["sTitle"] == first_sTitle:
        chapter_pages_im.append(ds["train"][i]["image"])

print(f"Number of pages in the first chapter: {len(chapter_pages_im)}")

# 加载预训练模型
model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()

# 读取图片的函数
def read_image(image):
    # 将 PIL.Image 转换为 RGB 格式的 numpy 数组
    image = image.convert("L").convert("RGB")
    image = np.array(image)
    return image

# 使用 chapter_pages_im 中的图片作为漫画页面
chapter_pages = [read_image(image) for image in chapter_pages_im]

# 使用 genshin_impact_images 中的图片作为角色图片
character_bank = {
    "images": [read_image(image) for image in name_image_dict_en.values()],  # 角色图片
    "names": list(name_image_dict_en.keys())  # 角色名称（图片路径中的名称）
}

# 使用模型进行预测
with torch.no_grad():
    per_page_results = model.do_chapter_wide_prediction(chapter_pages, character_bank, use_tqdm=True, do_ocr=True)

# 生成对话文本
transcript = []
for i, (image, page_result) in enumerate(zip(chapter_pages, per_page_results)):
    # 可视化预测结果并保存为图片
    output_path = os.path.join(output_dir, f"page_{i}.png")
    model.visualise_single_image_prediction(image, page_result, output_path)
    print(f"Saved visualization to {output_path}")
    
    # 获取说话者名称
    speaker_name = {
        text_idx: page_result["character_names"][char_idx] for text_idx, char_idx in page_result["text_character_associations"]
    }
    
    # 遍历每一页的 OCR 结果
    for j in range(len(page_result["ocr"])):
        if not page_result["is_essential_text"][j]:
            continue
        name = speaker_name.get(j, "unsure")  # 如果找不到对应的角色名称，使用 "unsure"
        transcript.append(f"<{name}>: {page_result['ocr'][j]}")

# 将对话文本保存到文件
transcript_path = os.path.join(output_dir, "transcript.txt")
with open(transcript_path, "w", encoding="utf-8") as fh:
    for line in transcript:
        fh.write(line + "\n")

print(f"Transcript saved to {transcript_path}")
```

- 使用tag 精确化判定人物名称的版本（可能后续还需要加入bbox 是否为人 等的判定，使用tag或者面积比率等）
- 也可以考虑使得图片不是灰度的
```python
#!/usr/bin/env python
# coding: utf-8

# 安装依赖
#get_ipython().system('pip install torch datasets huggingface_hub transformers scipy einops pulp shapely timm "dghs-imgutils[gpu]"')

# 导入必要的库
from datasets import load_dataset
import os
from PIL import Image
import numpy as np
from transformers import AutoModel
import torch
from imgutils.tagging import get_wd14_tags  # 导入标签预测函数

# 配置输出路径
output_dir = "first_sTitle_output_tag_pages"  # 可配置的输出路径
os.makedirs(output_dir, exist_ok=True)  # 确保路径存在

# 加载《原神》漫画的英文数据集
ds = load_dataset("svjack/Genshin-Impact-Manga-EN-US")

# 获取第一章的标题
first_sTitle = ds["train"][0]["sTitle"]
print(f"First chapter title: {first_sTitle}")

# 提取第一章的所有页面图片
chapter_pages_im = []
for i in range(len(ds["train"])):
    if ds["train"][i]["sTitle"] == first_sTitle:
        chapter_pages_im.append(ds["train"][i]["image"])

print(f"Number of pages in the first chapter: {len(chapter_pages_im)}")

# 加载预训练模型
model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()

# 读取图片的函数
def read_image(image, grayscale=False):
    # 将 PIL.Image 转换为 RGB 格式的 numpy 数组
    if grayscale:
        image = image.convert("L").convert("RGB")  # 转换为灰度图
    else:
        image = image.convert("RGB")  # 保持原色图
    image = np.array(image)
    return image

# 使用 chapter_pages_im 中的图片作为漫画页面
chapter_pages_grayscale = [read_image(image, grayscale=True) for image in chapter_pages_im]  # 灰度图用于推断
chapter_pages_color = [read_image(image, grayscale=False) for image in chapter_pages_im]  # 原色图用于绘制

# 使用空字典作为角色图片库
character_bank = {
    "images": [], "names": []
}

# 使用模型进行预测（使用灰度图）
with torch.no_grad():
    per_page_results = model.do_chapter_wide_prediction(chapter_pages_grayscale, character_bank, use_tqdm=True, do_ocr=True)

# 生成对话文本
transcript = []
for i, (image, page_result) in enumerate(zip(chapter_pages_color, per_page_results)):  # 使用原色图进行绘制
    # 更新角色名称
    for idx, bbox in enumerate(page_result["characters"]):
        # 裁剪图片
        x1, y1, x2, y2 = map(int, bbox)  # 将 bbox 转换为整数
        cropped_image = image[y1:y2, x1:x2]  # 裁剪图片

        # 将裁剪后的图片转换为 PIL.Image 对象
        cropped_image_pil = Image.fromarray(cropped_image)

        # 使用 get_wd14_tags 进行标签预测
        rating, features, chars = get_wd14_tags(cropped_image_pil)
        chars = dict(filter(lambda t2: "genshin_impact" in t2[0].lower(), chars.items()))

        # 获取得分最高的标签
        if chars:
            best_tag = max(chars, key=chars.get)
            page_result["character_names"][idx] = best_tag  # 更新角色名称

    # 打印更新后的角色名称
    print(f"Updated character names for page {i}: {page_result['character_names']}")

    # 获取说话者名称
    speaker_name = {
        text_idx: page_result["character_names"][char_idx] for text_idx, char_idx in page_result["text_character_associations"]
    }
    
    # 遍历每一页的 OCR 结果
    for j in range(len(page_result["ocr"])):
        if not page_result["is_essential_text"][j]:
            continue
        name = speaker_name.get(j, "unsure")  # 如果找不到对应的角色名称，使用 "unsure"
        transcript.append(f"<{name}>: {page_result['ocr'][j]}")

    # 可视化预测结果并保存为图片（使用原色图）
    output_path = os.path.join(output_dir, f"page_{i}.png")
    model.visualise_single_image_prediction(image, page_result, output_path)
    print(f"Saved visualization to {output_path}")

# 将对话文本保存到文件
transcript_path = os.path.join(output_dir, "transcript.txt")
with open(transcript_path, "w", encoding="utf-8") as fh:
    for line in transcript:
        fh.write(line + "\n")

print(f"Transcript saved to {transcript_path}")
```

![page_73 (1)](https://github.com/user-attachments/assets/335e6da3-1670-4753-a60a-e746d8eba11b)

- 音乐style transformation
```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
pip install git+https://github.com/facebookresearch/audiocraft.git
```

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=8)  # generate 8 seconds.

descriptions = ['happy rock', 'energetic EDM', 'sad jazz']

melody, sr = torchaudio.load('bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")

from IPython import display
display.Audio("bach.mp3")
display.Audio("0.wav")
display.Audio("1.wav")
display.Audio("2.wav")
```

```python
from IPython import display
display.Video("温迪.mp3")

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=1 * 60 + 24)  # generate 8 seconds.

descriptions = ['happy rock', 'energetic EDM', 'sad jazz']

melody, sr = torchaudio.load('温迪.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")

#### ffmpeg -i 0.wav -c:v libx264 -tune stillimage -c:a aac -b:a 192k -shortest 0.mp4
```




https://github.com/user-attachments/assets/5ccb2e56-d3c8-417c-926d-e04b3946ba80


- Another Text-to-X (have text to audio/music for sound or music)
- https://github.com/Alpha-VLLM/Lumina-T2X

# Table of Contents
1. [Magiv1](#magiv1)
2. [Magiv2](#magiv2)
3. [Datasets](#datasets)

# Magiv1
- The model is available at 🤗 [HuggingFace Model Hub](https://huggingface.co/ragavsachdeva/magi).
- Try it out for yourself using this 🤗 [HuggingFace Spaces Demo](https://huggingface.co/spaces/ragavsachdeva/the-manga-whisperer/) (no GPU, so slow).
- Basic model usage is provided below. Inspect [this file](https://huggingface.co/ragavsachdeva/magi/blob/main/modelling_magi.py) for more info.

![Magi_teaser](https://github.com/ragavsachdeva/magi/assets/26804893/0a6d44bc-12ef-4545-ab9b-577c77bdfd8a)

### v1 Usage
```python
from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import os

images = [
        "path_to_image1.jpg",
        "path_to_image2.png",
    ]

def read_image_as_np_array(image_path):
    with open(image_path, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

images = [read_image_as_np_array(image) for image in images]

model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()
with torch.no_grad():
    results = model.predict_detections_and_associations(images)
    text_bboxes_for_all_images = [x["texts"] for x in results]
    ocr_results = model.predict_ocr(images, text_bboxes_for_all_images)

for i in range(len(images)):
    model.visualise_single_image_prediction(images[i], results[i], filename=f"image_{i}.png")
    model.generate_transcript_for_single_image(results[i], ocr_results[i], filename=f"transcript_{i}.txt")
```

# Magiv2
- The model is available at 🤗 [HuggingFace Model Hub](https://huggingface.co/ragavsachdeva/magiv2).
- Try it out for yourself using this 🤗 [HuggingFace Spaces Demo](https://huggingface.co/spaces/ragavsachdeva/Magiv2-Demo) (with GPU, thanks HF Team!).
- Basic model usage is provided below. Inspect [this file](https://huggingface.co/ragavsachdeva/magiv2/blob/main/modelling_magiv2.py) for more info.

![magiv2](https://github.com/user-attachments/assets/e0cd1787-4a0c-49a5-a9d8-be2911d5ec08)

### v2 Usage
```python
from PIL import Image
import numpy as np
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()


def read_image(path_to_image):
    with open(path_to_image, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

chapter_pages = ["page1.png", "page2.png", "page3.png" ...]
character_bank = {
    "images": ["char1.png", "char2.png", "char3.png", "char4.png" ...],
    "names": ["Luffy", "Sanji", "Zoro", "Ussop" ...]
}

chapter_pages = [read_image(x) for x in chapter_pages]
character_bank["images"] = [read_image(x) for x in character_bank["images"]]

with torch.no_grad():
    per_page_results = model.do_chapter_wide_prediction(chapter_pages, character_bank, use_tqdm=True, do_ocr=True)

transcript = []
for i, (image, page_result) in enumerate(zip(chapter_pages, per_page_results)):
    model.visualise_single_image_prediction(image, page_result, f"page_{i}.png")
    speaker_name = {
        text_idx: page_result["character_names"][char_idx] for text_idx, char_idx in page_result["text_character_associations"]
    }
    for j in range(len(page_result["ocr"])):
        if not page_result["is_essential_text"][j]:
            continue
        name = speaker_name.get(j, "unsure") 
        transcript.append(f"<{name}>: {page_result['ocr'][j]}")
with open(f"transcript.txt", "w") as fh:
    for line in transcript:
        fh.write(line + "\n")
```

# Datasets

Disclaimer: In adherence to copyright regulations, we are unable to _publicly_ distribute the manga images that we've collected. The test images, however, are available freely, publicly and officially on [Manga Plus by Shueisha](https://mangaplus.shueisha.co.jp/).

[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20%20PopMangaX%20(Test)-Dataset-blue)](https://huggingface.co/datasets/ragavsachdeva/popmanga_test)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20%20PopCharacters-Dataset-blue)](https://huggingface.co/datasets/ragavsachdeva/popcharacters)

### Other notes
- Request to download Manga109 dataset [here](http://www.manga109.org/en/download.html).
- Download a large scale dataset from Mangadex using [this tool](https://github.com/EMACC99/mangadex).
- The Manga109 test splits are available here: [detection](https://github.com/barisbatuhan/DASS_Det_Inference/blob/main/dass_det/data/datasets/manga109.py#L109), [character clustering](https://github.com/kktsubota/manga-face-clustering/blob/master/dataset/test_titles.txt). Be careful that some background characters have the same label even though they are not the same character, [see](https://github.com/kktsubota/manga-face-clustering/blob/master/script/get_other_ids.py).



# License and Citation
The provided models and datasets are available for academic research purposes only.

```
@InProceedings{magiv1,
    author    = {Sachdeva, Ragav and Zisserman, Andrew},
    title     = {The Manga Whisperer: Automatically Generating Transcriptions for Comics},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {12967-12976}
}
```
```
@misc{magiv2,
      author={Ragav Sachdeva and Gyungin Shin and Andrew Zisserman},
      title={Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names}, 
      year={2024},
      eprint={2408.00298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00298}, 
}
```
