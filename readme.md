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

```python
#!/usr/bin/env python
# coding: utf-8

# 安装依赖
get_ipython().system('pip install torch datasets huggingface_hub transformers scipy einops pulp shapely timm')

# 导入必要的库
from datasets import load_dataset
import os
from PIL import Image
import numpy as np
from transformers import AutoModel
import torch

# 加载《原神》漫画的英文数据集
ds = load_dataset("svjack/Genshin-Impact-Manga-EN-US")

# 加载《原神》角色插图数据集
Genshin_Impact_Illustration_ds = load_dataset("svjack/Genshin-Impact-Illustration")["train"]
ds_size = len(Genshin_Impact_Illustration_ds)
name_image_dict = {}
for i in range(ds_size):
    row_dict = Genshin_Impact_Illustration_ds[i]
    name_image_dict[row_dict["name"]] = row_dict["image"]

# 保存图片的函数
def save_images(name_image_dict, output_dir):
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历字典，保存图片
    for name, image in name_image_dict.items():
        # 构造文件路径
        file_path = os.path.join(output_dir, f"{name}.png")  # 假设保存为 PNG 格式
        # 保存图片
        image.save(file_path)
        print(f"Saved {file_path}")

# 示例：保存图片到指定路径
output_directory = "genshin_impact_images"  # 替换为你想保存图片的路径
save_images(name_image_dict, output_directory)

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
    "images": [read_image(image) for image in name_image_dict.values()],  # 角色图片
    "names": list(name_image_dict.keys())  # 角色名称（图片路径中的名称）
}

# 使用模型进行预测
with torch.no_grad():
    per_page_results = model.do_chapter_wide_prediction(chapter_pages, character_bank, use_tqdm=True, do_ocr=True)

# 生成对话文本
transcript = []
for i, (image, page_result) in enumerate(zip(chapter_pages, per_page_results)):
    # 可视化预测结果并保存为图片
    model.visualise_single_image_prediction(image, page_result, f"page_{i}.png")
    
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
with open("transcript.txt", "w") as fh:
    for line in transcript:
        fh.write(line + "\n")

print("Transcript saved to transcript.txt")
```

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
