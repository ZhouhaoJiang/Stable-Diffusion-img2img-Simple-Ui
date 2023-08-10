from Deepbooru import deep_danbooru_model
from translate import Translator
from PIL import Image
import gradio as gr
import numpy as np
import torch
import tqdm

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))

model.eval()
model.half()
model.cuda()


def deepdanbooru(img_and_mask):
    # print(img_and_mask)
    try:
        img = img_and_mask["image"]
        pic = img.convert("RGB").resize((512, 512))
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(a).cuda()

            y = model(x)[0].detach().cpu().numpy()

            for n in tqdm.tqdm(range(10)):
                model(x)

            tags = ""
            for i, p in enumerate(y):
                if p >= 0.5:
                    tags += model.tags[i] + ","

            return tags
    except Exception as e:
        print(e)
        raise gr.Error("请先上传图片")

