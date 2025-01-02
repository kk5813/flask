import os
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from App.model.quality_model import efficientnetv2_l as create_model
from App.model.zhm_model_0 import u2net_full


class ZhmPredict0:
    def __init__(self, model_path):
        pass

    @staticmethod
    def abnormalDetect(img_url):
        # 获取当前脚本的绝对路径
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "../weights/zhm0.pth")
        weights_path = os.path.abspath(weights_path)
        img_path = img_url

        threshold = 0.5

        assert os.path.exists(img_path), f"image file {img_path} dose not exists."

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(640),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        h, w = origin_img.shape[:2]
        img = data_transform(origin_img)
        img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

        model = u2net_full()
        weights = torch.load(weights_path, map_location='cpu')
        if "model" in weights:
            model.load_state_dict(weights["model"])
        else:
            model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            # t_start = time_synchronized()
            pred = model(img)
            # t_end = time_synchronized()
            # print("inference time: {}".format(t_end - t_start))
            pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

            pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            pred_mask = np.where(pred > threshold, 1, 0)
            image = Image.fromarray(pred_mask.astype(np.uint8) * 255)
            pixel_data = list(image.getdata())
            total_count = len(pixel_data)
            count_255 = pixel_data.count(255)
            result = count_255 / total_count
            return result


if __name__ == '__main__':
    img_path = r"E:\python\flask_deploy\App\img\zhm\abnormal.JPG"
    print(ZhmPredict0.abnormalDetect(img_path))
