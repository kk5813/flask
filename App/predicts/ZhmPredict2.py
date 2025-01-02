import os
import json

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO
import uuid
from App.model.quality_model import efficientnetv2_l as create_model


class ZhmPredict2:
    def __init__(self, model_path):
        pass

    @staticmethod
    def quadrant_division(img):
        save_path = os.path.join("/", "zcc", "DownLoad", "project", "AI")
        img = Image.open(img)
        width, height = img.size
        # read class_indict
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "../weights/zhm2.pt")
        weights_path = os.path.abspath(weights_path)
        model = YOLO(weights_path)
        results = model.predict(source=img)
        # 遍历检测结果并计算中心点
        centers = []
        url = ""
        for r in results:
            # 提取边界框数据
            for box in r.boxes.data:
                x_min, y_min, x_max, y_max, *_ = box
                # 将 Tensor 转换为浮动数值并计算中心点
                x_center = (float(x_min) + float(x_max)) / 2
                y_center = (float(y_min) + float(y_max)) / 2
                centers.append((x_center, y_center))

                # 创建绘图对象
                draw = ImageDraw.Draw(img)
                # 绘制中心点
                draw.ellipse([x_center - 5, y_center - 5, x_center + 5, y_center + 5], fill="red")

                # 计算裁剪区域并确保坐标是整数
                x_center = int(x_center)
                y_center = int(y_center)

                # 裁剪四个象限
                quadrant1 = img.crop((x_center, 0, width, y_center))  # 右上
                quadrant2 = img.crop((0, 0, x_center, y_center))  # 左上
                quadrant3 = img.crop((0, y_center, x_center, height))  # 左下
                quadrant4 = img.crop((x_center, y_center, width, height))  # 右下

                # 展示裁剪后的四个象限
                # quadrant1.show(title="Quadrant 1 (Right-Top)")
                # quadrant2.show(title="Quadrant 2 (Left-Top)")
                # quadrant3.show(title="Quadrant 3 (Left-Bottom)")
                # quadrant4.show(title="Quadrant 4 (Right-Bottom)")

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                path = save_path + "\\" + str(uuid.uuid1()) + ".jpg"

                quadrant1.save(path)
                url = path

                path = save_path + "\\" + str(uuid.uuid1()) + ".jpg"
                quadrant2.save(path)
                url += "," + path

                path = save_path + "\\" + str(uuid.uuid1()) + ".jpg"
                quadrant3.save(path)
                url += "," + path

                path = save_path + "\\" + str(uuid.uuid1()) + ".jpg"
                quadrant4.save(path)
                url += "," + path

        return url


if __name__ == '__main__':
    url = ZhmPredict2.quadrant_division(r"E:\python\flask_deploy\App\img\zhm\bingzao.jpg")
    print(url)
