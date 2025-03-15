import os
import json
import uuid
from datetime import datetime

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from App.API import save_path
from App.model.quality_model import efficientnetv2_l as create_model
from ultralytics import YOLO


class ZhmPredict3:
    def __init__(self, model_path):
        pass

    @staticmethod
    def lesion_detection(img, visitNumber):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        img = Image.open(img)  # Open the image
        draw = ImageDraw.Draw(img)  # Create drawing context
        # Load the YOLO model
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "../weights/zhm3.pt")
        weights_path = os.path.abspath(weights_path)
        model = YOLO(weights_path)
        results = model.predict(source=img)  # Perform prediction

        for r in results:
            for box in r.boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_id = box

                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                class_name = r.names[int(class_id)]

                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                text = f"{class_name} {confidence:.2f}"
                draw.text((x_min, y_min - 10), text, fill="red")
        now = datetime.now()
        image_save_path = os.path.join(save_path, "dr",
                                       now.strftime('%Y'), now.strftime('%m')
                                       , visitNumber)
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        path = image_save_path + "/" + str(uuid.uuid1()) + ".jpg"
        img.save(path)

        return path


if __name__ == '__main__':
    path = ZhmPredict3.lesion_detection(r"E:\python\flask_deploy\App\img\zhm\bingzao.jpg", "MZ20251222")
    print(path)