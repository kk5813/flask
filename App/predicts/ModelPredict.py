import os
import json

import torch
from PIL import Image
from torchvision import transforms

from App.model.quality_model import efficientnetv2_l as create_model


class ModelPredicts:
    def __init__(self, model_path):
        pass

    @staticmethod
    def quality_classification(img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        img_size = {"s": [300, 384],  # train_size, val_size
                    "m": [384, 480],
                    "l": [512, 512]}
        num_model = "l"
        data_transform = transforms.Compose(
            [transforms.Resize(img_size[num_model][1]),
             transforms.CenterCrop(img_size[num_model][1]),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # load image
        # img_path = "./plot_img/011253-20201121@105551-L2-S.jpg"
        # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        # img = Image.open(img_path)

        # read class_indict
        json_path = 'App/class_indices/quality_class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_model(num_classes=4).to(device)
        # load model weights
        model_weight_path = "App/weights/quality_l/model-99.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))

        print("模型加载完成----")
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        for i in range(len(predict)):
            if i == predict_cla:
                print("该图片类别为:", class_indict[str(i)])
        return predict_cla
