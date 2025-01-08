import os
import json

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from App.model.zhm_model_1 import convnext_tiny as create_model


class ZhmPredict1:
    def __init__(self, model_path):
        pass

    @staticmethod
    def phase4orNo4Detect(img_url):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print(f"using {device} device.")

        num_classes = 2
        img_size = 512
        data_transform = transforms.Compose(
            [transforms.Resize(int(img_size * 1.14)),
             transforms.CenterCrop(img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # load image
        img_path = img_url
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        script_dir = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(script_dir, "../class_indices/zhm1_class_indices.json")
        json_path = os.path.abspath(json_path)
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_model(num_classes=num_classes).to(device)
        # load model weights
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "../weights/zhm1.pth")
        weights_path = os.path.abspath(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        res = class_indict[str(predict_cla)]
        if res == "noPhase4":
            res = "非四期"
        elif res == "phase4":
            res = "诊断为：四期"
        return res


if __name__ == '__main__':
    img_path = r"E:\python\flask_deploy\App\img\zhm\phase4.JPG"
    print(ZhmPredict1.phase4orNo4Detect(img_path))
