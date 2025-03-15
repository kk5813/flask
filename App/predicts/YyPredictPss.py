import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from App.model.yy_regnet_model import create_regnet


class YyPredictPss:
    def __init__(self, model_path):
        pass

    @staticmethod
    def predict_pss(img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # load image
        img_path = img
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        # json_path = r'E:\python\flask_deploy\App\class_indices\yy_pss_class_indices.json'
        script_dir = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(script_dir, "../class_indices/yy_pss_class_indices.json")
        json_path = os.path.abspath(json_path)
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_regnet(model_name="RegNetY_400MF", num_classes=2).to(device)
        # load model weights
        # model_weight_path = r"E:\python\flask_deploy\App\weights\yy_regnet_model-27.pth"
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "../weights/yy_regnet_model-27.pth")
        model_weight_path = os.path.abspath(weights_path)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                               predict[i].numpy()))
        # plt.show()
        return class_indict[str(predict_cla)]


if __name__ == '__main__':
    img_path = '../img/yy/001958-20191029@150326-R1-S.jpg'
    res = YyPredictPss.predict_pss(img_path)
    print(res)
'''
调用输出示例：
class: normal   prob: 0.565
'''
