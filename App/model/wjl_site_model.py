import os

import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18


def SitePredict(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = resnet18(weights=None)
    # 修改最后一层全连接层（fc 层）以适应 2 分类任务
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(script_dir, "../weights/wjl_resnet18.pth")
    weights_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights_dict)  # 加载权重
    model.to(device)  # 将模型移动到 GPU
    # 设置为评估模式
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为 3 通道
        transforms.Normalize(mean=0.5, std=0.5)

    ])

    # 加载并预处理图像
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()