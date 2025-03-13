import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18


def predict(image_path, model, device):
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = resnet18(pretrained=False)
    # 修改最后一层全连接层（fc 层）以适应 2 分类任务
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    weights = r"E:\python\flask_deploy\App\weights\wjl_resnet18.pth"
    weights_dict = torch.load(weights, map_location=device)
    model.load_state_dict(weights_dict)  # 加载权重
    model.to(device)  # 将模型移动到 GPU
    # 设置为评估模式
    model.eval()
    # 预测
    image_path = r"E:\python\flask_deploy\App\img\001148-20190825@090824-L4-S.jpg"  # BACH
    prediction = predict(image_path, model, device)
    print(f'Predicted class: {prediction}')