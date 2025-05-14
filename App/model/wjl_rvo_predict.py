import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torchvision.models import resnet34


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
    model = resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    weights = r"D:\wjl\pro\RVO_res\weight\UWF\resnet34\4\resnet34.pth"
    model.to(device)  # 将模型移动到 GPU
    # 设置为评估模式
    model.eval()
    # 预测
    # image_path = r"D:\wjl\pro\RVO\imgs\0000023448-20230508@105947-R1-S.jpg"  # BACH
    image_path = r"D:\wjl\pro\RVO\imgs\022233-20220523@090830-L2-S.jpg"  # BACH
    prediction = predict(image_path, model, device)
    print(f'Predicted class: {prediction}')