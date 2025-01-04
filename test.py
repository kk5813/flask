import os
from PIL import Image

# 指定文件夹的路径
folder_path = "/opt/resources/images"

# 检查文件夹是否存在
assert os.path.exists(folder_path), f"文件夹 {folder_path} 不存在！"

# 输出文件夹中的所有文件名，并查找第一个 .png 或 .jpg 文件
print(f"文件夹 {folder_path} 中的文件：")
image_found = False

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and (filename.endswith('.png') or filename.endswith('.jpg')):
        print(f"找到图片文件: {filename}")
        # 读取图片
        with Image.open(file_path) as img:
            img.show()  # 显示图片
        image_found = True
        break  # 找到第一个文件后退出循环

if not image_found:
    print("没有找到 .png 或 .jpg 文件。")
