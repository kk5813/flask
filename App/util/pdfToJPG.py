# -*- coding = utf-8 -*-
# @Time :2024/11/26 20:26
# @Author:sls
# @File:test.py
# @Annotation:

import os
import cv2
import fitz
import numpy as np
import uuid  # 导入 uuid 模块


def render_pdf_page_as_image(pdf_path, output_folder, dpi=300):
    """
    将 PDF 的每一页渲染为高分辨率图片，并使用 UUID 命名。
    :param pdf_path: PDF 文件路径
    :param output_folder: 保存图片的文件夹路径
    :param dpi: 图像分辨率（默认300）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_document = fitz.Document(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]

        # 设置渲染分辨率
        zoom = dpi / 36  # PyMuPDF 默认分辨率为 72 DPI
        mat = fitz.Matrix(zoom, zoom)

        # 渲染页面为图片
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # 使用 UUID 生成唯一的文件名
        unique_filename = str(uuid.uuid4()) + '.png'
        image_path = os.path.join(output_folder, unique_filename)
        pix.save(image_path)

        # print(f"页面 {page_number + 1} 渲染完成：{image_path}，大小：{pix.width}x{pix.height}")

    pdf_document.close()


def extract_and_crop_images(image_path, output_dir):
    """
    提取图片中的区域并裁剪，去除上下的文字部分，保存为单独的文件。

    Parameters:
    - image_path: 输入图片路径
    - output_dir: 输出文件夹路径，用于保存提取的图像

    Returns:
    - cropped_image_paths: 裁剪后的图像路径列表
    """
    # 加载图片
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值化方法来去除文字部分
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # 反转图像，将文字变为白色

    # 进行形态学操作（膨胀）来连接文字区域，帮助更好地识别文字区域
    kernel = np.ones((5, 5), np.uint8)  # 定义结构元素
    thresh = cv2.dilate(thresh, kernel, iterations=2)  # 对阈值化图像进行膨胀处理

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 保存裁剪后的图像路径
    cropped_image_paths = []

    # 遍历每个轮廓，检查其大小并裁剪出可能的图像区域
    for i, contour in enumerate(contours):
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤掉不需要的区域，假设图片是左右排列的
        if w > 500 and h > 500:  # 可以根据图像的实际尺寸调整这个条件
            # 裁剪出感兴趣区域
            cropped_image = image[y:y + h, x:x + w]

            # 上面裁剪60px，下面裁剪50px
            cropped_image = cropped_image[85:-50, :]  # 上面裁剪60px，下面裁剪50px，保留整个宽度

            # 使用 UUID 生成唯一的文件名
            unique_filename = str(uuid.uuid4()) + '.png'
            output_image_path = os.path.join(output_dir, unique_filename)

            # 保存提取的图像
            cv2.imwrite(output_image_path, cropped_image)
            cropped_image_paths.append(output_image_path)
            # print(f'提取的图像已保存到 {output_image_path}')

    return cropped_image_paths


def process_pdf(pdf_path, rendered_folder, extracted_folder, dpi=300):
    """
    处理PDF文件：首先渲染每一页为图像，然后提取并裁剪感兴趣区域。
    :param pdf_path: PDF文件路径
    :param rendered_folder: 渲染图像保存文件夹
    :param extracted_folder: 提取裁剪后图像保存文件夹
    :param dpi: 渲染分辨率

    Returns:
    - all_cropped_image_paths: 所有裁剪后的图像路径列表
    """
    # 渲染PDF页面为图片
    render_pdf_page_as_image(pdf_path, rendered_folder, dpi)

    # 保存所有裁剪后的图像路径
    all_cropped_image_paths = []

    # 遍历渲染后的每一页图像并进行裁剪
    for image_file in os.listdir(rendered_folder):
        if image_file.endswith(".png"):
            image_path = os.path.join(rendered_folder, image_file)
            cropped_image_paths = extract_and_crop_images(image_path, extracted_folder)
            all_cropped_image_paths.extend(cropped_image_paths)

    return all_cropped_image_paths


if __name__ == '__main__':
    # 调用示例
    pdf_path = r'E:\python\Script\SLO6.pdf'
    rendered_folder = r'E:\python\flask_deploy\App\img\rendered'  # pdf -> img,临时文件
    extracted_folder = r'E:\python\flask_deploy\App\img\extracted'  # 提取后的图像保存文件夹路径

    # 处理PDF文件
    all_cropped_image_paths = process_pdf(pdf_path, rendered_folder, extracted_folder, dpi=300)
    print("所有裁剪后的图像路径：", all_cropped_image_paths)
