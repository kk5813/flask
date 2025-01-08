import fitz  # PyMuPDF

def extract_pdf_content(pdf_path):
    # 打开PDF文件
    doc = fitz.open(pdf_path)

    # 获取第一页
    page = doc.load_page(0)

    # 提取文本内容
    text = page.get_text("text")
    print("Text Content:")
    print("Format: Text")
    print(text)
    print("="*50)

    # 提取矢量图形路径（包括曲线、直线等）
    shapes = page.get_drawings()
    print("Shape Content:")
    print("Format: Vector Graphics (Path/Curve/Line)")
    for shape in shapes:
        for item in shape['items']:
            if item[0] == 'curve':
                print("Curve:", item[1])  # 曲线的坐标数据
            elif item[0] == 'line':
                print("Line:", item[1])  # 直线的坐标数据
    print("="*50)

    # 提取嵌入的图像
    images = page.get_images(full=True)
    print("Image Content:")
    print("Format: Image (Embedded)")
    for img in images:
        xref = img[0]  # 图像的xref标识符
        image = doc.extract_image(xref)
        image_bytes = image["image"]
        print(f"Image XREF: {xref}, Image Size: {len(image_bytes)} bytes")
    print("="*50)

# 使用函数提取内容并打印
pdf_path = r"E:\python\flask_deploy\App\img\阿追-光学相干断层成像（OCT）.pdf"  # 请替换成你的PDF文件路径
extract_pdf_content(pdf_path)
