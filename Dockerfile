# 使用 Python 3.9 镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 升级 pip 并单独安装 pymupdf==1.25.1
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir pymupdf==1.18.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmupdf-dev \
    libjpeg-dev \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# 安装其他 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动应用
CMD ["python", "app.py"]