# 使用 Python 3.9 镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装系统依赖（包括 OpenCV 所需的 libGL.so.1 和其他图形相关库）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev

# 升级 pip 版本
RUN pip install --upgrade pip

# 安装所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 执行 app.py
CMD ["python", "app.py"]
