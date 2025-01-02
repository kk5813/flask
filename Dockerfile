# 使用 Python 3.10 镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 升级 pip 版本
RUN pip install --upgrade pip

# 安装所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 开放容器的 4091 端口
EXPOSE 4091

# 执行 app.py
CMD ["python", "app.py"]
