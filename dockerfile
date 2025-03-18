# 使用支持多架构的 PyTorch 镜像
FROM pytorch/pytorch:latest

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到容器的 /app 目录
COPY . /app

# 安装所需的 Python 库
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量（可选）
ENV PYTHONUNBUFFERED=1

# 运行程序
CMD ["python", "main.py"]
