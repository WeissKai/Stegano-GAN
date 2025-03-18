# 使用CPU版本的基础镜像
FROM pytorch/pytorch:1.9.0-cpu

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY config.py .
COPY train.py .
COPY models/ models/
COPY utils/ utils/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p /app/data/train /app/data/test /app/checkpoints /app/logs

# 设置环境变量
ENV PYTHONPATH=/app

# 设置入口点
ENTRYPOINT ["python", "train.py"] 