# SteganoGAN

基于生成对抗网络的图像隐写技术实现

## 项目结构

```
SteganoGAN/
├── app.py              # Web应用主程序
├── config.py           # 配置文件
├── train.py            # 训练脚本
├── inference.py        # 推理脚本
├── requirements.txt    # 项目依赖
├── models/            # 模型定义
│   ├── generator.py
│   ├── discriminator.py
│   └── decoder.py
├── utils/            # 工具函数
│   ├── data_loader.py
│   └── metrics.py
├── data/            # 数据目录
│   ├── train/
│   └── test/
├── checkpoints/     # 模型检查点
├── logs/           # 训练日志
├── docker/         # Docker相关文件
│   ├── Dockerfile
│   ├── Dockerfile.inference
│   └── docker-compose.yml
└── templates/      # Web界面模板
    ├── index.html
    └── error.html
```

## 环境要求

- Python 3.8+
- CUDA 11.1+ (GPU版本)
- Docker Desktop for Mac/Windows/Linux
- NVIDIA Container Toolkit (GPU版本)
- 8GB+ RAM

## 安装

1. 克隆项目：
```bash
git clone https://github.com/yourusername/SteganoGAN.git
cd SteganoGAN
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## Docker部署

1. 构建镜像：
```bash
cd docker
docker-compose build
```

2. 启动服务：
```bash
docker-compose up -d
```

## 使用方法

### Web界面

1. 启动Web服务：
```bash
python app.py
```

2. 访问 http://localhost:5000 打开Web界面

3. 上传图片进行测试：
   - 选择一张图片上传
   - 等待处理完成
   - 查看原始图像、隐写后图像和消息可视化结果
   - 查看评估指标（PSNR、SSIM、消息提取准确率）

### 训练模型

```bash
python train.py
```

### 推理

```bash
python inference.py --image path/to/image.jpg
```

## 配置选项

在 `config.py` 中可以修改以下配置：

- IMAGE_SIZE: 图像大小
- BATCH_SIZE: 批次大小
- MESSAGE_LENGTH: 隐写消息长度
- LEARNING_RATE: 学习率
- EPOCHS: 训练轮数

## 重要说明

1. 数据准备
   - 将训练图像放在 `data/train` 目录
   - 将测试图像放在 `data/test` 目录

2. 硬件要求
   - GPU版本需要NVIDIA显卡支持
   - CPU版本可在所有平台运行，但处理速度较慢

3. Docker使用
   - GPU版本需要安装NVIDIA Container Toolkit
   - CPU版本可直接使用Docker Desktop

4. Web界面
   - 支持实时图像隐写演示
   - 提供直观的评估指标展示
   - 响应式设计，支持移动端访问

## 许可证

MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
