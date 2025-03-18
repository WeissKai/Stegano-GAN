import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from models.generator import Generator
from models.decoder import Decoder
from utils.metrics import compute_psnr, compute_ssim, compute_message_accuracy
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates', static_folder='static')

# 全局变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = None
decoder = None
model_loaded = False

def load_models():
    global generator, decoder, model_loaded
    
    if model_loaded:
        return True
    
    try:
        # 初始化模型
        generator = Generator(input_channels=3, output_channels=3).to(device)
        decoder = Decoder(input_channels=3, message_length=100).to(device)
        
        # 加载模型权重
        model_dir = 'checkpoints'
        generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator_final.pth'), map_location=device))
        decoder.load_state_dict(torch.load(os.path.join(model_dir, 'decoder_final.pth'), map_location=device))
        
        # 设置为评估模式
        generator.eval()
        decoder.eval()
        
        model_loaded = True
        return True
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

def preprocess_image(image):
    # 调整图像大小
    image = image.resize((256, 256))
    # 转换为张量
    img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor

def generate_message(length=100):
    # 生成随机消息
    return torch.rand(1, length).mul(2).sub(1).to(device)

def encode_image(image_tensor, message_tensor):
    with torch.no_grad():
        # 生成隐藏图像
        hidden_image = generator(image_tensor)
        # 从隐藏图像中提取消息
        extracted_message = decoder(hidden_image)
        
    # 计算指标
    psnr_value = compute_psnr(image_tensor, hidden_image)
    ssim_value = compute_ssim(image_tensor, hidden_image)
    message_acc = compute_message_accuracy(message_tensor, extracted_message)
    
    # 转换为PIL图像
    orig_img = (image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    orig_img = (orig_img * 255).astype(np.uint8)
    
    hidden_img = (hidden_image[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
    hidden_img = (hidden_img * 255).astype(np.uint8)
    
    # 创建消息可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(message_tensor[0].detach().cpu().numpy())
    plt.title('原始消息')
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(extracted_message[0].detach().cpu().numpy())
    plt.title('提取消息')
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return {
        'original_image': Image.fromarray(orig_img),
        'hidden_image': Image.fromarray(hidden_img),
        'message_plot': buf,
        'psnr': psnr_value,
        'ssim': ssim_value,
        'accuracy': message_acc
    }

def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    if not load_models():
        return render_template('error.html', error="模型加载失败，请确保模型文件存在。")
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        # 读取上传的图像
        image = Image.open(image_file).convert('RGB')
        
        # 预处理图像
        image_tensor = preprocess_image(image)
        
        # 生成随机消息
        message_tensor = generate_message()
        
        # 编码图像
        result = encode_image(image_tensor, message_tensor)
        
        # 将图像转换为base64字符串
        original_base64 = image_to_base64(result['original_image'])
        hidden_base64 = image_to_base64(result['hidden_image'])
        
        # 将消息图转换为base64字符串
        message_buf = result['message_plot']
        message_base64 = base64.b64encode(message_buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'original_image': original_base64,
            'hidden_image': hidden_base64,
            'message_plot': message_base64,
            'psnr': f"{result['psnr']:.2f} dB",
            'ssim': f"{result['ssim']:.4f}",
            'accuracy': f"{result['accuracy']:.4f}"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 