import os
import torch
import numpy as np
from models.generator import Generator
from models.decoder import Decoder
from utils.metrics import compute_psnr, compute_ssim, compute_message_accuracy, compute_steganography_capacity
from utils.model_utils import count_parameters, calculate_model_size, test_inference_speed, prune_model

def test_evaluation():
    """测试评估指标和模型优化功能"""
    print("=== 开始测试评估指标和优化功能 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("\n初始化模型...")
    generator = Generator(input_channels=3, output_channels=3).to(device)
    decoder = Decoder(input_channels=3, message_length=100).to(device)
    
    # 测试模型信息
    print("\n模型信息:")
    print(f"生成器参数数量: {count_parameters(generator):,}")
    print(f"解码器参数数量: {count_parameters(decoder):,}")
    
    print(f"生成器大小: {calculate_model_size(generator):.2f} MB")
    print(f"解码器大小: {calculate_model_size(decoder):.2f} MB")
    
    # 测试推理速度
    print("\n测试推理速度:")
    # 创建随机输入
    random_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 测量生成器推理速度
    generator_inference_time = test_inference_speed(generator, random_input, num_iterations=10, warm_up=2)
    print(f"生成器推理时间: {generator_inference_time:.2f} ms/样本")
    
    # 生成隐藏图像
    with torch.no_grad():
        hidden_image = generator(random_input)
    
    # 测量解码器推理速度
    decoder_inference_time = test_inference_speed(decoder, hidden_image, num_iterations=10, warm_up=2)
    print(f"解码器推理时间: {decoder_inference_time:.2f} ms/样本")
    
    # 测试评估指标
    print("\n测试评估指标:")
    
    # 创建模拟数据
    original_image = torch.randn(2, 3, 256, 256).to(device)
    generated_image = original_image + 0.1 * torch.randn(2, 3, 256, 256).to(device)
    original_message = torch.randn(2, 100).to(device)
    extracted_message = original_message + 0.1 * torch.randn(2, 100).to(device)
    
    # 计算PSNR
    psnr_value = compute_psnr(original_image, generated_image)
    print(f"PSNR: {psnr_value:.2f} dB")
    
    # 计算SSIM
    ssim_value = compute_ssim(original_image, generated_image)
    print(f"SSIM: {ssim_value:.4f}")
    
    # 计算消息提取准确率
    message_acc = compute_message_accuracy(original_message, extracted_message)
    print(f"消息提取准确率: {message_acc:.4f}")
    
    # 计算隐写容量
    message_length = 100
    image_size = (256, 256)
    capacity = compute_steganography_capacity(message_length, image_size)
    print(f"隐写容量: {capacity:.6f} bpp")
    
    # 测试模型剪枝
    print("\n测试模型剪枝:")
    
    # 剪枝前的参数统计
    gen_params_before = count_parameters(generator)
    dec_params_before = count_parameters(decoder)
    
    # 执行剪枝
    pruned_generator = prune_model(generator.cpu(), pruning_rate=0.3)
    pruned_decoder = prune_model(decoder.cpu(), pruning_rate=0.3)
    
    # 剪枝后的参数统计（注意：实际参数数量不会变，只是部分权重被设为0）
    # 这里我们计算非零参数的数量
    non_zero_gen_params = sum(
        (param != 0).float().sum().item() 
        for name, param in pruned_generator.named_parameters() if 'weight' in name
    )
    non_zero_dec_params = sum(
        (param != 0).float().sum().item() 
        for name, param in pruned_decoder.named_parameters() if 'weight' in name
    )
    
    print(f"剪枝前生成器参数数量: {gen_params_before:,}")
    print(f"剪枝后生成器非零参数数量: {non_zero_gen_params:,.0f}")
    print(f"剪枝后生成器参数稀疏度: {(1 - non_zero_gen_params / gen_params_before) * 100:.2f}%")
    
    print(f"剪枝前解码器参数数量: {dec_params_before:,}")
    print(f"剪枝后解码器非零参数数量: {non_zero_dec_params:,.0f}")
    print(f"剪枝后解码器参数稀疏度: {(1 - non_zero_dec_params / dec_params_before) * 100:.2f}%")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_evaluation() 