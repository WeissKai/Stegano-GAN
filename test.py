import torch
import torch.nn as nn
import os
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.generator import Generator
from models.discriminator import Discriminator
from models.decoder import Decoder
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_message_accuracy, compute_steganography_capacity
from utils.model_utils import count_parameters, calculate_model_size, test_inference_speed, prune_model

def evaluate_model(generator_path, discriminator_path, decoder_path, batch_size=16, num_samples=100):
    """
    评估模型性能
    参数:
        generator_path: 生成器模型路径
        discriminator_path: 判别器模型路径
        decoder_path: 解码器模型路径
        batch_size: 批次大小
        num_samples: 评估样本数量
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("\n加载模型...")
    generator = Generator(input_channels=3, output_channels=3).to(device)
    discriminator = Discriminator(input_channels=3).to(device)
    decoder = Decoder(input_channels=3, message_length=100).to(device)
    
    # 加载模型权重
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    # 设置模型为评估模式
    generator.eval()
    discriminator.eval()
    decoder.eval()
    
    # 获取数据加载器
    print("\n加载数据...")
    _, test_loader = get_data_loaders(batch_size=batch_size)
    
    # 模型信息
    print("\n模型信息:")
    print(f"生成器参数数量: {count_parameters(generator):,}")
    print(f"判别器参数数量: {count_parameters(discriminator):,}")
    print(f"解码器参数数量: {count_parameters(decoder):,}")
    
    print(f"生成器大小: {calculate_model_size(generator):.2f} MB")
    print(f"判别器大小: {calculate_model_size(discriminator):.2f} MB")
    print(f"解码器大小: {calculate_model_size(decoder):.2f} MB")
    
    # 评估指标
    print("\n开始评估...")
    psnr_values = []
    ssim_values = []
    message_acc_values = []
    inference_times = []
    
    # 示例图像保存目录
    sample_dir = 'samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    num_batches = min(num_samples // batch_size, len(test_loader))
    
    with torch.no_grad():
        for i, (images, messages) in enumerate(tqdm(test_loader, total=num_batches)):
            if i >= num_batches:
                break
                
            images, messages = images.to(device), messages.to(device)
            
            # 测量推理时间
            start_time = time.time()
            hidden_images = generator(images)
            extracted_messages = decoder(hidden_images)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000 / batch_size  # 毫秒/样本
            inference_times.append(inference_time)
            
            # 计算评估指标
            psnr_value = compute_psnr(images, hidden_images)
            ssim_value = compute_ssim(images, hidden_images)
            message_acc = compute_message_accuracy(messages, extracted_messages)
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            message_acc_values.append(message_acc)
            
            # 保存第一批次的示例图像
            if i == 0:
                for j in range(min(5, batch_size)):
                    # 原始图像
                    orig_img = (images[j].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
                    # 隐藏图像
                    hidden_img = (hidden_images[j].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
                    
                    # 原始消息和提取消息（二值化）
                    orig_msg = (messages[j].detach().cpu().numpy() + 1) / 2 > 0.5
                    extract_msg = (extracted_messages[j].detach().cpu().numpy() + 1) / 2 > 0.5
                    
                    # 可视化
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 4, 1)
                    plt.imshow(orig_img)
                    plt.title('原始图像')
                    plt.axis('off')
                    
                    plt.subplot(1, 4, 2)
                    plt.imshow(hidden_img)
                    plt.title('隐藏图像')
                    plt.axis('off')
                    
                    plt.subplot(1, 4, 3)
                    plt.plot(orig_msg.astype(int))
                    plt.title('原始消息')
                    plt.ylim(-0.1, 1.1)
                    
                    plt.subplot(1, 4, 4)
                    plt.plot(extract_msg.astype(int))
                    plt.title('提取消息')
                    plt.ylim(-0.1, 1.1)
                    
                    plt.tight_layout()
                    plt.savefig(f'{sample_dir}/sample_{j}.png')
                    plt.close()
    
    # 计算平均指标
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_message_acc = sum(message_acc_values) / len(message_acc_values)
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    # 计算隐写容量
    message_length = 100  # 消息长度
    image_size = (256, 256)  # 图像大小
    capacity = compute_steganography_capacity(message_length, image_size)
    
    # 输出评估结果
    print("\n评估结果:")
    print(f"平均PSNR: {avg_psnr:.2f} dB")
    print(f"平均SSIM: {avg_ssim:.4f}")
    print(f"平均消息提取准确率: {avg_message_acc:.4f}")
    print(f"隐写容量: {capacity:.6f} bpp")
    print(f"平均推理时间: {avg_inference_time:.2f} ms/样本")
    
    # 模型剪枝和优化
    print("\n模型优化...")
    print("执行剪枝以优化模型大小...")
    pruned_generator = prune_model(generator.cpu(), pruning_rate=0.3)
    pruned_decoder = prune_model(decoder.cpu(), pruning_rate=0.3)
    
    print(f"剪枝后生成器大小: {calculate_model_size(pruned_generator):.2f} MB")
    print(f"剪枝后解码器大小: {calculate_model_size(pruned_decoder):.2f} MB")
    
    # 保存优化后的模型
    print("\n保存优化后的模型...")
    opt_dir = 'optimized_models'
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    
    torch.save(pruned_generator.state_dict(), f'{opt_dir}/generator_optimized.pth')
    torch.save(pruned_decoder.state_dict(), f'{opt_dir}/decoder_optimized.pth')
    
    print("\n评估完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SteganoGAN模型评估')
    parser.add_argument('--generator', type=str, default='checkpoints/generator_test.pth', help='生成器模型路径')
    parser.add_argument('--discriminator', type=str, default='checkpoints/discriminator_test.pth', help='判别器模型路径')
    parser.add_argument('--decoder', type=str, default='checkpoints/decoder_test.pth', help='解码器模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_samples', type=int, default=100, help='评估样本数量')
    args = parser.parse_args()
    
    evaluate_model(
        args.generator,
        args.discriminator,
        args.decoder,
        args.batch_size,
        args.num_samples
    ) 