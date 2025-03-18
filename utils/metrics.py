import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_psnr(original, generated):
    """
    计算峰值信噪比（PSNR）
    参数:
        original: 原始图像，形状为 [B, C, H, W]
        generated: 生成图像，形状为 [B, C, H, W]
    返回:
        psnr_value: PSNR值，标量
    """
    # 转换为numpy数组
    original = original.detach().cpu().numpy().transpose(0, 2, 3, 1)
    generated = generated.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # 归一化到[0, 1]范围
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)
    
    batch_size = original.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        psnr_values.append(psnr(original[i], generated[i], data_range=1.0))
    
    return np.mean(psnr_values)

def compute_ssim(original, generated):
    """
    计算结构相似性（SSIM）
    参数:
        original: 原始图像，形状为 [B, C, H, W]
        generated: 生成图像，形状为 [B, C, H, W]
    返回:
        ssim_value: SSIM值，标量
    """
    # 转换为numpy数组
    original = original.detach().cpu().numpy().transpose(0, 2, 3, 1)
    generated = generated.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # 归一化到[0, 1]范围
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)
    
    batch_size = original.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        ssim_values.append(ssim(original[i], generated[i], channel_axis=2, data_range=1.0))
    
    return np.mean(ssim_values)

def compute_message_accuracy(original_message, extracted_message, threshold=0.5):
    """
    计算消息提取准确率
    参数:
        original_message: 原始消息，形状为 [B, M]
        extracted_message: 提取的消息，形状为 [B, M]
        threshold: 二值化阈值，默认为0.5
    返回:
        accuracy: 准确率，标量
    """
    # 转换为numpy数组
    original_message = original_message.detach().cpu().numpy()
    extracted_message = extracted_message.detach().cpu().numpy()
    
    # 将范围从[-1, 1]转换为[0, 1]，然后进行二值化
    original_message = (original_message + 1) / 2 > threshold
    extracted_message = (extracted_message + 1) / 2 > threshold
    
    # 计算准确率
    accuracy = np.mean(original_message == extracted_message)
    
    return accuracy

def compute_steganography_capacity(message_length, image_size):
    """
    计算隐写容量（每像素比特数）
    参数:
        message_length: 消息长度，标量
        image_size: 图像大小，元组 (H, W)
    返回:
        capacity: 隐写容量（bpp），标量
    """
    total_pixels = image_size[0] * image_size[1]
    capacity = message_length / total_pixels
    
    return capacity 