import torch
import torch.nn as nn
import time
import numpy as np
import os
from torch.optim import Adam

def count_parameters(model):
    """
    计算模型的参数数量
    参数:
        model: PyTorch模型
    返回:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_model_size(model, save_path=None):
    """
    计算模型的大小（MB）
    参数:
        model: PyTorch模型
        save_path: 如果提供，则保存模型到该路径并计算保存文件的大小
    返回:
        模型大小（MB）
    """
    if save_path:
        # 保存模型
        torch.save(model.state_dict(), save_path)
        # 计算文件大小
        size_in_bytes = os.path.getsize(save_path)
        size_in_mb = size_in_bytes / (1024 * 1024)
        return size_in_mb
    else:
        # 估计模型大小
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_in_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_in_mb

def test_inference_speed(model, input_tensor, num_iterations=100, warm_up=10):
    """
    测试模型的推理速度
    参数:
        model: PyTorch模型
        input_tensor: 输入张量
        num_iterations: 测试迭代次数
        warm_up: 预热迭代次数
    返回:
        平均推理时间（毫秒）
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(input_tensor)
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            times.append(end_time - start_time)
    
    # 计算平均推理时间（毫秒）
    avg_time = np.mean(times) * 1000
    
    return avg_time

def optimize_learning_rate(model, train_loader, init_lr=0.001, lr_range=(0.0001, 0.01), num_epochs=5):
    """
    使用学习率范围测试来优化学习率
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        init_lr: 初始学习率
        lr_range: 学习率范围，元组 (min_lr, max_lr)
        num_epochs: 测试的轮数
    返回:
        最佳学习率
    """
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=init_lr)
    
    # 创建学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=np.exp(np.log(lr_range[1] / lr_range[0]) / (num_epochs * len(train_loader)))
    )
    
    losses = []
    lrs = []
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        for inputs, _ in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # 后向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录学习率和损失
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            
            # 更新学习率
            lr_scheduler.step()
    
    # 找到损失最小的学习率
    min_loss_idx = np.argmin(losses)
    best_lr = lrs[min_loss_idx]
    
    return best_lr

def prune_model(model, pruning_rate=0.3):
    """
    剪枝模型，移除不重要的权重
    参数:
        model: PyTorch模型
        pruning_rate: 剪枝率，即要移除的参数比例
    返回:
        剪枝后的模型
    """
    # 对每个参数应用掩码
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 计算阈值
            threshold = torch.quantile(torch.abs(param.data), pruning_rate)
            # 创建掩码
            mask = torch.abs(param.data) > threshold
            # 应用掩码
            param.data.mul_(mask.float())
    
    return model 