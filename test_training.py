import os
import torch
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from models.decoder import Decoder
from utils.data_loader import get_data_loaders
from utils.loss import total_loss

def test_training():
    print("=== 开始测试训练过程 ===")
    
    # 创建checkpoints目录
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("\n初始化模型...")
    generator = Generator(input_channels=3, output_channels=3).to(device)
    discriminator = Discriminator(input_channels=3).to(device)
    decoder = Decoder(input_channels=3, message_length=100).to(device)
    
    # 初始化优化器
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 获取数据加载器
    print("\n加载数据...")
    train_loader, test_loader = get_data_loaders()
    
    # 测试一个batch的训练
    print("\n测试一个batch的训练...")
    try:
        # 获取一个batch的数据
        images, messages = next(iter(train_loader))
        images, messages = images.to(device), messages.to(device)
        batch_size = images.size(0)
        
        print(f"Batch size: {batch_size}")
        print(f"Images shape: {images.shape}")
        print(f"Messages shape: {messages.shape}")
        
        # 生成隐藏图像
        hidden_images = generator(images)
        print(f"Generated images shape: {hidden_images.shape}")
        
        # 判别器前向传播
        real_output = discriminator(images)
        fake_output = discriminator(hidden_images)
        print(f"Discriminator outputs shape: {real_output.shape}")
        
        # 计算判别器损失
        d_loss = nn.BCELoss()(real_output, torch.ones(batch_size, 1).to(device)) + \
                 nn.BCELoss()(fake_output, torch.zeros(batch_size, 1).to(device))
        
        # 判别器反向传播
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
        # 生成器和解码器前向传播
        hidden_images = generator(images)
        fake_output = discriminator(hidden_images)
        extracted_messages = decoder(hidden_images)
        print(f"Extracted messages shape: {extracted_messages.shape}")
        
        # 计算生成器和解码器损失
        g_loss = total_loss(images, hidden_images, fake_output, 
                          torch.ones(batch_size, 1).to(device), 
                          messages, extracted_messages)
        
        # 生成器和解码器反向传播
        optimizer_g.zero_grad()
        optimizer_dec.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        optimizer_dec.step()
        
        print(f"\n训练成功！")
        print(f"判别器损失: {d_loss.item():.4f}")
        print(f"生成器损失: {g_loss.item():.4f}")
        
        # 测试模型保存
        print("\n测试模型保存...")
        torch.save(generator.state_dict(), 'checkpoints/generator_test.pth')
        torch.save(discriminator.state_dict(), 'checkpoints/discriminator_test.pth')
        torch.save(decoder.state_dict(), 'checkpoints/decoder_test.pth')
        print("模型保存成功！")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        raise e
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_training() 