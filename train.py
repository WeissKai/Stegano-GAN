import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.generator import Generator
from models.discriminator import Discriminator
from models.decoder import Decoder
from utils.data_loader import get_data_loaders
from utils.loss import total_loss
from utils.metrics import compute_psnr, compute_ssim, compute_message_accuracy

def train(args):
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("\n初始化模型...")
    generator = Generator(input_channels=3, output_channels=3).to(device)
    discriminator = Discriminator(input_channels=3).to(device)
    decoder = Decoder(input_channels=3, message_length=args.message_length).to(device)
    
    # 从检查点加载模型（如果存在）
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"正在加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            decoder.load_state_dict(checkpoint['decoder'])
            print(f"加载成功，从epoch {start_epoch} 开始继续训练")
        else:
            print(f"未找到检查点 '{args.resume}'")
    
    # 初始化优化器
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 获取数据加载器
    print("\n加载数据...")
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    
    # 训练循环
    print("\n开始训练...")
    
    # 记录训练日志
    train_losses_g = []
    train_losses_d = []
    val_psnr = []
    val_ssim = []
    val_message_acc = []
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        decoder.train()
        
        # 训练一个epoch
        train_loss_g = 0
        train_loss_d = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (images, messages) in enumerate(progress_bar):
            images, messages = images.to(device), messages.to(device)
            batch_size = images.size(0)
            
            # ----- 训练判别器 -----
            # 生成隐藏图像
            hidden_images = generator(images)
            
            # 判别器前向传播
            real_output = discriminator(images)
            fake_output = discriminator(hidden_images.detach())
            
            # 计算判别器损失
            d_loss = nn.BCELoss()(real_output, torch.ones(batch_size, 1).to(device)) + \
                     nn.BCELoss()(fake_output, torch.zeros(batch_size, 1).to(device))
            
            # 判别器反向传播
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            
            # ----- 训练生成器和解码器 -----
            # 重新计算判别器输出（因为参数已更新）
            hidden_images = generator(images)
            fake_output = discriminator(hidden_images)
            extracted_messages = decoder(hidden_images)
            
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
            
            # 更新进度条
            train_loss_g += g_loss.item()
            train_loss_d += d_loss.item()
            progress_bar.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item()
            })
            
            # 每n步保存中间结果
            if (i + 1) % args.sample_interval == 0:
                with torch.no_grad():
                    # 保存一个批次的示例
                    sample_idx = np.random.randint(0, batch_size)
                    sample_image = images[sample_idx].detach().cpu().numpy().transpose(1, 2, 0)
                    sample_hidden = hidden_images[sample_idx].detach().cpu().numpy().transpose(1, 2, 0)
                    
                    # 将数据范围从[-1, 1]转换为[0, 1]
                    sample_image = (sample_image + 1) / 2
                    sample_hidden = (sample_hidden + 1) / 2
                    
                    # 原始消息和提取消息
                    sample_message = messages[sample_idx].detach().cpu().numpy()
                    sample_extracted = extracted_messages[sample_idx].detach().cpu().numpy()
                    
                    # 可视化
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(2, 2, 1)
                    plt.imshow(sample_image)
                    plt.title('原始图像')
                    plt.axis('off')
                    
                    plt.subplot(2, 2, 2)
                    plt.imshow(sample_hidden)
                    plt.title('隐藏图像')
                    plt.axis('off')
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(sample_message)
                    plt.title('原始消息')
                    plt.ylim(-1.1, 1.1)
                    
                    plt.subplot(2, 2, 4)
                    plt.plot(sample_extracted)
                    plt.title('提取消息')
                    plt.ylim(-1.1, 1.1)
                    
                    plt.tight_layout()
                    plt.savefig(f"{args.log_dir}/sample_epoch{epoch+1}_step{i+1}.png")
                    plt.close()
                
        # 计算平均训练损失
        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)
        train_losses_g.append(train_loss_g)
        train_losses_d.append(train_loss_d)
        
        # 验证
        generator.eval()
        decoder.eval()
        
        val_psnr_epoch = []
        val_ssim_epoch = []
        val_message_acc_epoch = []
        
        with torch.no_grad():
            for images, messages in test_loader:
                images, messages = images.to(device), messages.to(device)
                
                # 生成隐藏图像
                hidden_images = generator(images)
                extracted_messages = decoder(hidden_images)
                
                # 计算评估指标
                psnr_value = compute_psnr(images, hidden_images)
                ssim_value = compute_ssim(images, hidden_images)
                message_acc = compute_message_accuracy(messages, extracted_messages)
                
                val_psnr_epoch.append(psnr_value)
                val_ssim_epoch.append(ssim_value)
                val_message_acc_epoch.append(message_acc)
        
        # 计算平均验证指标
        avg_psnr = sum(val_psnr_epoch) / len(val_psnr_epoch)
        avg_ssim = sum(val_ssim_epoch) / len(val_ssim_epoch)
        avg_message_acc = sum(val_message_acc_epoch) / len(val_message_acc_epoch)
        
        val_psnr.append(avg_psnr)
        val_ssim.append(avg_ssim)
        val_message_acc.append(avg_message_acc)
        
        # 打印训练和验证结果
        print(f"Epoch {epoch+1}/{args.epochs}, "
              f"Train G Loss: {train_loss_g:.4f}, "
              f"Train D Loss: {train_loss_d:.4f}, "
              f"Val PSNR: {avg_psnr:.2f} dB, "
              f"Val SSIM: {avg_ssim:.4f}, "
              f"Val Msg Acc: {avg_message_acc:.4f}")
        
        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'optimizer_dec': optimizer_dec.state_dict(),
            }, checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")
    
    # 保存最终模型
    torch.save(generator.state_dict(), f"{args.checkpoint_dir}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{args.checkpoint_dir}/discriminator_final.pth")
    torch.save(decoder.state_dict(), f"{args.checkpoint_dir}/decoder_final.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses_g, label='Generator')
    plt.plot(train_losses_d, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_psnr)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    
    plt.subplot(1, 3, 3)
    plt.plot(val_message_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Message Extraction Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{args.log_dir}/training_curves.png")
    
    print("\n训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SteganoGAN训练')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--message_length', type=int, default=100, help='消息长度')
    parser.add_argument('--sample_interval', type=int, default=100, help='示例保存间隔（批次）')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔（轮数）')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--resume', type=str, default='', help='从检查点恢复训练')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='禁用CUDA')
    args = parser.parse_args()
    
    train(args)