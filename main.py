import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义路径和文件名
file_path_dis = "discriminator.pth"  # 替换为你的路径和文件名
file_path_gen = "generator.pth"  # 替换为你的路径和文件名


class Generator(nn.Module):
    def __init__(self, input_dim, output_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_channels),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

if os.path.exists(file_path_gen) and os.path.exists(file_path_dis):
    # 初始化生成器和判别器
    input_dim = 100  # 随机噪声的维度
    output_channels = 28 * 28  # MNIST图像的大小
    generator = Generator(input_dim, output_channels)
    discriminator = Discriminator(output_channels)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载MNIST数据集
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_epochs = 2
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            # 真实图像
            real_images = images.view(images.size(0), -1)
            real_labels = torch.ones(images.size(0), 1)
            
            # 生成隐写图像
            noise = torch.randn(images.size(0), input_dim)
            fake_images = generator(noise)
            fake_labels = torch.zeros(images.size(0), 1)
            
            # 训练判别器
            optimizer_D.zero_grad()
            real_outputs = discriminator(real_images)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # 训练生成器
            optimizer_G.zero_grad()
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')



# 生成隐写图像
noise = torch.randn(64, input_dim)
# 生成图像
with torch.no_grad():
    fake_image = generator(noise).view(-1, 28, 28).detach().numpy()

# 显示并保存图像
plt.imshow(fake_image[0], cmap='gray')
plt.axis('off')
plt.savefig('generated_image.png', bbox_inches='tight', pad_inches=0)  # 保存为 PNG 文件
plt.show()

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
