import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),  # 增加一层卷积
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 将特征图大小调整为1x1
        
    def forward(self, x):
        x = self.main(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), 1)  # 调整维度为(batch_size, 1) 