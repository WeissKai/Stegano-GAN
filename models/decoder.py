import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_channels, message_length):
        super(Decoder, self).__init__()
        self.message_length = message_length
        
        # 卷积特征提取
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层，输出消息
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, message_length),
            nn.Tanh()  # 输出范围限制在[-1, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x 