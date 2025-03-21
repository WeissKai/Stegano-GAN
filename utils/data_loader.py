import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from config import Config
import torchvision
import torchvision.transforms as transforms

class SteganoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        Args:
            root_dir: 数据集根目录
            transform: 数据转换操作
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
            
        print(f"Loaded {len(self.image_files)} images from {root_dir}")
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取一个数据样本
        Args:
            idx: 样本索引
        Returns:
            image: 图像张量
            message: 随机生成的消息
        """
        # 加载图像
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        # 生成随机消息，维度与图像相匹配
        message = torch.randint(0, 2, (Config.MESSAGE_LENGTH, 1, 1), dtype=torch.float32)
        message = message.expand(-1, image.size(1), image.size(2))  # 扩展到与图像相同的空间维度
        
        return image, message

def get_data_loaders(batch_size=Config.BATCH_SIZE):
    """
    获取训练和测试数据加载器
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集
    train_dataset = SteganoDataset(Config.TRAIN_DIR, train_transform)
    test_dataset = SteganoDataset(Config.TEST_DIR, test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    return train_loader, test_loader

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载和加载训练数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# 下载和加载测试数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)