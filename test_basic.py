import os
import torch
from utils.data_loader import get_data_loaders
import matplotlib.pyplot as plt
from config import Config

def test_data_loader():
    """测试数据加载器功能"""
    print("开始测试数据加载器...")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders()
    
    # 测试训练数据加载器
    print("\n测试训练数据加载器...")
    for i, (images, messages) in enumerate(train_loader):
        print(f"批次 {i+1}:")
        print(f"图像形状: {images.shape}")
        print(f"消息形状: {messages.shape}")
        
        # 显示第一张图像
        if i == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(images[0].permute(1, 2, 0).numpy())
            plt.title("训练图像")
            
            plt.subplot(1, 2, 2)
            plt.plot(messages[0].numpy())
            plt.title("随机消息")
            plt.savefig('test_data_loader.png')
            plt.close()
        
        if i >= 2:  # 只测试前3个批次
            break
    
    # 测试测试数据加载器
    print("\n测试测试数据加载器...")
    for i, (images, messages) in enumerate(test_loader):
        print(f"批次 {i+1}:")
        print(f"图像形状: {images.shape}")
        print(f"消息形状: {messages.shape}")
        
        if i >= 2:  # 只测试前3个批次
            break

def test_docker():
    """测试Docker环境"""
    print("\n开始测试Docker环境...")
    
    # 检查Docker是否安装
    try:
        import docker
        client = docker.from_env()
        print("Docker已安装")
        
        # 检查Docker服务是否运行
        client.ping()
        print("Docker服务正在运行")
        
        # 检查可用镜像
        images = client.images.list()
        print("\n可用的Docker镜像:")
        for image in images:
            print(f"- {image.tags}")
            
    except Exception as e:
        print(f"Docker测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("=== SteganoGAN基础功能测试 ===")
    
    # 测试数据加载器
    test_data_loader()
    
    # 测试Docker环境
    test_docker()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 