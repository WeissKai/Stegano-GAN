import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar10_images(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch in range(1, 6):
        batch_name = f'data_batch_{batch}'
        batch_path = os.path.join(data_dir, batch_name)
        batch_data = unpickle(batch_path)
        
        for i, img_data in enumerate(batch_data[b'data']):
            img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img)
            img.save(os.path.join(output_dir, f'{batch_name}_{i}.png'))

    # Process test batch
    test_batch = 'test_batch'
    test_path = os.path.join(data_dir, test_batch)
    test_data = unpickle(test_path)
    
    for i, img_data in enumerate(test_data[b'data']):
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'{test_batch}_{i}.png'))

# 使用示例
data_dir = './data/cifar-10-batches-py'  # CIFAR-10 数据集解压后的目录
output_dir = './data/images'  # 输出图像的目录
save_cifar10_images(data_dir, output_dir) 