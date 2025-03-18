import os
import torch

class Config:
    # 数据集配置
    DATA_ROOT = 'data'
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    TEST_DIR = os.path.join(DATA_ROOT, 'test')
    
    # 数据预处理配置
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # 训练配置
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    
    # 模型配置
    MESSAGE_LENGTH = 100  # 隐藏消息的长度
    CHANNELS = 3  # RGB图像通道数
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 保存配置
    SAVE_DIR = 'checkpoints'
    SAVE_INTERVAL = 5  # 每隔多少个epoch保存一次模型
    
    # 日志配置
    LOG_DIR = 'logs'
    LOG_INTERVAL = 100  # 每隔多少个batch记录一次日志
    
    # Docker配置
    DOCKER_DATA_DIR = '/app/data'  # Docker容器内的数据目录
    DOCKER_MODEL_DIR = '/app/models'  # Docker容器内的模型目录
    DOCKER_CHECKPOINT_DIR = '/app/checkpoints'  # Docker容器内的检查点目录
    
    # API配置
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    API_DEBUG = False
    
    @classmethod
    def get_docker_path(cls, local_path):
        """将本地路径转换为Docker容器内的路径"""
        path_mapping = {
            cls.DATA_ROOT: cls.DOCKER_DATA_DIR,
            'models': cls.DOCKER_MODEL_DIR,
            'checkpoints': cls.DOCKER_CHECKPOINT_DIR
        }
        for local, docker in path_mapping.items():
            if local_path.startswith(local):
                return local_path.replace(local, docker)
        return local_path 