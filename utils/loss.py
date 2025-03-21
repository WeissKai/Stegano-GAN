import torch
import torch.nn as nn

def reconstruction_loss(original, generated):
    return nn.MSELoss()(original, generated)

def adversarial_loss(discriminator_output, target):
    return nn.BCELoss()(discriminator_output, target)

def message_extraction_loss(original_message, extracted_message):
    return nn.MSELoss()(original_message, extracted_message)

def total_loss(original_image, generated_image, discriminator_output, target, original_message, extracted_message):
    # 调整损失权重
    return 0.4 * reconstruction_loss(original_image, generated_image) + \
           0.3 * adversarial_loss(discriminator_output, target) + \
           0.3 * message_extraction_loss(original_message, extracted_message) 