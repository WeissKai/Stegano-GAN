import torch
import torch.nn as nn

def reconstruction_loss(original, generated):
    return nn.MSELoss()(original, generated)

def adversarial_loss(discriminator_output, target):
    return nn.BCELoss()(discriminator_output, target)

def message_extraction_loss(original_message, extracted_message):
    return nn.MSELoss()(original_message, extracted_message)

def total_loss(original_image, generated_image, discriminator_output, target, original_message, extracted_message):
    return reconstruction_loss(original_image, generated_image) + \
           adversarial_loss(discriminator_output, target) + \
           message_extraction_loss(original_message, extracted_message) 