import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def model_load():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()
    return model

def preprocess_input(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = preprocess(image)
    input_batch = input_image.unsqueeze(0)

    return input_batch

def to_cuda(model):
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    return model

def predict(model, input_batch):
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

def get_class(probabilities):
    with open('classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    probabilities_np = probabilities.cpu().detach().numpy()
    max_idx = np.argmax(probabilities_np)

    class_index = classes[max_idx]
    return class_index
