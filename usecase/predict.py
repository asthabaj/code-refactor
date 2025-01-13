import torch
from model.neunet import neunet
from utils.helpers import load_image

def predict_image(model, image_path, device):
    cifar10_classes = [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck"
    ]
    model.eval()
    image = load_image(image_path, device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_index = predicted.item()
        class_name = cifar10_classes[class_index]
        print(f"The image '{image_path}' is classified as: {class_name}")