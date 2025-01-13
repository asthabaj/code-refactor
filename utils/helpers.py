import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

def imshow(img):
    img = img / 2 + 0.5  
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

def load_image(image_path, device):
    new_transform = transforms.Compose([
      transforms.Resize((32,32)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0).to(device)
    return image