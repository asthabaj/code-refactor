import torch
from dataset.cifar10 import load_cifar10_data
from trainin.train import train_model,load_checkpoint
from model.neunet import neunet  
from evaluation.eval import evaluate_model
from  usecase.predict import predict_image
import os

def load_model_from_checkpoint(checkpoint_dir, device):
    model = neunet(num_classes=10).to(device)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded from checkpoint.")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return model

def main():
    data_dir = './data' 
    checkpoint_dir = './checkpoints' 

    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_loader, test_loader = load_cifar10_data(data_dir)
    # model = train_model(data_dir, device, num_epochs=50, checkpoint_dir=checkpoint_dir)
    model = load_model_from_checkpoint(checkpoint_dir, device)
    _, test_loader = load_cifar10_data(data_dir)
    evaluate_model(model, test_loader, device, data_dir)

    #train_one_image('path/to/your/image.jpg', 0, device) 
    print("Enter the path to the image you want to classify:")
    image_path = input("Image path: ").strip()
    predict_image(model, image_path, device)
    
if __name__ == "__main__":
    main()