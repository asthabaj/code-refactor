import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.neunet import neunet
import torch.nn as nn  
from dataset.cifar10 import load_cifar10_data
from utils.helpers import load_image
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(data_dir, device, num_epochs=2, checkpoint_dir='./checkpoints'): 
    train_loader, test_loader = load_cifar10_data(data_dir)
    model = neunet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter("runs/cifar10loss")

    os.makedirs(checkpoint_dir, exist_ok=True) 

    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        print(f'Training epoch {epoch}..')
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        writer.add_scalar('Loss/Test', avg_test_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir)

    return model

# def train_one_image(image_path, label, device, num_epochs=100):
#     model = neunet(num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     image = load_image(image_path, device)
#     label = torch.tensor([label]).to(device)
#     oneimgdataset = TensorDataset(image, label)
#     oneimgloader = DataLoader(oneimgdataset, batch_size=1)

#     for epoch in range(num_epochs):
#         model.train()
#         for inputs, labels in oneimgloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#     return model

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer_checkpoint.pth')
    epoch_path = os.path.join(checkpoint_dir, 'epoch_checkpoint.pth')

    torch.save(model.state_dict(), checkpoint_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save({'epoch': epoch, 'loss': loss}, epoch_path)

def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer_checkpoint.pth')
    epoch_path = os.path.join(checkpoint_dir, 'epoch_checkpoint.pth')

    # if os.path.isfile(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     optimizer.load_state_dict(torch.load(optimizer_path))
    #     checkpoint = torch.load(epoch_path)
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Resuming training from epoch {start_epoch}...")
    #     return start_epoch
    # else:
    #     print("No checkpoint found. Starting from scratch.")
    #     return 0
    return model