import torch
from dataset import get_dataloaders
from model import WildfireCNN
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    data_dir = "datasets"  # make sure structure is datasets/train, datasets/valid, datasets/test
    batch_size = 32
    num_epochs = 10

    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = WildfireCNN(num_classes=len(class_names)).to(device)

    model = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs)

    evaluate_model(model, test_loader, device, class_names)
