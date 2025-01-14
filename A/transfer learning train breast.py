import torch
import torchvision.transforms as transforms
from medmnist import BreastMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model save path
save_dir = "E:\\ML_CW\\A"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
model_save_path = os.path.join(save_dir, "breast_mnist_resnet.pth")

# Data Preprocessing (Same as custom CNN)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Dataset
train_dataset = BreastMNIST(split='train', transform=train_transform, download=True)
val_dataset = BreastMNIST(split='val', transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the ResNet-based Model
class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Load pretrained ResNet50
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single channel
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # Adjust for binary classification

    def forward(self, x):
        return self.resnet(x)

# Initialize model, loss function, and optimizer
model = ResNetBinaryClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
train_losses = []
val_losses = []

print("Training started...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.squeeze().to(device).long()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.squeeze().to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the model and losses
torch.save({
    "model_state_dict": model.state_dict(),
    "train_losses": train_losses,
    "val_losses": val_losses
}, model_save_path)

print(f"Model and training metrics saved to {model_save_path}")
