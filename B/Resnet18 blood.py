import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from medmnist import BloodMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save the model
save_dir = "E:\\ML_CW\\B"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
model_save_path = os.path.join(save_dir, "resnet18_bloodmnist.pth")

# Data Preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# Load Dataset
train_dataset = BloodMNIST(split='train', transform=train_transform, download=True)
val_dataset = BloodMNIST(split='val', transform=val_transform, download=True)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Modify the final layer for 8 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)  # 8 classes
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 10
train_losses, val_losses = [], []

print("Training started...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
        inputs, labels = inputs.to(device), labels.squeeze().to(device).long()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:  # Log every 10 batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.squeeze().to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save({
    "model_state_dict": model.state_dict(),
    "train_losses": train_losses,
    "val_losses": val_losses
}, model_save_path)
print(f"Model and training metrics saved to {model_save_path}")
