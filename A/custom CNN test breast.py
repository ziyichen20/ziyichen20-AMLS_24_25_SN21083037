import torch
import torchvision.transforms as transforms
from medmnist import BreastMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model
model_save_path = "E:\\ML_CW\\A\\breast_mnist_cnn.pth"

# Data Preprocessing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Dataset
val_dataset = BreastMNIST(split='val', transform=test_transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = BreastMNIST(split='test', transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SimpleCNN().to(device)
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint.get("val_losses", [])
    print(f"Model loaded from {model_save_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
if val_losses:
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Validation loop
print("Calculating Validation Loss...")
model.eval()
val_loss = 0.0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.squeeze().to(device).long()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

val_loss /= len(val_loader)
print(f"Validation Loss: {val_loss:.4f}")

# Testing loop
print("Testing the model...")
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.squeeze().to(device).long()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Evaluation metrics
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='binary')
recall = recall_score(all_targets, all_preds, average='binary')
f1 = f1_score(all_targets, all_preds, average='binary')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
