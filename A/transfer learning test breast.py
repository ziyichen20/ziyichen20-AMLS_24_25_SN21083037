import torch
import torchvision.transforms as transforms
from medmnist import BreastMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model
model_save_path = "E:/ML_CW/ziyichen20-AMLS_24_25_SN21083037/A/breast_mnist_resnet.pth"

# Data Preprocessing (same as training)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Dataset
test_dataset = BreastMNIST(split='test', transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the ResNet-based Model (same structure as in training)
class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)  # No need to load pretrained weights for testing
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Single channel
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # Binary classification

    def forward(self, x):
        return self.resnet(x)

# Load the model
model = ResNetBinaryClassifier().to(device)
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_losses = checkpoint.get("train_losses", [])  # Load training losses if available
    val_losses = checkpoint.get("val_losses", [])  # Load validation losses if available
    print(f"Model loaded from {model_save_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

# Plot Learning Curve
if train_losses and val_losses:
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
else:
    print("No training or validation losses found in the checkpoint.")

# Testing loop
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.squeeze().to(device).long()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Evaluation Metrics
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='binary')
recall = recall_score(all_targets, all_preds, average='binary')
f1 = f1_score(all_targets, all_preds, average='binary')

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")

# Confusion Matrix
class_labels = ['Class 0', 'Class 1']  # Replace with meaningful class names if available
conf_matrix = confusion_matrix(all_targets, all_preds)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
