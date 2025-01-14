import torch
import torchvision.transforms as transforms
from torchvision import models
from medmnist import BloodMNIST
from torch.utils.data import DataLoader



import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model
model_save_path = "E:/ML_CW/ziyichen20-AMLS_24_25_SN21083037/B/resnet18_bloodmnist.pth"

# Data Preprocessing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# Load Dataset
test_dataset = BloodMNIST(split='test', transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)  # 8 classes
model = model.to(device)

# Load the saved model
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    print(f"Model loaded from {model_save_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

# Plot Learning Curves
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Testing
print("Testing started...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.squeeze().to(device).long()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Evaluation Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Classification Report
print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(8)]))

# Sample Visualization
classes = [f"Class {i}" for i in range(8)]
plt.figure(figsize=(8, 8))
for i in range(9):
    image, label = test_dataset[i]
    image = image * 0.5 + 0.5  # De-normalize for visualization
    label = int(label)
    plt.subplot(3, 3, i + 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(classes[label])
    plt.axis('off')
plt.tight_layout()
plt.show()
