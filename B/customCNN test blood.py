import torch
import torchvision.transforms as transforms
from medmnist import BloodMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model
model_save_path = "E:\ML_CW\ziyichen20-AMLS_24_25_SN21083037\B\enhanced_blood_mnist.pth"

# Data Preprocessing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Dataset
test_dataset = BloodMNIST(split='test', transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the Enhanced CNN Model
class EnhancedBloodCellCNN(nn.Module):
    def __init__(self):
        super(EnhancedBloodCellCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load Model
model = EnhancedBloodCellCNN().to(device)
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    print(f"Model loaded from {model_save_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

# Testing with Loss Tracking
test_loss = 0.0
test_losses = []
model.eval()
all_preds = []
all_labels = []

criterion = nn.CrossEntropyLoss()  # Loss function for testing

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.squeeze().to(device).long()
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_losses = [test_loss] * len(train_losses)  # Make it match the number of epochs for plotting

# Plot Training, Validation, and Test Loss
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

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

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(8)], yticklabels=[f"Class {i}" for i in range(8)])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Sample Visualization
classes = [f"Class {i}" for i in range(8)]
plt.figure(figsize=(8, 8))
for i in range(9):
    image, label = test_dataset[i]
    image = image * 0.5 + 0.5  # De-normalize for visualization
    plt.subplot(3, 3, i + 1)
    plt.imshow(image.permute(1, 2, 0))  # Convert from CHW to HWC format
    plt.title(classes[int(label)])
    plt.axis('off')
plt.tight_layout()
plt.show()
