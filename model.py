import torch
import torch.nn as nn  # neural network
import torch.optim as optim  # optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split  # split dataset into train and validation
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm  # pretrained models (EfficientNet-B0)

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm  # for progress bar during training loop

# PYTORCH DATASET PREPROCESSING
class ChestXRays(Dataset):
    def __init__(self, data_directory, transform=None):
        self.data = ImageFolder(data_directory, transform=transform)  # handles labeling (normal (0) & pneumonia (1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

transform = transforms.Compose([
    transforms.Resize((224, 224)), # EfficientNetB0 Image Input SIze
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# PYTORCH MODEL
class SimplePneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplePneumoniaClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) 

        enet_out_size = 1280  # Features extracted by EfficientNet
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # using mps instead of cuda so it can use m1 gpu

model = SimplePneumoniaClassifier(num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))  # Increase weight of pneumonia class
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Setup datasets
train_folder = './chest_xray_raw/train'
test_folder = './chest_xray_raw/test'

train_dataset = ChestXRays(train_folder, transform=transform)
test_dataset = ChestXRays(test_folder, transform=transform)

train_size = int(0.9 * len(train_dataset))
validation_size = len(train_dataset) - train_size

train_subset, validation_subset = random_split(train_dataset, [train_size, validation_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 5
train_losses, validation_losses = [], []

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()  
    running_loss = 0.0
    with torch.no_grad():  
        for images, labels in tqdm(validation_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

    validation_loss = running_loss / len(validation_loader.dataset)
    validation_losses.append(validation_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}")

# Plot training vs validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

# Evaluation on test dataset
from sklearn.metrics import confusion_matrix

model.eval()
true_labels = []
predictions = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing loop'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

cm = confusion_matrix(true_labels, predictions)
TN, FP, FN, TP = cm.ravel()

accuracy = (TP + TN) / (TP + TN + FP + FN)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP) 

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"False Positive: {FP}")
print(f"False Negative: {FN}")
print(f"False Positive Rate: {FPR:.2f}")
print(f"False Negative Rate: {FNR:.2f}")