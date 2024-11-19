import torch
import torch.nn as nn # neural network
import torch.optim as optim # optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split # since kaggle dataset does not come with validation folder, will create one using train images
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm # will fine-tune pre trained image classifier model (EfficientNet-B0)

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm # for progress bar during training loop 

# PYTORCH DATASET PREPROCESSING
class ChestXRays(Dataset):
    def __init__(self, data_directory, transform = None):
        self.data = ImageFolder(data_directory, transform = transform) # will handle creating labels (normal (0) & pneumonia (1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

data_directory = './chest_xray_raw/train'
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor() # converts PIL Image into a Pytorch Tensor. Scales pixel values from [0, 255] to [0.0, 1.0] for easier processing during model training
])
dataset = ChestXRays(data_directory = data_directory, 
                     transform = transform) # creates tuple for each (image, label). indeces 0-1348 are normal (class = 0), 1349-3882 are pneumonia (class = 1)

# print(len(dataset)) -> 5232
# print(dataset[1348]) -> (<PIL.Image.Image image mode=RGB size=128x128 at 0x16BF3F750>, 0)
# print(dataset[1349]) -> (<PIL.Image.Image image mode=RGB size=128x128 at 0x16BF3F750>, 1)

image, label = dataset[0]
# print(image.shape) -> torch.Size([3, 128, 128])
# print(label) -> 0 

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True) # batching divides images into groups (model trains faster in batches, instead of one image at a time)

# for images, labels in dataloader:
#     print(images.shape) -> torch.Size([32, 3, 128, 128])
#     print(labels.shape) -> torch.Size([32])
#     print(labels) -> tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
#         1, 1, 1, 1, 1, 1, 1, 1])
#     break

# PYTORCH MODEL
class SimplePneumoniaClassifer(nn.Module):
    def __init__(self, num_classes = 2): # defining parts of the model
        super(SimplePneumoniaClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True) # B0 indicates size of model (on the smaller side) 
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # removing last layer which has 1000 possible outputs (ImageNet has 1000 categories)
        
        enet_out_size = 1280 # number of features (dimensions) the efficientnet_b0 can classify on 
        self.classifier = nn.Linear(enet_out_size, num_classes) # replacing last layer with 2 possible classes (normal or pneumonia)

    def forward(self, x): # feeding data to go through the neural network
        x = self.features(x) # passing data through feature extractor 
        output = self.classifier(x) # passing output through classifier
        return output

model = SimplePneumoniaClassifer(num_classes = 2)

# for images, labels in dataloader: -> processes the images and outputs predictions
#     print(model(images).shape) -> torch.Size([32, 2]) [batch_size, num_classes]
#     break


# PYTORCH TRAINING LOOP
criterion = nn.CrossEntropyLoss() # loss function for multi-class classification. defines how well model is performing by calculating loss
optimizer = optim.Adam(model.parameters(), lr = 0.001) # helps model learn by adjusting model's parameters (including weights). minimizes loss

# for images, labels in dataloader:
#     print(criterion(model(images), labels)) -> tensor(0.8155, grad_fn=<NllLossBackward0>) (0.8155 is loss)
#     break

# setting up datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])
train_folder = './chest_xray_raw/train'
test_folder = './chest_xray_raw/test'

train_dataset = ChestXRays(train_folder, transform = transform)
test_dataset = ChestXRays(test_folder, transform = transform)

train_size = int(0.9 * len(train_dataset))
validation_size = len(train_dataset) - train_size

train_subset, validation_subset = random_split(train_dataset, [train_size, validation_size])

train_loader = DataLoader(train_subset, batch_size = 32, shuffle = True)
validation_loader = DataLoader(validation_subset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# loop time
num_epochs = 5 # runs through entire training data set
train_losses, validation_losses = [], []

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # install latest version of pytorch to util torch mps (leverages apple's metal framework for silicon devices like my m1 macbook pro)
print(device)

model = SimplePneumoniaClassifer(num_classes = 2)
model.to(device)

for epoch in range(num_epochs):
    model.train() # training, fine tuning weights
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc = 'Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # back propogation. updates model weights
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad(): # not changing model weights
        for images, labels in tqdm(validation_loader, desc = 'Validation loop'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    
    validation_loss = running_loss / len(validation_loader.dataset)
    validation_losses.append(validation_loss)

    # Print epoch stats
    print(f"Epoch {epoch + 1} / {num_epochs} - Train:loss {train_loss}, Validation:loss {validation_loss}")

# EVALUATING RESULTS

# training loss vs validation loss 
plt.plot(train_losses, label = 'Training loss')
plt.plot(validation_losses, label = 'Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show() 

# performance evalutation on test dataset
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

# Calculating and printing performance metrics
# Convert to numpy for easier processing
true_labels = np.array(true_labels)
predictions = np.array(predictions)

cm = confusion_matrix(true_labels, predictions)

# Extract values from confusion matrix
TN, FP, FN, TP = cm.ravel()  # True Negative, False Positive, False Negative, True Positive

accuracy = (TP + TN) / (TP + TN + FP + FN)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"False Positive: {FP}")
print(f"False Negative: {FN}")
print(f"False Positive Rate: {FPR:.2f}")
print(f"False Negative Rate: {FNR:.2f}")

plt.figure(figsize=(6, 4))
plt.scatter(FPR, FNR, color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('FPR vs FNR')
plt.grid(True)
plt.show()