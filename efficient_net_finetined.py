import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os
import json
import matplotlib.pyplot as plt


#dataset class

class CustomDataset(Dataset):
    def __init__(self, data_dir, metadata_file, transform):
        self.data_dir = data_dir
        self.transform = transform

        self.metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                item = json.loads(line)
                self.metadata.append({"file_name": item["file_name"], "label": item["class"]})

        #label mapping
        self.class_to_idx = {label: idx for idx, label in enumerate(
            sorted({item["label"] for item in self.metadata})
        )}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        label = self.class_to_idx[item["label"]]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label



transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#load data

train_data_dir = "./vege/data/train"
train_metadata_file = "./vege/data/train/metadata.jsonl"
test_data_dir = "./vege/data/test"
test_metadata_file = "./vege/data/test/metadata.jsonl"

train_dataset = CustomDataset(train_data_dir, train_metadata_file, transform)
test_dataset = CustomDataset(test_data_dir, test_metadata_file, transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#define model

def load_pretrained_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    #freeze early layers
    for param in model.features.parameters():
        param.requires_grad = False

    # fit num of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

num_classes = len(train_dataset.class_to_idx)
model = load_pretrained_model(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


#train

num_epochs = 500
patience = 100
best_test_accuracy = 0
epochs_no_improve = 0

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_train_loss, correct_train, total_train = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_train_loss / len(train_loader)
    train_acc = 100.0 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    running_test_loss, correct_test, total_test = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_loss = running_test_loss / len(test_loader)
    test_acc = 100.0 * correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # check for patience
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc
        torch.save(model.state_dict(), "efficientnet_b0_finetuned.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            break


# plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
plt.title('Training vs. Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
plt.plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
plt.title('Training vs. Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()