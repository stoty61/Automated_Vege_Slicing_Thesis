import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.ops as ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import numpy as np


#Dataset Class

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = []
        
        with open(annotations_file, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["bbox"] is not None:
                    x_min, y_min, x_max, y_max = item["bbox"]
                    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

                    width = x_max - x_min
                    height = y_max - y_min
                    if width > 0 and height > 0:
                        bbox = [x_min, y_min, x_max, y_max]
                        print(f"✅ Valid bbox: {bbox}")
                        self.annotations.append({
                            "file_name": item["file_name"],
                            "bbox": bbox,
                            "pred_class_idx": item["pred_class_idx"]
                        })
                    else:
                        print(f"Skipping invalid bbox: {item['bbox']} for {item['file_name']}")
        print("DONE ADDING BOXES!")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        image = Image.open(img_path).convert("RGB")

        target = {
            "boxes": torch.tensor([item["bbox"]], dtype=torch.float32),
            "labels": torch.tensor([min(num_classes - 1, max(1, item["pred_class_idx"] + 1))], dtype=torch.int64),
        }

        if self.transform:
            image = self.transform(image)

        return image, target

#helper function to compute IoU

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Each box is [x_min, y_min, x_max, y_max].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area_box2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area_box1 + area_box2 - inter_area
    return inter_area / union if union != 0 else 0


#transformations

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#create splits

data_dir = "./vege/data/train"
annotations_file = "bboxes_cam_output.jsonl"
dataset = ObjectDetectionDataset(data_dir, annotations_file, transform=transform)
val_ratio = 0.2
n_val = int(len(dataset) * val_ratio)
n_train = len(dataset) - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


#setup for model

max_class_idx = max(item["pred_class_idx"] for item in dataset.annotations)
num_classes = max_class_idx + 2 
print(f"🔹 Number of classes (including background): {num_classes}")

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda avail:" , torch.cuda.is_available())
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_epochs = 25
epoch_losses = []
epoch_avg_ious = []


# training loop with eval 

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

    # evaluate validation set

    model.eval()
    ious = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred, target in zip(predictions, targets):
                if len(pred["boxes"]) > 0:
                    pred_box = pred["boxes"][0].cpu().numpy()
                    gt_box = target["boxes"][0].numpy()
                    iou = compute_iou(gt_box, pred_box)
                    ious.append(iou)
    avg_iou = np.mean(ious) if ious else 0
    epoch_avg_ious.append(avg_iou)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average IoU on Validation: {avg_iou:.4f}")

    torch.save(model.state_dict(), "faster_rcnn_latest_new.pth")
    print("Model checkpoint saved as 'faster_rcnn_latest.pth'")


# plot

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), epoch_avg_ious, marker='o', color='orange')
plt.title('Average IoU on Validation Set over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average IoU')
plt.grid(True)
plt.savefig('average_iou.png')
plt.show()
