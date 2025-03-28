import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import os
import numpy as np


#dataset class

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
                        self.annotations.append({
                            "file_name": item["file_name"],
                            "bbox": bbox,
                            "pred_class_idx": item["pred_class_idx"]
                        })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        image = Image.open(img_path).convert("RGB")
        target = {
            "boxes": torch.tensor([item["bbox"]], dtype=torch.float32),
            "labels": torch.tensor([item["pred_class_idx"] + 1], dtype=torch.int64),
        }
        if self.transform:
            image = self.transform(image)
        return image, target


#calc distilation loss

def distillation_loss(student_outputs, teacher_outputs, alpha=0.5, temperature=2.0, topk=10):
    total_loss = 0.0
    count = 0
    for s_out, t_out in zip(student_outputs, teacher_outputs):
        if len(t_out["boxes"]) == 0 or len(s_out["boxes"]) == 0:
            continue

        s_scores, s_boxes = s_out["scores"], s_out["boxes"]
        t_scores, t_boxes = t_out["scores"], t_out["boxes"]

        s_topk = min(topk, s_scores.shape[0])
        t_topk = min(topk, t_scores.shape[0])
        k = min(s_topk, t_topk)

        s_probs = torch.nn.functional.log_softmax(s_scores[:k] / temperature, dim=0)
        t_probs = torch.nn.functional.softmax(t_scores[:k] / temperature, dim=0)

        ce_loss = torch.nn.functional.kl_div(s_probs, t_probs, reduction='batchmean') * (temperature ** 2)
        reg_loss = torch.nn.functional.mse_loss(s_boxes[:k], t_boxes[:k])
        loss = alpha * ce_loss + (1 - alpha) * reg_loss
        total_loss += loss
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0, requires_grad=True)


#IoU Calculation

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


data_dir = "./vege/data/train"
annotations_file = "bboxes_cam_output.jsonl"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ObjectDetectionDataset(data_dir, annotations_file, transform)

val_ratio = 0.2
n_val = int(len(dataset) * val_ratio)
n_train = len(dataset) - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


#load teacher

num_classes = max([a["pred_class_idx"] for a in dataset.annotations]) + 2
teacher_model = fasterrcnn_resnet50_fpn(pretrained=False)
teacher_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    teacher_model.roi_heads.box_predictor.cls_score.in_features, num_classes)
teacher_model.load_state_dict(torch.load("faster_rcnn_latest_new.pth", map_location="cpu"))
teacher_model.eval()


#initialize student

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
student_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    student_model.roi_heads.box_predictor.cls_score.in_features, num_classes)
student_model = student_model.to(device)
teacher_model = teacher_model.to(device)

optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
num_epochs = 20


#train w/ distill
for epoch in range(num_epochs):
    student_model.eval()
    total_loss = 0.0
    total_iou = 0.0
    iou_count = 0

    for i, (images, _) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        student_outputs = student_model(images)
        loss = distillation_loss(student_outputs, teacher_outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #iou calc
        for s_out, t_out in zip(student_outputs, teacher_outputs):
            s_boxes = s_out["boxes"].detach().cpu()
            t_boxes = t_out["boxes"].detach().cpu()
            if len(s_boxes) == 0 or len(t_boxes) == 0:
                continue

            k = min(len(s_boxes), len(t_boxes))
            for j in range(k):
                iou = calculate_iou(s_boxes[j], t_boxes[j])
                total_iou += iou
                iou_count += 1

    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Distillation Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")
    torch.save(student_model.state_dict(), f"student_fasterrcnn_epoch{epoch+1}.pth")
    print(f"Saved student model for epoch {epoch+1}")
