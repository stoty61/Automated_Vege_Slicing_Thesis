import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# define dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, metadata_file, transform):
        self.data_dir = data_dir
        self.transform = transform

        #load metadata
        self.metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                self.metadata.append({
                    "file_name": item["file_name"],
                    "label": item["class"]
                })

        # map class labels to indexs
        unique_labels = sorted({item["label"] for item in self.metadata})
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        print(self.idx_to_class)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.data_dir, item["file_name"])
        label = self.class_to_idx[item["label"]]

        image = Image.open(img_path).convert("RGB")
        
        #trasnform image
        image = self.transform(image)

        return image, label, item["file_name"]



# load model
def load_model(weight_path, num_classes, device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #load weights
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model



# bbox extraction 

def get_gradcam_bbox(
    model, 
    input_tensor, 
    target_class_idx, 
    target_layer, 
    orig_width, 
    orig_height,
    threshold=0.5
):


    # create gradcam
    cam = GradCAM(model=model, target_layers=[target_layer])

    #list of targets
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    # resize cam
    cam_resized = cv2.resize(grayscale_cam, (orig_width, orig_height))
    bin_cam = cam_resized >= threshold

    ys, xs = np.where(bin_cam == True)
    if len(xs) == 0 or len(ys) == 0:
        return None  # return none if no activation above thresh

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return (x_min, y_min, x_max, y_max)


def main():
    data_dir = "./vege/data/train"
    metadata_file = "./vege/data/train/metadata.jsonl"
    model_weight_path = "restnet_50_finetuned.pth"
    output_jsonl = "bboxes_cam_output.jsonl" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # transforms for data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(data_dir, metadata_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    num_classes = len(dataset.class_to_idx)
    model = load_model(model_weight_path, num_classes, device)

    target_layer = model.layer4[-1]

    # cam bounding box
    results = []
    for batch_idx, (images, labels, file_names) in enumerate(data_loader):
        image_tensor = images.to(device)
        label_idx = labels.item()
        file_name = file_names[0]

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_idx = outputs.argmax(dim=1).item()
    
        target_class = pred_idx

        original_image_path = os.path.join(data_dir, file_name)
        original_pil = Image.open(original_image_path).convert("RGB")
        orig_width, orig_height = original_pil.size
        
        # generate box
        bbox = get_gradcam_bbox(
            model=model,
            input_tensor=image_tensor,
            target_class_idx=target_class,
            target_layer=target_layer,
            orig_width=orig_width,
            orig_height=orig_height,
            threshold=0.5       
        )


        if bbox is None:
            #nothing significant 
            result = {
                "file_name": file_name,
                "pred_class_idx": target_class,
                "bbox": None
            }
            print(f"[{batch_idx}] {file_name}: No activation found.")
        else:
            x_min, y_min, x_max, y_max = bbox
            result = {
                "file_name": file_name,
                "pred_class_idx": target_class,
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
            }
            print(f"[{batch_idx}] {file_name}: BBox -> {bbox}")

        results.append(result)

    #write to file
    with open(output_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Bounding boxes saved to {output_jsonl}")


if __name__ == "__main__":
    main()
