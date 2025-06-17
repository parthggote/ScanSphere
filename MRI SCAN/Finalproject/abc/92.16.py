import os
import zipfile
import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kaggle.api.kaggle_api_extended import KaggleApi
import torch.multiprocessing as mp
from typing import Tuple, Optional

# Configuration
CONFIG = {
    "dataset_url": "navoneel/brain-mri-images-for-brain-tumor-detection",
    "dataset_zip": "brain-mri-images-for-brain-tumor-detection.zip",
    "data_dir": "brain_tumor_dataset",
    "image_size": (224, 224),
    "batch_size": 8,
    "model_path": "brain_mri_model.onnx",
    "class_names": ["no", "yes"],
    "learning_rate": 0.0003,
    "epochs": 35,
    "weight_decay": 0.0003,
    "num_workers": 0,
}

def download_and_extract_dataset():
    try:
        if not os.path.exists(CONFIG["dataset_zip"]):
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(CONFIG["dataset_url"], path=".", unzip=False)
        
        with zipfile.ZipFile(CONFIG["dataset_zip"], "r") as zip_ref:
            zip_ref.extractall(CONFIG["data_dir"])
        os.remove(CONFIG["dataset_zip"])
    except Exception as e:
        print(f"Error downloading/extracting dataset: {e}")
        raise

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # Filter out invalid images during initialization
        self.valid_pairs = []
        for img_path, label in zip(image_paths, labels):
            try:
                with Image.open(img_path) as img:
                    # Verify the image can be opened and converted to RGB
                    img.convert("RGB")
                self.valid_pairs.append((img_path, label))
            except Exception as e:
                print(f"Skipping invalid image {img_path}: {e}")
        
        self.transform = transform
        if not self.valid_pairs:
            raise ValueError("No valid images found in the dataset")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.valid_pairs[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a zero tensor with the correct shape instead of None
            if self.transform:
                dummy = torch.zeros((3, *CONFIG["image_size"]))
            else:
                dummy = torch.zeros((3, 224, 224))
            return dummy, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def load_data():
    data_dir = CONFIG["data_dir"]
    categories = CONFIG["class_names"]
    
    images, labels = [], []
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        for file_name in os.listdir(category_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, file_name)
                images.append(img_path)
                labels.append(idx)
    
    if not images:
        raise ValueError("No images found in the dataset")
    
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_transform, test_transform = get_transforms()
    
    train_dataset = BrainMRIDataset(train_imgs, train_labels, transform=train_transform)
    test_dataset = BrainMRIDataset(test_imgs, test_labels, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        drop_last=False
    )
    
    return train_loader, test_loader

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_model(train_loader, model, criterion, optimizer, scheduler, device, num_epochs=None):
    if num_epochs is None:
        num_epochs = CONFIG["epochs"]
        
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
        
        epoch_acc = 100. * correct / total
        scheduler.step()
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {running_loss/len(train_loader):.4f}')
        print(f'Accuracy: {epoch_acc:.2f}%\n')
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({'model_state_dict': model.state_dict()}, 'best_model.pth')

def evaluate_model(test_loader, model, device):
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
    
    print("\nModel Evaluation:")
    print(classification_report(ground_truth, predictions, target_names=CONFIG["class_names"]))
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")
    return accuracy

def convert_to_onnx(model, onnx_path, input_size=(3, 224, 224), device='cpu'):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        onnx_path (str): Path to save the ONNX model.
        input_size (tuple): Input size for the model (default: (3, 224, 224)).
        device (str): Device to use for exporting the model (default: 'cpu').

    Returns:
        None
    """
    model = model.to(device)
    model.eval()

    # Dummy input for the model
    dummy_input = torch.randn(1, *input_size, device=device)

    try:
        print(f"Exporting model to ONNX format at {onnx_path}...")
        torch.onnx.export(
            model,  # Model to export
            dummy_input,  # Dummy input
            onnx_path,  # File path to save the ONNX model
            export_params=True,  # Store the trained parameters in the model
            opset_version=11,  # ONNX version to export to
            do_constant_folding=True,  # Fold constant nodes for optimization
            input_names=['input'],  # Input layer names
            output_names=['output'],  # Output layer names
            dynamic_axes={
                'input': {0: 'batch_size'},  # Variable batch size for input
                'output': {0: 'batch_size'}  # Variable batch size for output
            }
        )
        print("Model successfully exported to ONNX format.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        raise

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download dataset if needed
    if not os.path.exists(CONFIG["data_dir"]):
        download_and_extract_dataset()

    try:
        # Load data
        train_loader, test_loader = load_data()

        # Initialize model
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(CONFIG["class_names"]))
        model = model.to(device)

        # Initialize training components
        criterion = FocalLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["epochs"]
        )

        # Train model
        print("\nStarting training...")
        train_model(train_loader, model, criterion, optimizer, scheduler, device)

        # Load best model and evaluate
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\nEvaluating best model...")
        final_accuracy = evaluate_model(test_loader, model, device)

        print(f"\nTraining completed! Best accuracy: {final_accuracy:.4f}")

        # Convert to ONNX format
        convert_to_onnx(model, CONFIG["model_path"], input_size=(3, *CONFIG["image_size"]), device=device)

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
