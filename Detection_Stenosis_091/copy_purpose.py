# Cell 1: Import libraries
import os
import cv2
import numpy as np
import pandas as pd
import json
import random
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import tqdm as notebook_tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO

# Cell 2: Define paths - using relative paths
BASE_DIR = "."
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
COCO_DIR = os.path.join(DATASET_DIR, "coco_format")
YOLO_DIR = os.path.join(DATASET_DIR, "yolo_format")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create necessary directories
for directory in [DATASET_DIR, YOLO_DIR, PROCESSED_DIR, RESULTS_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# For YOLO format
for split in ["train", "val", "test"]:
    for subdir in ["images", "labels"]:
        os.makedirs(os.path.join(YOLO_DIR, split, subdir), exist_ok=True)

# For processed data
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(PROCESSED_DIR, split), exist_ok=True)

# Cell 3: Check for GPU availability
import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")  # Use GPU
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")  # Use CPU

# Cell 4: Define function to convert COCO to YOLO format
def convert_all_datasets():
    """
    Convert all datasets from COCO to YOLO format based on correct file paths
    """
    # Update paths to work with the global variables
    base_dir = os.path.join(BASE_DIR, "dataset", "stenosis")
    output_base_dir = YOLO_DIR
    
    print(f"Looking for COCO annotations in: {base_dir}")
    print(f"Output YOLO annotations will be saved to: {output_base_dir}")
    
    for split in ["train", "val", "test"]:
        coco_file = os.path.join(base_dir, split, "annotations", f"{split}.json")
        output_dir = os.path.join(output_base_dir, split, "labels")
        images_dir = os.path.join(base_dir, split, "images")
        
        print(f"Checking for COCO file: {coco_file}")
        
        if os.path.exists(coco_file):
            os.makedirs(output_dir, exist_ok=True)
            
            # Create image output directory
            images_output_dir = os.path.join(output_base_dir, split, "images")
            os.makedirs(images_output_dir, exist_ok=True)
            
            print(f"Converting {split} dataset...")
            # Load COCO annotations
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create dictionary for image lookup
            image_dict = {img['id']: img for img in coco_data['images']}
            
            # Find stenosis category id
            stenosis_category_id = None
            for category in coco_data['categories']:
                if category.get('name', '').lower() == 'stenosis':
                    stenosis_category_id = category['id']
                    break
            
            if stenosis_category_id is None:
                print("Warning: 'stenosis' category not found in the dataset.")
                print("Available categories:", [cat.get('name', cat.get('id')) for cat in coco_data['categories']])
                # Use first category if stenosis not found
                stenosis_category_id = coco_data['categories'][0]['id']
                print(f"Using category ID {stenosis_category_id} as default")
            
            # Process each annotation
            annotation_count = 0
            for ann in coco_data['annotations']:
                # Skip if not stenosis
                if ann['category_id'] != stenosis_category_id:
                    continue
                
                img_info = image_dict[ann['image_id']]
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Get bounding box coordinates
                x, y, width, height = ann['bbox']
                
                # Convert to YOLO format (x_center, y_center, width, height) normalized
                x_center = (x + width / 2) / img_width
                y_center = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # Ensure values are within bounds [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                # Create YOLO format label (class 0 for stenosis)
                yolo_line = f"0 {x_center} {y_center} {norm_width} {norm_height}\n"
                
                # Save to label file
                image_filename = img_info['file_name']
                image_basename = os.path.splitext(image_filename)[0]
                label_path = os.path.join(output_dir, f"{image_basename}.txt")
                
                # Append to label file (or create if it doesn't exist)
                with open(label_path, 'a') as f:
                    f.write(yolo_line)
                
                # Copy image if it doesn't exist in output directory
                src_path = os.path.join(images_dir, image_filename)
                dst_path = os.path.join(images_output_dir, image_filename)
                if not os.path.exists(dst_path) and os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                
                annotation_count += 1
            
            print(f"Conversion complete for {split}. Processed {annotation_count} annotations.")
        else:
            print(f"COCO file {coco_file} not found. Skipping {split} split.")
    
    # Create classes.txt file
    classes_file = os.path.join(output_base_dir, "classes.txt")
    with open(classes_file, 'w') as f:
        f.write("stenosis\n")
    
    print("All datasets converted successfully.")

# Cell 5: Define image preprocessing functions
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast"""
    if len(image.shape) == 3:  # Color image
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split channels
        l, a, b = cv2.split(lab)
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # Apply CLAHE to L-channel
        cl = clahe.apply(l)
        # Merge channels
        merged = cv2.merge((cl, a, b))
        # Convert back to BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced
    else:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def preprocess_image(image_path, output_path, target_size=(640, 640), convert_to_gray=False):
    """
    Preprocess an angiographic image with multiple techniques
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image
        target_size: Target image size as (width, height)
        convert_to_gray: Whether to convert to grayscale
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale if requested
    if convert_to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for consistent processing
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Apply CLAHE for contrast enhancement
    enhanced = apply_clahe(image, clip_limit=3.0)
    
    # Apply noise reduction (Gaussian filter)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Normalize pixel values to range [0, 1]
    normalized = denoised.astype(np.float32) / 255.0
    
    # Convert back to uint8 for saving
    processed = (normalized * 255).astype(np.uint8)
    
    # Resize to target size
    if target_size:
        processed = cv2.resize(processed, target_size)
    
    # Save the processed image
    cv2.imwrite(output_path, processed)
    
    return processed

# Cell 6: Define dataset preprocessing function
def preprocess_dataset(input_dir, output_dir, split="train", target_size=(640, 640), convert_to_gray=False):
    """
    Preprocess all images in a dataset split
    
    Args:
        input_dir: Directory containing the input images
        output_dir: Directory to save the processed images
        split: Dataset split (train, val, test)
        target_size: Target image size
        convert_to_gray: Whether to convert to grayscale
    """
    images_dir = os.path.join(input_dir, split, "images")
    labels_dir = os.path.join(input_dir, split, "labels")
    
    output_images_dir = os.path.join(output_dir, split, "images")
    output_labels_dir = os.path.join(output_dir, split, "labels")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Process each image
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(image_files, desc=f"Preprocessing {split} images"):
        # Preprocess and save the image
        img_path = os.path.join(images_dir, img_file)
        output_img_path = os.path.join(output_images_dir, img_file)
        
        preprocess_image(img_path, output_img_path, target_size, convert_to_gray)
        
        # Copy the corresponding label file if it exists
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            output_label_path = os.path.join(output_labels_dir, label_file)
            shutil.copy2(label_path, output_label_path)
    
    print(f"Preprocessing complete for {split} split.")

# Cell 7: Define data augmentation function
def augment_data(images_dir, labels_dir, output_images_dir, output_labels_dir, augmentation_factor=3):
    """
    Augment training data with various transformations
    
    Args:
        images_dir: Directory containing original images
        labels_dir: Directory containing original labels
        output_images_dir: Directory to save augmented images
        output_labels_dir: Directory to save augmented labels
        augmentation_factor: Number of augmented samples per original image
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Copy original files to output directories
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy2(os.path.join(images_dir, file), os.path.join(output_images_dir, file))
            
            # Copy corresponding label if it exists
            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_labels_dir, label_file))
    
    # Create augmentation pipeline
    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.8),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.8),
        ], p=1.0),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.5),
        ], p=0.8),
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
        ], p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Augment each image
    for file in tqdm(os.listdir(images_dir), desc="Augmenting data"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(images_dir, file)
            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
            
            # Read image and labels
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Parse YOLO labels
            bboxes = []
            class_labels = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
            
            if not bboxes:
                continue
            
            # Generate augmented samples
            for i in range(augmentation_factor):
                try:
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_class_labels = transformed['class_labels']
                    
                    # Save augmented image
                    aug_filename = f"{os.path.splitext(file)[0]}_aug{i}{os.path.splitext(file)[1]}"
                    cv2.imwrite(os.path.join(output_images_dir, aug_filename), aug_image)
                    
                    # Save augmented labels
                    aug_label_filename = f"{os.path.splitext(file)[0]}_aug{i}.txt"
                    with open(os.path.join(output_labels_dir, aug_label_filename), 'w') as f:
                        for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                except Exception as e:
                    print(f"Warning: Augmentation failed for {file} (iteration {i}): {str(e)}")
    
    print(f"Data augmentation complete. Generated {augmentation_factor} augmented samples per original image.")

# Cell 8: Define YAML configuration function
def create_dataset_yaml(base_dir, train_path, val_path, test_path, yaml_path, class_names=None):
    """
    Create a YAML configuration file for the dataset
    
    Args:
        base_dir: Base directory for the project
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        yaml_path: Path to save the YAML file
        class_names: List of class names (default: ["stenosis"])
    """
    if class_names is None:
        # Try to read class names from classes.txt
        classes_file = os.path.join(base_dir, "dataset", "yolo_format", "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = ["stenosis"]
    
    # Create YAML content
    yaml_content = {
        "path": os.path.join(base_dir, "dataset"),
        "train": os.path.join(train_path, "images"),
        "val": os.path.join(val_path, "images"),
        "test": os.path.join(test_path, "images"),
        "nc": len(class_names),
        "names": class_names
    }
    
    # Write YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML configuration created at {yaml_path}")
    return yaml_path

# Cell 9: Define YOLO training function with updated parameters
def train_yolo_model(yaml_path, epochs=110, batch_size=16, img_size=640, patience=25, model_size='m'):
    """
    Train YOLOv8 model for stenosis detection
    
    Args:
        yaml_path: Path to dataset YAML file
        epochs: Number of training epochs (updated to 110 as requested)
        batch_size: Batch size for training
        img_size: Image size for training
        patience: Early stopping patience (updated to 25 for optimal results)
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        
    Returns:
        Path to best weights
    """
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    print(f"Training YOLOv8 model with {epochs} epochs, batch size {batch_size}, image size {img_size}...")
    print(f"Early stopping patience: {patience}")
    
    # Define output directory paths
    output_dir = os.path.join("coronary_stenosis", "yolov8_stenosis_detector")
    weights_dir = os.path.join(output_dir, "weights")
    
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,  # Updated early stopping patience for optimal results
        project="coronary_stenosis",
        name="yolov8_stenosis_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",  # Use SGD optimizer
        cos_lr=True,      # Cosine learning rate scheduler
        amp=True,         # Mixed precision training
        hsv_h=0.015,      # HSV hue augmentation
        hsv_s=0.7,        # HSV saturation augmentation
        hsv_v=0.4,        # HSV value augmentation
        degrees=10.0,     # Rotation augmentation
        scale=0.5,        # Scale augmentation
        flipud=0.5,       # Vertical flip augmentation
        fliplr=0.5,       # Horizontal flip augmentation
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.1,        # Mixup augmentation
        plots=True,       # Generate plots to monitor training
    )
    
    # Find the best weights file directly
    best_weights_path = os.path.join(weights_dir, "best.pt")
    
    # Verify if the file exists
    if not os.path.exists(best_weights_path):
        print(f"Warning: Best weights file not found at {best_weights_path}")
        # Try to find the last weights as a fallback
        last_weights_path = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_weights_path):
            print(f"Using last weights instead: {last_weights_path}")
            best_weights_path = last_weights_path
    
    print(f"Training complete. Best weights saved at: {best_weights_path}")
    
    # Plot Train vs Val Loss/Accuracy to monitor generalization
    try:
        results_csv = os.path.join(output_dir, "results.csv")
        if os.path.exists(results_csv):
            results_df = pd.read_csv(results_csv)
            
            plt.figure(figsize=(12, 6))
            
            # Plot training vs validation loss
            plt.subplot(1, 2, 1)
            plt.plot(results_df['              train/box_loss'], label='Train Loss')
            plt.plot(results_df['              val/box_loss'], label='Val Loss')
            plt.title('Train vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot mAP metrics (as a proxy for accuracy)
            plt.subplot(1, 2, 2)
            plt.plot(results_df['metrics/mAP50(B)'], label='mAP50')
            plt.plot(results_df['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.title('Model Performance (mAP)')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "train_vs_val_performance.png"))
            plt.show()
            
            print("Train vs Val Loss/Accuracy plot created to help monitor how well the model generalizes during training.")
        else:
            print(f"Results file not found: {results_csv}")
    except Exception as e:
        print(f"Error creating Train vs Val performance plot: {str(e)}")
    
    return best_weights_path

# Cell 10: Define main function
def main():
    """
    Main function to run the complete pipeline
    """
    print("Starting Coronary Stenosis Detection and Classification Pipeline")
    
    # Define global constants if they aren't already defined
    global BASE_DIR, YOLO_DIR, PROCESSED_DIR, RESULTS_DIR
    
    # Use relative paths - with fallback for Jupyter notebooks
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # We're in a Jupyter notebook
        BASE_DIR = os.path.abspath('.')
    
    YOLO_DIR = os.path.join(BASE_DIR, "dataset", "yolo_format")
    PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Create necessary directories
    os.makedirs(YOLO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Convert COCO format to YOLO format
    convert_all_datasets()
    
    # Step 2: Preprocess dataset
    for split in ["train", "val", "test"]:
        input_dir = os.path.join(YOLO_DIR, split)
        output_dir = os.path.join(PROCESSED_DIR, split)
        
        if os.path.exists(input_dir):
            preprocess_dataset(input_dir, output_dir, split="", target_size=(640, 640), convert_to_gray=False)
        else:
            print(f"Warning: Input directory {input_dir} not found. Skipping preprocessing for {split}.")
    
    # Step 3: Augment training data
    train_images_dir = os.path.join(PROCESSED_DIR, "train", "images")
    train_labels_dir = os.path.join(PROCESSED_DIR, "train", "labels")
    augmented_images_dir = os.path.join(PROCESSED_DIR, "train_augmented", "images")
    augmented_labels_dir = os.path.join(PROCESSED_DIR, "train_augmented", "labels")
    
    if os.path.exists(train_images_dir) and os.path.exists(train_labels_dir):
        os.makedirs(os.path.join(PROCESSED_DIR, "train_augmented"), exist_ok=True)
        os.makedirs(augmented_images_dir, exist_ok=True)
        os.makedirs(augmented_labels_dir, exist_ok=True)
        
        augment_data(train_images_dir, train_labels_dir, augmented_images_dir, augmented_labels_dir, augmentation_factor=3)
    else:
        print("Warning: Training images or labels not found. Skipping data augmentation.")
    
    # Step 4: Create dataset YAML
    yaml_path = os.path.join(PROCESSED_DIR, "dataset.yaml")
    create_dataset_yaml(
        base_dir=BASE_DIR,
        train_path=os.path.join(PROCESSED_DIR, "train_augmented") if os.path.exists(os.path.join(PROCESSED_DIR, "train_augmented")) else os.path.join(PROCESSED_DIR, "train"),
        val_path=os.path.join(PROCESSED_DIR, "val"),
        test_path=os.path.join(PROCESSED_DIR, "test"),
        yaml_path=yaml_path
    )

    # Step 5: Train YOLOv8 model with updated parameters
    yolo_weights = train_yolo_model(
        yaml_path=yaml_path,
        epochs=110,  # Updated to 110 epochs as requested
        batch_size=16,
        img_size=640,
        patience=25,  # Updated patience for optimal results
        model_size='m'
    )

# Cell 11: Run the main function
if __name__ == "__main__":
    main()
    