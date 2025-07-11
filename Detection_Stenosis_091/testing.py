import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def test_stenosis_detection(model_path, test_images_dir, output_dir=None, conf_threshold=0.25):
    """
    Test stenosis detection on images using trained YOLO model
    
    Args:
        model_path: Path to trained YOLO weights
        test_images_dir: Directory containing test images
        output_dir: Directory to save output images with detections (optional)
        conf_threshold: Confidence threshold for detections
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get list of test images
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} images for testing")
    
    # Process each image
    for img_file in test_images:
        img_path = os.path.join(test_images_dir, img_file)
        print(f"\nProcessing: {img_file}")
        
        # Read original image for display
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Convert from BGR to RGB for visualization
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model.predict(img_path, conf=conf_threshold, verbose=False)[0]
        
        # Check if stenosis was detected
        if len(results.boxes) > 0:
            print(f"Stenosis detected in {img_file}!")
            
            # Create a copy of the image to draw on
            img_with_bbox = original_img_rgb.copy()
            
            # Extract detection information
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            # Draw bounding boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = confidences[i]
                
                # Draw rectangle
                cv2.rectangle(
                    img_with_bbox, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 0), 
                    2
                )
                
                # Add confidence text
                cv2.putText(
                    img_with_bbox, 
                    f"Stenosis: {conf:.2f}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                print(f"  - Bounding box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf:.2f}")
            
            # Display the image with detections
            plt.figure(figsize=(10, 8))
            plt.imshow(img_with_bbox)
            plt.title(f"Stenosis Detection - {img_file}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the image with detections if output directory is provided
            if output_dir:
                output_path = os.path.join(output_dir, f"detection_{img_file}")
                plt.savefig(output_path, bbox_inches='tight')
                print(f"  - Detection saved to: {output_path}")
            
            plt.show()
            
        else:
            print(f"No stenosis detected in {img_file}")
            
            # Display the original image without any bounding box
            plt.figure(figsize=(10, 8))
            plt.imshow(original_img_rgb)
            plt.title(f"No Stenosis Detected - {img_file}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the image if output directory is provided
            if output_dir:
                output_path = os.path.join(output_dir, f"no_detection_{img_file}")
                plt.savefig(output_path, bbox_inches='tight')
                print(f"  - Image saved to: {output_path}")
            
            plt.show()

def process_single_image(model_path, image_path, conf_threshold=0.25):
    """
    Process a single image for stenosis detection
    
    Args:
        model_path: Path to trained YOLO weights
        image_path: Path to the input image
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (has_stenosis, bounding_boxes, confidences)
    """
    # Load YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False, [], []
    
    # Read image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Could not read image: {image_path}")
        return False, [], []
    
    # Run inference
    results = model.predict(image_path, conf=conf_threshold, verbose=False)[0]
    
    # Check if stenosis was detected
    has_stenosis = len(results.boxes) > 0
    
    if has_stenosis:
        # Extract boxes and confidences
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        
        return True, boxes, confidences
    else:
        return False, [], []

def display_results(image_path, has_stenosis, boxes, confidences):
    """
    Display the results of stenosis detection
    
    Args:
        image_path: Path to the input image
        has_stenosis: Boolean indicating if stenosis was detected
        boxes: List of bounding boxes
        confidences: List of confidence scores
    """
    # Read image
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    
    if has_stenosis:
        # Create a copy to draw bounding boxes
        img_with_bbox = original_img_rgb.copy()
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confidences[i]
            
            # Draw rectangle
            cv2.rectangle(
                img_with_bbox, 
                (x1, y1), 
                (x2, y2), 
                (0, 255, 0), 
                2
            )
            
            # Add confidence text
            cv2.putText(
                img_with_bbox, 
                f"Stenosis: {conf:.2f}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        plt.imshow(img_with_bbox)
        plt.title("Stenosis Detected")
    else:
        plt.imshow(original_img_rgb)
        plt.title("No Stenosis Detected")
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define paths
    MODEL_PATH = r"E:\project_091\coronary_stenosis\yolov8_stenosis_detector\weights\best.pt"
    TEST_IMAGES_DIR = r"E:\project_091\dataset\stenosis\test\images"  # Update this to your test images directory
    OUTPUT_DIR = r"E:\project_091\testing_results"    # Optional: directory to save detection results
    
    # Method 1: Test on all images in a directory
    test_stenosis_detection(
        model_path=MODEL_PATH,
        test_images_dir=TEST_IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25
    )
    
    # Method 2: Process a single image
    """
    image_path = "path/to/single/test/image.jpg"  # Update with your image path
    has_stenosis, boxes, confidences = process_single_image(
        model_path=MODEL_PATH,
        image_path=image_path,
        conf_threshold=0.25
    )
    
    if has_stenosis:
        print(f"Stenosis detected in the image!")
        for i, box in enumerate(boxes):
            print(f"Bounding box {i+1}: {box}, Confidence: {confidences[i]:.2f}")
    else:
        print("No stenosis detected in the image.")
    
    # Display results
    display_results(image_path, has_stenosis, boxes, confidences)
    """
