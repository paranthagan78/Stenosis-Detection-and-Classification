import os
import io
import torch
import streamlit as st
from PIL import Image
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from pathlib import Path
import torch.nn as nn
from torchvision import transforms, models
from captum.attr import IntegratedGradients, Occlusion, GradientShap
from ultralytics import YOLO
import torchvision.models as tv_models
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Coronary Artery Stenosis Analysis",
    page_icon="🫀",
    layout="wide"
)

# Function to load CSS
def load_css():
    try:
        with open("styles.css", "r") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as e:
        # Just use some default styles if CSS file not found
        default_css = """
        .result-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #3B82F6;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1E3A8A;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        """
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)

# Call this function ONLY ONCE at the beginning of your script
load_css()

# Constants
CLASS_NAMES = ['RCA', 'LAD', 'LCX']
CLASS_COLORS = {
    'LAD': '#FF5733',  # Red-Orange
    'LCX': '#33FF57',  # Green
    'RCA': '#3357FF',  # Blue
    'No Stenosis': '#888888'  # Gray
}
CLASS_DESCRIPTIONS = {
    'LAD': "Left Anterior Descending artery stenosis - Supplies blood to the front and left side of the heart.",
    'LCX': "Left Circumflex artery stenosis - Supplies blood to the left side and back of the heart.",
    'RCA': "Right Coronary Artery stenosis - Supplies blood to the right ventricle, bottom portion of the left ventricle and back of the heart.",
    'No Stenosis': "No significant stenosis detected in the coronary arteries."
}

# Default paths
DEFAULT_YOLO_MODEL_PATH = r"C:\Users\paran\OneDrive\Desktop\Projects\Stenosis Project\Web_Project\best.pt"
DEFAULT_RESNET_MODEL_PATH = r"C:\Users\paran\OneDrive\Desktop\Projects\Stenosis Project\Classification_Stenosis_091\resnet50_artery_model.pth"
DEFAULT_EFFICIENTNET_MODEL_PATH = r"C:\Users\paran\OneDrive\Desktop\Projects\Stenosis Project\Classification_Stenosis_091\efficientnet_b0_artery_model.pth"
DEFAULT_VGG_MODEL_PATH = r"C:\Users\paran\OneDrive\Desktop\Projects\Stenosis Project\Classification_Stenosis_091\vgg16_artery_model.pth"
DEFAULT_ENSEMBLE_MODEL_PATH = r"C:\Users\paran\OneDrive\Desktop\Projects\Stenosis Project\Classification_Stenosis_091\meta_classifier_model.pth"
DEFAULT_DATASET_PATH = r"C:\Users\paran\OneDrive\Desktop\Projects\Stenosis Project\test\images"

# Label map
LABEL_MAP = {0: "RCA", 1: "LAD", 2: "LCX"}

def load_yolo_model(model_path):
    """
    Load YOLOv8 model for stenosis detection.
    
    Args:
        model_path (str): Path to the YOLOv8 model
        
    Returns:
        model: Loaded YOLOv8 model
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        return None

def load_classification_model(model_path, model_type="resnet"):
    """
    Load the classification model.
    
    Args:
        model_path (str): Path to the saved model file
        model_type (str): Type of model (resnet, efficientnet, vgg)
        
    Returns:
        model: Loaded PyTorch model, device
    """
    try:
        # First check if the file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the appropriate model architecture
        if model_type == "resnet":
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            # Fix the model structure to match the saved weights
            model.fc = nn.Linear(num_features, len(CLASS_NAMES))
        elif model_type == "efficientnet":
            model = tv_models.efficientnet_b0(weights=None)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
        elif model_type == "vgg":
            model = tv_models.vgg16(weights=None)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, len(CLASS_NAMES))
        elif model_type == "ensemble":
            # Define the ensemble model with 32 neurons to match the saved weights
            class EnsembleModel(nn.Module):
                def __init__(self):
                    super(EnsembleModel, self).__init__()
                    self.fc1 = nn.Linear(len(CLASS_NAMES) * 3, 32)  # Changed from 64 to 32
                    self.fc2 = nn.Linear(32, len(CLASS_NAMES))      # Changed input from 64 to 32
                
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            model = EnsembleModel()
        else:
            st.error(f"Unknown model type: {model_type}")
            return None, None
        
        # Load model weights
        try:
            # Use map_location to avoid CUDA errors
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model = model.to(device)
            model.eval()
            
            return model, device
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            # Create a more detailed error message
            if isinstance(e, RuntimeError) and "Missing key(s)" in str(e):
                st.error("Model structure doesn't match saved weights. This might be because the model was saved with a different architecture.")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None, None
            
def detect_stenosis(model, image_np):
    """
    Detect stenosis in the image using YOLOv8.
    
    Args:
        model: YOLOv8 model
        image_np: Numpy array of the image
        
    Returns:
        results: YOLOv8 detection results
        annotated_image: Image with bounding boxes
    """
    try:
        if model is None:
            st.error("YOLO model not loaded properly")
            return None, None
            
        # Run inference
        results = model(image_np)
        
        # Get the annotated image with bounding boxes
        annotated_image = results[0].plot()
        
        return results[0], annotated_image
    except Exception as e:
        st.error(f"Error during stenosis detection: {e}")
        return None, None

def classify_stenosis(models, image, devices):
    """
    Classify the type of stenosis in the image using multiple models.
    
    Args:
        models: Dictionary of PyTorch models
        image: PIL Image
        devices: Dictionary of torch devices
        
    Returns:
        dict: Model predictions and confidences
    """
    try:
        # Check if models dictionary is empty
        if not models:
            st.error("No classification models loaded")
            return None
            
        # Check if image is valid
        if image is None:
            st.error("Invalid image for classification")
            return None
            
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert and preprocess
        image_tensor = preprocess(image)
        
        results = {}
        
        # Make predictions with individual models first (excluding ensemble)
        individual_outputs = []
        model_probs = {}
        
        for model_name, model in models.items():
            if model is None or model_name == 'ensemble':
                continue
                
            device = devices[model_name]
            model_tensor = image_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(model_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get all probabilities
            all_probs = probabilities.squeeze().cpu().numpy()
            model_probs[model_name] = all_probs
            individual_outputs.append(all_probs)
            
            predicted_class_idx = predicted.item()
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            confidence_score = confidence.item()
            
            results[model_name] = {
                'class_idx': predicted_class_idx,
                'class_name': predicted_class_name,
                'confidence': confidence_score,
                'probabilities': all_probs
            }
        
        # Now handle the ensemble model if it exists
        if 'ensemble' in models and models['ensemble'] is not None:
            try:
                # Combine individual model outputs for ensemble input
                if individual_outputs:
                    ensemble_input = torch.tensor(np.concatenate(individual_outputs)).float()
                    ensemble_input = ensemble_input.unsqueeze(0).to(devices['ensemble'])
                    
                    with torch.no_grad():
                        ensemble_output = models['ensemble'](ensemble_input)
                        ensemble_probs = F.softmax(ensemble_output, dim=1)
                        confidence, predicted = torch.max(ensemble_probs, 1)
                    
                    all_probs = ensemble_probs.squeeze().cpu().numpy()
                    predicted_class_idx = predicted.item()
                    predicted_class_name = CLASS_NAMES[predicted_class_idx]
                    confidence_score = confidence.item()
                    
                    results['ensemble'] = {
                        'class_idx': predicted_class_idx,
                        'class_name': predicted_class_name,
                        'confidence': confidence_score,
                        'probabilities': all_probs
                    }
            except Exception as e:
                st.warning(f"Ensemble model prediction failed: {e}")
                # If ensemble fails, don't include it in results
        
        # If we have ensemble results, make it the primary result
        if 'ensemble' in results:
            results['primary'] = results['ensemble'] 
        elif 'resnet' in results:
            # Otherwise use resnet as the primary
            results['primary'] = results['resnet']
        else:
            # If neither ensemble nor resnet, use the first available model
            results['primary'] = next(iter(results.values()))
            
        return results
    except Exception as e:
        st.error(f"Error during stenosis classification: {e}")
        return None
    
def tensor_to_image(tensor):
    """Convert a normalized tensor to a displayable image."""
    # Move to CPU if needed
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # Remove batch dimension and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from CxHxW to HxWxC format
    img = tensor.permute(1, 2, 0).numpy()
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    
    # Clip to [0, 1]
    img = np.clip(img, 0, 1)
    
    return img

def apply_integrated_gradients(model, image_tensor, target_class, device):
    """Apply Integrated Gradients for model interpretability."""
    try:
        # Create IntegratedGradients object
        ig = IntegratedGradients(model)
        
        # Ensure the input image requires gradient
        input_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)
        
        # Compute attributions
        attributions_ig = ig.attribute(input_tensor, target=target_class)
        
        # Convert to numpy and visualize
        attr_np = attributions_ig.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        
        # Normalize attributions for visualization
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        
        return attr_np
    except Exception as e:
        st.error(f"Error generating Integrated Gradients: {e}")
        return None

def apply_gradient_shap(model, image_tensor, target_class, device):
    """Apply Gradient SHAP for model interpretability."""
    try:
        # Create a baseline (black image)
        baseline = torch.zeros_like(image_tensor).to(device)
        
        # Create GradientShap object
        shap = GradientShap(model)
        
        # Ensure the input image requires gradient
        input_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)
        
        # Compute attributions
        attributions_shap = shap.attribute(input_tensor, 
                                        baselines=baseline,
                                        target=target_class,
                                        n_samples=5)
        
        # Convert to numpy and visualize
        attr_shap_np = attributions_shap.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        
        # Normalize attributions for visualization
        attr_shap_np = np.abs(attr_shap_np)  # Take absolute value for visualization
        attr_shap_np = (attr_shap_np - attr_shap_np.min()) / (attr_shap_np.max() - attr_shap_np.min() + 1e-8)
        
        return attr_shap_np
    except Exception as e:
        st.error(f"Error generating Gradient SHAP: {e}")
        return None

def apply_occlusion(model, image_tensor, target_class, device):
    """Apply Occlusion for model interpretability."""
    try:
        # Create Occlusion object
        occlusion = Occlusion(model)
        
        # Ensure the input image requires gradient
        input_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)
        
        # Compute attributions with a sliding window
        window_size = 10
        stride = 5
        
        attributions_occ = occlusion.attribute(input_tensor,
                                           target=target_class,
                                           sliding_window_shapes=(3, window_size, window_size),
                                           strides=(3, stride, stride))
        
        # Convert to numpy and visualize
        attr_occ_np = attributions_occ.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        
        # Sum across channels
        attr_occ_np = np.mean(attr_occ_np, axis=2)
        
        # Normalize attributions for visualization
        attr_occ_np = (attr_occ_np - attr_occ_np.min()) / (attr_occ_np.max() - attr_occ_np.min() + 1e-8)
        
        return attr_occ_np
    except Exception as e:
        st.error(f"Error generating Occlusion analysis: {e}")
        return None

def apply_xai_to_image(model, image_tensor, device, class_names, predicted_class_idx):
    """Apply XAI techniques to the model prediction."""
    # First check if model is None
    if model is None:
        st.error("Model not available for XAI analysis")
        return
        
    model.eval()
        
    # Create column layout for XAI methods
    col1, col2 = st.columns(2)
    
    # 1. Integrated Gradients
    with col1:
        st.subheader("Integrated Gradients")
        
        attr_np = apply_integrated_gradients(model, image_tensor, predicted_class_idx, device)
        
        if attr_np is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(attr_np)
            ax.set_title(f"Features contributing to {class_names[predicted_class_idx]}")
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretation**: Highlighted areas show pixels that most influenced the model's prediction.
            Brighter regions indicate stronger positive influence on the classification.
            """)
    
    # 2. GradientShap
    with col2:
        st.subheader("Gradient SHAP")
        
        attr_shap_np = apply_gradient_shap(model, image_tensor, predicted_class_idx, device)
        
        if attr_shap_np is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(attr_shap_np)
            ax.set_title(f"SHAP values for {class_names[predicted_class_idx]}")
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretation**: Gradient SHAP highlights features that contribute to the prediction 
            relative to a baseline (black image). This shows which parts of the image were most
            important for the classification.
            """)
    
    # 3. Occlusion as a third method
    st.subheader("Occlusion Analysis")
    
    attr_occ_np = apply_occlusion(model, image_tensor, predicted_class_idx, device)
    
    if attr_occ_np is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(attr_occ_np, cmap='hot')
        ax.set_title(f"Occlusion Analysis for {class_names[predicted_class_idx]}")
        ax.axis('off')
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation**: This heatmap shows how the model's prediction changes when different
        areas of the image are occluded (blocked out). Brighter areas indicate regions where
        occlusion significantly decreases the prediction score, meaning these regions are
        crucial for the model's decision.
        """)
    
    st.markdown("""
    ### XAI Summary
    These visualizations help understand which parts of the angiogram image are most important 
    for the model's classification decision. The highlighted regions often correspond to 
    anatomical structures or abnormalities that radiologists would focus on, such as the 
    specific coronary artery and areas of potential stenosis.
    """)

def display_model_predictions(results):
    """Display classification results from multiple models."""
    if not results:
        return
    
    # Primary result
    primary = results['primary']
    predicted_class_name = primary['class_name']
    confidence_score = primary['confidence']
    
    # Display result with color
    result_color = CLASS_COLORS[predicted_class_name]
    st.markdown(f"""
    <div class="result-card" style="background-color: {result_color}30; border: 2px solid {result_color};">
        <h3 style="color: {result_color};">Prediction: {predicted_class_name}</h3>
        <p><b>Confidence:</b> {confidence_score:.2%}</p>
        <p>{CLASS_DESCRIPTIONS[predicted_class_name]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show all models' predictions
    st.subheader("Model Predictions")
    
    # Create comparison table
    models_data = []
    for model_name, result in results.items():
        if model_name != 'primary':
            models_data.append({
                'Model': model_name.capitalize(),
                'Prediction': result['class_name'],
                'Confidence': result['confidence']
            })
    
    if models_data:
        models_df = pd.DataFrame(models_data)
        st.table(models_df.style.format({'Confidence': '{:.2%}'}))
    
    # Show probability distribution for primary model
    st.subheader("Probability Distribution")
    all_probs = primary['probabilities']
    
    probs_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Probability': all_probs
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Create bar chart for probabilities
    bars = sns.barplot(
        x='Class', 
        y='Probability', 
        data=probs_df,
        palette=[CLASS_COLORS[c] for c in CLASS_NAMES],
        ax=ax
    )
    
    # Add probability values on top of bars
    for i, p in enumerate(all_probs):
        ax.text(i, p + 0.02, f"{p:.2%}", ha='center')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Class')
    ax.set_title('Class Probabilities')
    
    st.pyplot(fig)

def scan_dataset(dataset_path):
    """
    Scan the dataset directory and return statistics about the data.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        dict: Statistics about the dataset
    """
    stats = {
        'total_images': 0,
        'class_distribution': {},
        'class_paths': {},
        'image_dimensions': [],
        'sample_images': {}
    }
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        return stats
    
    # Scan all class directories
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_dir):
            stats['class_distribution'][class_name] = 0
            stats['class_paths'][class_name] = []
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        stats['class_distribution'][class_name] = len(image_files)
        stats['total_images'] += len(image_files)
        stats['class_paths'][class_name] = [os.path.join(class_dir, f) for f in image_files]
        
        # Get sample images
        if image_files:
            sample_count = min(5, len(image_files))
            stats['sample_images'][class_name] = random.sample([os.path.join(class_dir, f) for f in image_files], sample_count)
        
        # Get image dimensions from first 10 images
        sample_for_dims = image_files[:10] if len(image_files) > 10 else image_files
        for img_file in sample_for_dims:
            try:
                with Image.open(os.path.join(class_dir, img_file)) as img:
                    stats['image_dimensions'].append(img.size)
            except Exception:
                pass
    
    return stats

def display_eda(dataset_path):
    """Display EDA visualizations for the dataset."""
    stats = scan_dataset(dataset_path)
    
    # Display basic dataset statistics
    st.header("📊 Dataset Statistics")
    
    if stats['total_images'] == 0:
        st.warning(f"No images found at {dataset_path}")
        return
    
    st.write(f"**Total Images:** {stats['total_images']}")
    
    # Class distribution
    st.subheader("Class Distribution")
    class_dist_df = pd.DataFrame({
        'Class': list(stats['class_distribution'].keys()),
        'Count': list(stats['class_distribution'].values())
    })
    
    # Calculate percentages
    class_dist_df['Percentage'] = class_dist_df['Count'] / class_dist_df['Count'].sum() * 100
    
    # Sort by count
    class_dist_df = class_dist_df.sort_values('Count', ascending=False)
    
    # Display statistics
    st.dataframe(class_dist_df)
    
    # Create class distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    sns.barplot(data=class_dist_df, x='Class', y='Count', palette=[CLASS_COLORS[c] for c in class_dist_df['Class']], ax=ax1)
    ax1.set_title('Number of Images per Class')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Class')
    
    # Add count labels on top of bars
    for i, count in enumerate(class_dist_df['Count']):
        ax1.text(i, count + 1, str(count), ha='center')
    
    # Pie chart
    ax2.pie(class_dist_df['Count'], labels=class_dist_df['Class'], autopct='%1.1f%%', 
           colors=[CLASS_COLORS[c] for c in class_dist_df['Class']])
    ax2.set_title('Percentage of Images per Class')
    
    st.pyplot(fig)
    
    # Display image dimensions
    if stats['image_dimensions']:
        st.subheader("Image Dimensions")
        
        # Convert to DataFrame
        dims_df = pd.DataFrame(stats['image_dimensions'], columns=['Width', 'Height'])
        dims_df['Aspect Ratio'] = dims_df['Width'] / dims_df['Height']
        
        # Display summary statistics
        st.write("**Image Dimension Statistics:**")
        st.dataframe(dims_df.describe())
        
        # Plot dimension distributions
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.histplot(dims_df['Width'], kde=True, ax=ax1)
        ax1.set_title('Width Distribution')
        
        sns.histplot(dims_df['Height'], kde=True, ax=ax2)
        ax2.set_title('Height Distribution')
        
        sns.histplot(dims_df['Aspect Ratio'], kde=True, ax=ax3)
        ax3.set_title('Aspect Ratio Distribution')
        
        st.pyplot(fig)
    
    # Sample images
    st.subheader("Sample Images")
    
    # Create tabs for each class
    tabs = st.tabs(CLASS_NAMES)
    
    for i, class_name in enumerate(CLASS_NAMES):
        with tabs[i]:
            if class_name in stats['sample_images'] and stats['sample_images'][class_name]:
                st.write(f"**{class_name}**: {CLASS_DESCRIPTIONS.get(class_name, '')}")
                
                # Display sample images in a grid
                cols = st.columns(min(5, len(stats['sample_images'][class_name])))
                for j, img_path in enumerate(stats['sample_images'][class_name]):
                    try:
                        img = Image.open(img_path)
                        cols[j].image(img, caption=os.path.basename(img_path), use_container_width=True)
                    except Exception as e:
                        cols[j].error(f"Could not load image: {e}")
            else:
                st.warning(f"No sample images for class {class_name}")

def process_directory_data(uploaded_folder):
    """Process directory data for better analysis"""
    stats = {
        'total_images': 0,
        'class_distribution': {},
        'class_paths': {},
        'image_dimensions': [],
        'sample_images': {}
    }
    
    # Initialize class distributions
    for class_name in CLASS_NAMES:
        stats['class_distribution'][class_name] = 0
        stats['class_paths'][class_name] = []
        stats['sample_images'][class_name] = []
    
    # Add 'Unknown' category for unclassified images
    stats['class_distribution']['Unknown'] = 0
    stats['class_paths']['Unknown'] = []
    stats['sample_images']['Unknown'] = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    # Process each uploaded file
    for uploaded_file in uploaded_folder:
        try:
            # Extract class from file name if possible
            file_name = uploaded_file.name
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Skip non-image files
            if file_ext not in image_extensions:
                continue
                
            # Try to determine class from file name
            found_class = None
            for class_name in CLASS_NAMES:
                if class_name.lower() in file_name.lower():
                    found_class = class_name
                    break
            
            # If class not found, use 'Unknown'
            if found_class is None:
                found_class = 'Unknown'
            
            # Process the image
            try:
                img = Image.open(uploaded_file).convert('RGB')
                
                # Store dimensions
                stats['image_dimensions'].append(img.size)
                
                # Create a buffer to store the image in memory
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                
                # Store the buffer in class_paths
                stats['class_paths'][found_class].append(buffer)
                
                # Update counts
                stats['class_distribution'][found_class] += 1
                stats['total_images'] += 1
                
                # If we have fewer than 5 samples for this class, add to sample_images
                if len(stats['sample_images'][found_class]) < 5:
                    # Create a copy of the buffer for samples
                    sample_buffer = io.BytesIO()
                    img.save(sample_buffer, format="PNG")
                    sample_buffer.seek(0)
                    stats['sample_images'][found_class].append((file_name, sample_buffer))
                
            except Exception as e:
                st.warning(f"Could not process image {file_name}: {e}")
                
        except Exception as e:
            st.warning(f"Error with file {uploaded_file.name}: {e}")
    
    return stats

def display_eda_from_upload(uploaded_folder):
    """Display EDA visualizations for uploaded dataset."""
    stats = process_directory_data(uploaded_folder)
    
    # Display basic dataset statistics
    st.header("📊 Dataset Statistics")
    
    if stats['total_images'] == 0:
        st.warning("No valid images found in the uploaded files")
        return
    
    st.write(f"**Total Images:** {stats['total_images']}")
    
    # Remove empty classes
    filtered_class_dist = {k: v for k, v in stats['class_distribution'].items() if v > 0}
    
    # Class distribution
    st.subheader("Class Distribution")
    class_dist_df = pd.DataFrame({
        'Class': list(filtered_class_dist.keys()),
        'Count': list(filtered_class_dist.values())
    })
    
    # Calculate percentages
    class_dist_df['Percentage'] = class_dist_df['Count'] / class_dist_df['Count'].sum() * 100
    
    # Sort by count
    class_dist_df = class_dist_df.sort_values('Count', ascending=False)
    
    # Display statistics
    st.dataframe(class_dist_df.style.format({'Percentage': '{:.1f}%'}))
    
    # Create class distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get colors for each class
    class_colors = []
    for cls in class_dist_df['Class']:
        if cls in CLASS_COLORS:
            class_colors.append(CLASS_COLORS[cls])
        else:
            class_colors.append('#AAAAAA')  # Default gray for unknown classes
    
    # Bar plot - FIXED CODE HERE
    bars = sns.barplot(data=class_dist_df, x='Class', y='Count', palette=class_colors, ax=ax1)
    ax1.set_title('Number of Images per Class')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Class')
    
    # Add count labels on top of bars - FIXED CODE HERE
    for i, p in enumerate(ax1.patches):
        ax1.text(
            p.get_x() + p.get_width()/2, 
            p.get_height() + 0.3, 
            str(int(p.get_height())), 
            ha='center'
        )
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        class_dist_df['Count'], 
        labels=class_dist_df['Class'], 
        autopct='%1.1f%%',
        colors=class_colors,
        startangle=90
    )
    # Make percentage labels more readable
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_weight('bold')
    
    ax2.set_title('Percentage of Images per Class')
    
    plt.tight_layout()
    st.pyplot(fig)
        
    # Display image dimensions analysis
    if stats['image_dimensions']:
        st.subheader("Image Dimensions Analysis")
        
        # Convert to DataFrame
        dims_df = pd.DataFrame(stats['image_dimensions'], columns=['Width', 'Height'])
        dims_df['Aspect Ratio'] = dims_df['Width'] / dims_df['Height']
        dims_df['Area (pixels)'] = dims_df['Width'] * dims_df['Height']
        
        # Display summary statistics
        st.write("**Dimension Statistics:**")
        st.dataframe(dims_df.describe().style.format("{:.1f}"))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Width histogram
        sns.histplot(dims_df['Width'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        
        # Height histogram
        sns.histplot(dims_df['Height'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        
        # Aspect ratio histogram
        sns.histplot(dims_df['Aspect Ratio'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
        
        # Width vs Height scatter
        sns.scatterplot(data=dims_df, x='Width', y='Height', ax=axes[1, 1])
        axes[1, 1].set_title('Width vs Height')
        axes[1, 1].set_xlabel('Width (pixels)')
        axes[1, 1].set_ylabel('Height (pixels)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add more detailed image size analysis
        st.subheader("Image Size Categories")
        
        # Define size categories
        def categorize_size(row):
            area = row['Width'] * row['Height']
            if area < 100000:  # e.g., smaller than 316x316
                return 'Small'
            elif area < 500000:  # e.g., smaller than 707x707
                return 'Medium'
            else:
                return 'Large'
        
        dims_df['Size Category'] = dims_df.apply(categorize_size, axis=1)
        
        # Display size distribution
        size_counts = dims_df['Size Category'].value_counts().reset_index()
        size_counts.columns = ['Size Category', 'Count']
        
        # Calculate percentages
        size_counts['Percentage'] = size_counts['Count'] / size_counts['Count'].sum() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=size_counts, x='Size Category', y='Count', ax=ax, order=['Small', 'Medium', 'Large'])
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{int(p.get_height())} ({size_counts.iloc[i]["Percentage"]:.1f}%)', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom')
        
        ax.set_title('Distribution of Image Sizes')
        st.pyplot(fig)
    
    # Sample images
    st.subheader("Sample Images from Dataset")
    
    # Only show tabs for classes that have images
    non_empty_classes = [cls for cls in list(stats['class_distribution'].keys()) 
                         if cls in stats['sample_images'] and stats['sample_images'][cls]]
    
    if non_empty_classes:
        tabs = st.tabs(non_empty_classes)
        
        for i, class_name in enumerate(non_empty_classes):
            with tabs[i]:
                st.write(f"**{class_name}**: {CLASS_DESCRIPTIONS.get(class_name, 'No description available')}")
                
                # Display sample images in a grid
                samples = stats['sample_images'][class_name]
                
                if samples:
                    cols = st.columns(min(len(samples), 5))
                    for j, (file_name, img_buffer) in enumerate(samples):
                        try:
                            img_buffer.seek(0)  # Reset buffer position
                            img = Image.open(img_buffer)
                            cols[j].image(img, caption=file_name, use_container_width=True)
                        except Exception as e:
                            cols[j].error(f"Could not load image: {e}")
                else:
                    st.warning(f"No sample images for class {class_name}")
    else:
        st.warning("No sample images available for display")
        
    # Add recommendation for dataset balance
    if stats['total_images'] > 0:
        st.subheader("Dataset Balance Analysis")
        
        # Calculate imbalance
        max_class = class_dist_df.iloc[0]['Count']
        min_class = class_dist_df.iloc[-1]['Count']
        
        if min_class == 0:
            # Find smallest non-zero class
            min_class = min([v for v in class_dist_df['Count'] if v > 0])
        
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio > 3:
            st.warning(f"""
            **Dataset Imbalance Warning**:
            Your dataset has a significant imbalance with a ratio of {imbalance_ratio:.1f}:1 between the largest and smallest classes.
            This may lead to biased model performance. Consider:
            - Adding more images to underrepresented classes
            - Using data augmentation techniques
            - Applying class weighting during training
            """)
        else:
            st.success(f"""
            **Dataset Balance**:
            Your dataset has a reasonable balance with a ratio of {imbalance_ratio:.1f}:1 between classes.
            """)

def process_batch_files(batch_files):
    """Process batch files and return images with their paths and classes"""
    batch_data = []
    
    for uploaded_file in batch_files:
        try:
            # Try to determine class from file name
            file_name = uploaded_file.name
            found_class = None
            for class_name in CLASS_NAMES:
                if class_name.lower() in file_name.lower():
                    found_class = class_name
                    break
            
            # If class not found in name, use a default
            if found_class is None:
                found_class = 'Unknown'
            
            # Open and process the image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Add to batch data
            batch_data.append({
                'file_name': file_name,
                'true_class': found_class,
                'image': image
            })
            
        except Exception as e:
            st.warning(f"Could not process file {uploaded_file.name}: {e}")
    
    return batch_data

def run_batch_analysis(yolo_model, classification_models, devices, batch_files):
    """Run detection and classification on a batch of uploaded images."""
    
    # Process uploaded files
    batch_data = process_batch_files(batch_files)
    
    if not batch_data:
        st.warning("No valid images were found in the uploaded files.")
        return
    
    # Prepare dataframe to store results
    results = []
    
    # Preprocess function for classification
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each image
    for i, item in enumerate(batch_data):
        try:
            status_text.text(f"Processing image {i+1}/{len(batch_data)}: {item['file_name']}")
            progress_bar.progress((i) / len(batch_data))
            
            # Get image data
            image = item['image']
            true_class = item['true_class']
            file_name = item['file_name']
            
            # Convert to numpy for YOLOv8
            image_np = np.array(image)
            
            # YOLO detection
            detection_results, annotated_image = detect_stenosis(yolo_model, image_np)
            
            has_stenosis = False
            detected_boxes = 0
            if detection_results and len(detection_results.boxes) > 0:
                has_stenosis = True
                detected_boxes = len(detection_results.boxes)
            
            # Classification with all available models
            classification_results = classify_stenosis(classification_models, image, devices)
            
            # Create result entry
            result_entry = {
                'Image': file_name,
                'True Class': true_class,
                'Has Stenosis': has_stenosis,
                'Detected Regions': detected_boxes,
                'Original Image': image,
                'Annotated Image': annotated_image if annotated_image is not None else None
            }
            
            # Add results from each model
            for model_name, model_result in classification_results.items():
                if model_name != 'primary':
                    result_entry[f'{model_name.capitalize()} Prediction'] = model_result['class_name']
                    result_entry[f'{model_name.capitalize()} Confidence'] = model_result['confidence']
            
            # Add primary prediction
            result_entry['Primary Prediction'] = classification_results['primary']['class_name']
            result_entry['Primary Confidence'] = classification_results['primary']['confidence']
            
            # Add to results
            results.append(result_entry)
            
        except Exception as e:
            st.warning(f"Error processing {item['file_name']}: {e}")
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Analyze results
    st.header("🔄 Batch Analysis Results")
    
    # Basic statistics
    st.subheader("Summary Statistics")
    
    total_images = len(df)
    stenosis_detected = df['Has Stenosis'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Stenosis Detected", f"{stenosis_detected} ({stenosis_detected/total_images:.1%})")
    
    # If true classes are provided, calculate classification accuracy
    if not all(df['True Class'] == 'Unknown'):
        # Calculate accuracy for primary model
        accuracy = (df['Primary Prediction'] == df['True Class']).mean()
        with col3:
            st.metric("Classification Accuracy", f"{accuracy:.1%}")
            
        # Show confusion matrix
        st.subheader("Classification Confusion Matrix")
        
        # Only include entries with known true classes
        valid_mask = df['True Class'] != 'Unknown'
        if sum(valid_mask) > 0:
            # Create confusion matrix
            conf_matrix = pd.crosstab(
                df.loc[valid_mask, 'True Class'], 
                df.loc[valid_mask, 'Primary Prediction'],
                rownames=['True Class'], 
                colnames=['Predicted Class'],
                normalize='index'
            )
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', ax=ax)
            ax.set_title('Classification Confusion Matrix')
            st.pyplot(fig)
    
    # Show prediction distribution
    st.subheader("Prediction Distribution")
    
    # Create counts of predictions
    pred_counts = df['Primary Prediction'].value_counts().reset_index()
    pred_counts.columns = ['Class', 'Count']
    
    # Calculate percentages
    pred_counts['Percentage'] = pred_counts['Count'] / pred_counts['Count'].sum() * 100
    
    # Create color map for classes
    class_colors = [CLASS_COLORS.get(cls, '#AAAAAA') for cls in pred_counts['Class']]
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(pred_counts['Class'], pred_counts['Count'], color=class_colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    ax.set_title('Distribution of Predictions')
    ax.set_ylabel('Number of Images')
    st.pyplot(fig)
    
    # Display table of results
    st.subheader("Results Table")
    
    # Create a displayable DataFrame (without images)
    display_columns = [col for col in df.columns if not col.endswith('Image')]
    display_df = df[display_columns].copy()
    
    # Format confidence values
    for col in display_df.columns:
        if 'Confidence' in col:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
    
    st.dataframe(display_df)
    
    # Sample visualizations
    st.subheader("Sample Predictions")
    
    # Select a few random samples to display (or all if less than 5)
    num_samples = min(5, len(df))
    sample_indices = np.random.choice(len(df), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        
        st.markdown(f"### Sample {i+1}: {row['Image']}")
        
        col1, col2 = st.columns(2)
        
        # Display original image
        col1.image(row['Original Image'], caption="Original Image", use_container_width=True)
        
        # Display annotated image if available
        if row['Annotated Image'] is not None:
            col2.image(row['Annotated Image'], caption="Detection Results", use_container_width=True)
        else:
            col2.warning("No detection visualization available")
        
        # Display prediction info
        st.markdown(f"""
        **Results:**
        - True Class (from filename): {row['True Class']}
        - Primary Prediction: {row['Primary Prediction']} (Confidence: {row['Primary Confidence']:.2%})
        - Stenosis Detected: {"Yes" if row['Has Stenosis'] else "No"} ({row['Detected Regions']} regions)
        """)
        
        # Show individual model predictions if available
        model_preds = []
        for col_name in row.index:
            if 'Prediction' in col_name and col_name != 'Primary Prediction':
                model_name = col_name.replace(' Prediction', '')
                confidence_col = f"{model_name} Confidence"
                if confidence_col in row.index:
                    model_preds.append(f"- {model_name}: {row[col_name]} (Confidence: {row[confidence_col]:.2%})")
        
        if model_preds:
            st.markdown("**Individual Model Predictions:**")
            for pred in model_preds:
                st.markdown(pred)
        
        st.markdown("---")
    
    return df

def show_about():
    """Show information about the application."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Coronary Artery Stenosis Analysis
    
    This application provides tools for detecting and classifying coronary artery stenosis from angiogram images. Stenosis is the abnormal narrowing of a blood vessel, which can restrict blood flow and lead to serious health issues including heart attacks.
    
    ### Features
    
    1. **Stenosis Detection**: Uses a YOLOv8 object detection model to identify areas of potential stenosis in coronary angiograms.
    
    2. **Artery Classification**: Classifies the affected artery into one of three categories:
       - **RCA (Right Coronary Artery)**: Supplies blood to the right ventricle, bottom portion of the left ventricle and back of the heart.
       - **LAD (Left Anterior Descending)**: Supplies blood to the front and left side of the heart.
       - **LCX (Left Circumflex)**: Supplies blood to the left side and back of the heart.
    
    3. **Explainable AI (XAI)**: Provides visualizations to help understand what features the models are using to make predictions:
       - Integrated Gradients
       - Gradient SHAP
       - Occlusion Analysis
    
    4. **Dataset Analysis**: Provides exploratory data analysis of the training dataset.
    
    5. **Batch Analysis**: Run the models on multiple images to evaluate performance.
    
    ### Models
    
    This application uses:
    
    1. **YOLOv8** for stenosis detection
    2. **ResNet50**, **EfficientNet-B0**, **VGG16** for artery classification
    3. **Ensemble Model** that combines the strengths of the individual classifiers
    
    ### Technical Details
    
    The models were trained on coronary angiogram images with labeled stenosis regions. The deep learning architectures were fine-tuned specifically for this medical imaging task.
    
    ### Clinical Importance
    
    Accurate and timely identification of coronary stenosis is crucial for:
    
    - Diagnosing coronary artery disease
    - Planning appropriate interventions (e.g., stenting, bypass surgery)
    - Predicting and preventing potential heart attacks
    - Monitoring disease progression over time
    
    ### Disclaimer
    
    This application is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for medical advice.
    """)

def main():
    """Main function to run the Streamlit app."""
    # Display header
    st.markdown("<h1 class='main-header'>Coronary Artery Stenosis Analysis</h1>", unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🔍 Image Analysis", "📊 Dataset Analysis (EDA)", "🔄 Batch Analysis", "ℹ️ About"]
    )
    
    # Path configurations in sidebar
    st.sidebar.title("Model Configuration")
    
    # YOLOv8 model path
    yolo_model_path = st.sidebar.text_input(
        "YOLOv8 Model Path",
        value=DEFAULT_YOLO_MODEL_PATH
    )
    
    # Classification model paths
    resnet_model_path = st.sidebar.text_input(
        "ResNet Model Path",
        value=DEFAULT_RESNET_MODEL_PATH
    )
    
    efficientnet_model_path = st.sidebar.text_input(
        "EfficientNet Model Path",
        value=DEFAULT_EFFICIENTNET_MODEL_PATH
    )
    
    vgg_model_path = st.sidebar.text_input(
        "VGG Model Path",
        value=DEFAULT_VGG_MODEL_PATH
    )
    
    ensemble_model_path = st.sidebar.text_input(
        "Ensemble Model Path",
        value=DEFAULT_ENSEMBLE_MODEL_PATH
    )
    
    # Dataset path for batch analysis only (hidden in other modes)
    dataset_path = DEFAULT_DATASET_PATH
    if app_mode == "🔄 Batch Analysis":
        dataset_path = st.sidebar.text_input(
            "Dataset Path", 
            value=DEFAULT_DATASET_PATH,
            key="batch_dataset_path"
        )
    
    # Load models based on selected mode
    yolo_model = None
    classification_models = {}
    devices = {}
    
    # Common function to load all models
    def load_all_models():
        nonlocal yolo_model, classification_models, devices
        
        # Load YOLOv8 model
        with st.spinner("Loading YOLOv8 model..."):
            yolo_model = load_yolo_model(yolo_model_path)
            if yolo_model is None:
                st.sidebar.error("❌ Failed to load YOLOv8 model")
        
        # Load classification models
        model_loading_progress = st.progress(0)
        
        # ResNet
        model_loading_progress.progress(0)
        with st.spinner("Loading ResNet model..."):
            classification_models['resnet'], devices['resnet'] = load_classification_model(
                resnet_model_path, model_type="resnet"
            )
        model_loading_progress.progress(25)
        
        # EfficientNet
        with st.spinner("Loading EfficientNet model..."):
            classification_models['efficientnet'], devices['efficientnet'] = load_classification_model(
                efficientnet_model_path, model_type="efficientnet"
            )
        model_loading_progress.progress(50)
        
        # VGG
        with st.spinner("Loading VGG model..."):
            classification_models['vgg'], devices['vgg'] = load_classification_model(
                vgg_model_path, model_type="vgg"
            )
        model_loading_progress.progress(75)
        
        # Ensemble
        with st.spinner("Loading Ensemble model..."):
            classification_models['ensemble'], devices['ensemble'] = load_classification_model(
                ensemble_model_path, model_type="ensemble"
            )
        model_loading_progress.progress(100)
        
        # Check if at least one model loaded
        if all(model is None for model in classification_models.values()):
            st.sidebar.error("❌ Failed to load any classification models")
        else:
            st.sidebar.success("✅ Models loaded successfully")
            
        # Remove progress bar
        model_loading_progress.empty()
            
    # Run different modes based on selection
    if app_mode == "🔍 Image Analysis":
        # Load models
        load_all_models()
        
        st.markdown("<h2 class='section-header'>🔍 Coronary Angiogram Analysis</h2>", unsafe_allow_html=True)
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an angiogram image", type=["jpg", "jpeg", "png"])
        
        # Sample images option
        st.markdown("### Or select a sample image:")
        sample_stats = scan_dataset(dataset_path)
        
        # Create a list of all sample images
        all_samples = []
        for class_name, paths in sample_stats['class_paths'].items():
            if paths:
                all_samples.extend([(os.path.basename(p), p, class_name) for p in paths[:5]])
        
        if all_samples:
            # Create a selectbox with image names
            sample_names = ["None"] + [f"{name} ({cls})" for name, _, cls in all_samples]
            selected_sample_name = st.selectbox("Sample images", sample_names)
            
            # Process selected sample
            if selected_sample_name != "None":
                selected_idx = sample_names.index(selected_sample_name) - 1  # Subtract 1 for "None"
                selected_sample_path = all_samples[selected_idx][1]
                
                # Use the selected sample
                if os.path.exists(selected_sample_path):
                    uploaded_file = open(selected_sample_path, "rb")
        
        # Process the image if uploaded
        if uploaded_file is not None:
            try:
                # Read the image
                file_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                
                # Display original image
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Angiogram", use_container_width=True)
                
                # Convert to numpy for YOLOv8
                image_np = np.array(image)
                
                # Run stenosis detection
                st.subheader("Stenosis Detection")
                detection_results, annotated_image = detect_stenosis(yolo_model, image_np)
                
                if detection_results is not None:
                    # Display annotated image
                    st.image(annotated_image, caption="Detected Stenosis Regions", use_container_width=True)
                    
                    # Show detection details
                    num_detections = len(detection_results.boxes)
                    st.write(f"Found {num_detections} potential stenosis regions")
                    
                    if num_detections > 0:
                        # Create table of detections
                        detections_data = []
                        for i, box in enumerate(detection_results.boxes):
                            confidence = box.conf.item()
                            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                            cls = int(box.cls.item())
                            detections_data.append({
                                'Detection #': i+1,
                                'Confidence': confidence,
                                'Box Coordinates': f"({xmin:.1f}, {ymin:.1f}), ({xmax:.1f}, {ymax:.1f})"
                            })
                        
                        st.table(pd.DataFrame(detections_data).style.format({'Confidence': '{:.2%}'}))
                    else:
                        st.info("No stenosis detected in this image")
                else:
                    st.error("Detection model failed to process the image")
                
                # Run classification
                st.subheader("Artery Classification")
                
                if any(classification_models.values()):
                    classification_results = classify_stenosis(classification_models, image, devices)
                    
                    if classification_results:
                        display_model_predictions(classification_results)
                        
                        # XAI Analysis
                        st.markdown("<h2 class='section-header'>🔬 Explainable AI Analysis</h2>", unsafe_allow_html=True)
                        
                        # Choose a model for XAI
                        xai_model_options = [name for name, model in classification_models.items() if model is not None]
                        
                        if xai_model_options:
                            selected_xai_model = st.selectbox(
                                "Select model for XAI analysis",
                                xai_model_options
                            )
                            
                            # Apply XAI
                            if selected_xai_model in classification_models:
                                model = classification_models[selected_xai_model]
                                device = devices[selected_xai_model]
                                
                                # Get prediction
                                prediction = classification_results[selected_xai_model]
                                predicted_class_idx = prediction['class_idx']
                                
                                # Preprocess image for XAI
                                preprocess = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                                
                                image_tensor = preprocess(image).unsqueeze(0)
                                
                                # Apply XAI
                                apply_xai_to_image(model, image_tensor, device, CLASS_NAMES, predicted_class_idx)
                    else:
                        st.error("Classification failed")
                else:
                    st.error("No classification models available")
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)
    
    # For the Dataset Analysis (EDA) mode:
    elif app_mode == "📊 Dataset Analysis (EDA)":
        st.markdown("<h2 class='section-header'>📊 Dataset Analysis and Visualization</h2>", unsafe_allow_html=True)
        
        # Provide option to either upload files or use existing dataset path
        analysis_source = st.radio(
            "Choose data source:",
            ["Upload Files", "Use Dataset Path"]
        )
        
        if analysis_source == "Upload Files":
            # Allow multiple file upload
            uploaded_files = st.file_uploader(
                "Upload dataset images (JPG, PNG, etc.)", 
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True, 
                key="eda_upload"
            )
            
            if uploaded_files:
                # Display file count and proceed button
                st.write(f"Uploaded {len(uploaded_files)} files")
                
                if st.button("Analyze Uploaded Dataset"):
                    with st.spinner("Analyzing uploaded images..."):
                        display_eda_from_upload(uploaded_files)
            else:
                st.info("Please upload images to analyze. You can select multiple files at once.")
                
        else:  # Use Dataset Path
            # Allow user to specify dataset path
            custom_dataset_path = st.text_input(
                "Dataset directory path", 
                value=DEFAULT_DATASET_PATH
            )
            
            if st.button("Analyze Dataset Directory"):
                if os.path.exists(custom_dataset_path):
                    with st.spinner("Analyzing dataset directory..."):
                        display_eda(custom_dataset_path)
                else:
                    st.error(f"Directory not found: {custom_dataset_path}")
                    st.info("Please provide a valid directory path containing image files.")

    # For the Batch Analysis mode:
    elif app_mode == "🔄 Batch Analysis":
        # Load models
        load_all_models()
        
        st.markdown("<h2 class='section-header'>🔄 Batch Analysis</h2>", unsafe_allow_html=True)
        
        st.write("""
        This mode analyzes multiple images to evaluate model performance.
        Upload multiple images below or use the sample dataset.
        """)
        
        # Provide option to either upload files or use existing dataset
        batch_source = st.radio(
            "Choose data source:",
            ["Upload Files", "Use Sample Dataset"]
        )
        
        if batch_source == "Upload Files":
            # File upload for batch analysis
            batch_files = st.file_uploader(
                "Upload images for batch analysis", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                key="batch_upload"
            )
            
            # Run batch analysis on button click
            if batch_files:
                st.write(f"Uploaded {len(batch_files)} files")
                
                if st.button("Run Batch Analysis"):
                    with st.spinner("Running batch analysis..."):
                        run_batch_analysis(yolo_model, classification_models, devices, batch_files)
            else:
                st.info("Please upload images to run batch analysis. You can select multiple files at once.")
        
        else:  # Use Sample Dataset
            # Allow user to specify dataset path
            custom_dataset_path = st.text_input(
                "Dataset directory path", 
                value=DEFAULT_DATASET_PATH
            )
            
            sample_count = st.slider("Number of images per class to analyze", 1, 20, 5)
            
            if st.button("Run Batch Analysis on Sample Dataset"):
                if os.path.exists(custom_dataset_path):
                    with st.spinner(f"Analyzing {sample_count} images per class from dataset..."):
                        run_batch_analysis_from_path(yolo_model, classification_models, devices, custom_dataset_path, sample_count)
                else:
                    st.error(f"Directory not found: {custom_dataset_path}")
                    st.info("Please provide a valid directory path containing image files.")

    elif app_mode == "ℹ️ About":
        show_about()
    
# Add this helper function for running batch analysis from a directory path
def run_batch_analysis_from_path(yolo_model, classification_models, devices, dataset_path, sample_count=5):
    """Run batch analysis on images from the dataset directory."""
    stats = scan_dataset(dataset_path)
    
    # Create a list of file paths to analyze
    batch_files = []
    
    # For each class, get the specified number of samples
    for class_name in CLASS_NAMES:
        if class_name in stats['class_paths'] and stats['class_paths'][class_name]:
            # Get random samples
            paths = stats['class_paths'][class_name]
            selected_paths = random.sample(paths, min(sample_count, len(paths)))
            
            # Open each image and add to batch files
            for path in selected_paths:
                try:
                    with open(path, 'rb') as f:
                        # Create a BytesIO object
                        file_bytes = io.BytesIO(f.read())
                        file_bytes.name = os.path.basename(path)
                        batch_files.append(file_bytes)
                except Exception as e:
                    st.warning(f"Could not read file {path}: {e}")
    
    if batch_files:
        # Run the batch analysis
        run_batch_analysis(yolo_model, classification_models, devices, batch_files)
    else:
        st.error("No valid images found in the dataset directory.")

if __name__ == "__main__":
    main()
