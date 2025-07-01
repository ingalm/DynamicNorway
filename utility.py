import numpy as np
import torch
from collections import Counter
import matplotlib.pyplot as plt
import os
from architectures.CustomUNet import CustomUNet
from architectures.DynamicNorway import DynamicNorway
from constants import NUM_CLASSES, grunn_class_mapping
from matplotlib import colors

# Script containing utility functions for model training and evaluation.

def calculate_class_weights_and_count(loader):
    class_counts = Counter()

    for _, masks in loader:
        class_counts.update(masks.view(-1).cpu().numpy())
        
    # Remove Unclassified class
    if 99 in class_counts:
        del class_counts[99]
        
    class_counts = dict(sorted(class_counts.items()))

    total_pixels = sum(class_counts.values())

    class_weights = {key: total_pixels / (len(class_counts) * count) for key, count in class_counts.items()}

    return class_counts, class_weights

def calculate_class_weights_and_count_with_log(loader, epsilon=1.1):
    class_counts = Counter()

    for _, masks in loader:
        flat = masks.view(-1).cpu().numpy()
        flat = flat[flat != 99]  # Exclude ignore index
        class_counts.update(flat.tolist())

    class_counts = dict(sorted(class_counts.items()))
    total_pixels = sum(class_counts.values())

    # Compute class frequencies
    class_freqs = {cls: count / total_pixels for cls, count in class_counts.items()}

    # Compute log-scaled weights
    class_weights = {
        cls: 1.0 / np.log(freq + epsilon)
        for cls, freq in class_freqs.items()
    }

    # Normalize weights
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    return class_counts, class_weights


def calculate_class_weights_and_count_with_median_frequency(loader):
    class_counts = Counter()

    # Count pixel occurrences per class
    for _, masks in loader:
        flat = masks.view(-1).cpu().numpy()
        flat = flat[flat != 99]  # Exclude ignore label
        class_counts.update(flat.tolist())

    class_counts = dict(sorted(class_counts.items()))
    total_pixels = sum(class_counts.values())

    # Compute frequency per class
    class_freqs = {cls: count / total_pixels for cls, count in class_counts.items()}

    # Median frequency
    median_freq = np.median(list(class_freqs.values()))

    # Compute median frequency-balanced weights
    class_weights = {
        cls: median_freq / freq
        for cls, freq in class_freqs.items()
    }

    return class_counts, class_weights


def get_class_weights(loader, device):  
    # Uncomment one of the following lines to choose the method for calculating class weights
    
    # class_counts, class_weights = calculate_class_weights_and_count(loader)
    # class_counts, class_weights = calculate_class_weights_and_count_with_median_frequency(loader)
    class_counts, class_weights = calculate_class_weights_and_count_with_log(loader)

    class_weights_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(NUM_CLASSES)], dtype=torch.float32)    
    class_weights_tensor = class_weights_tensor.to(device)

    print("\n--- Class Counts and Weights ---")
    for i in range(NUM_CLASSES):
        class_name = grunn_class_mapping.get(i, f"Class {i}")
        count = class_counts.get(i, 0)
        weight = class_weights.get(i, 1.0)
        print(f"{i:2d}: {class_name:20s} | Count: {count:7d} | Weight: {weight:.4f}")

    print(f"\nClass weights tensor: {class_weights_tensor}")
    
    return class_weights_tensor

# Weights used to emphasize the 'Built' class in the dataset.
def get_custom_built_class_weight(device, built_class_index=6, built_weight=2.0):
    weights = [1.0] * NUM_CLASSES
    weights[built_class_index] = built_weight

    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\nCustom weights with emphasis on 'Built' class (index {built_class_index}): {weight_tensor}")
    return weight_tensor

PREDICTIONS_DIR = "predictions"
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)

# Function to visualize model predictions and save them to a directory. Used during preliminary evaluation of models.
def visualize_predictions(model, model_name, loader, num_images=4, c_map=None, label_type="grunnkartLabel"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    images_shown = 0
    
    model_name_without_extension = model_name.split(".")[0]
    
    model_predictions_dir = os.path.join(PREDICTIONS_DIR, model_name_without_extension)
    os.makedirs(model_predictions_dir, exist_ok=True)
    
    images_shown = 0
    timestep = 0 # Used to ensure diversity in images shown, due to patches coming from the same image. Skipping every 5th image
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(images.shape[0]):
                if images_shown >= num_images:
                    return
                
                # Skip every 5th image
                if timestep % 5 != 0:
                    timestep += 1
                    continue
                timestep += 1
                    
                
                image_np = images[i].cpu().numpy()

                # Normalize Image for Proper Display
                image_rgb = image_np[[2, 1, 0]].transpose(1, 2, 0) 
                image_rgb = (image_rgb - np.percentile(image_rgb, 2)) / (
                    np.percentile(image_rgb, 98) - np.percentile(image_rgb, 2)
                ) 
                image_rgb = np.clip(image_rgb, 0, 1)
                
                if c_map == "gray":
                    gt_cmap = c_map
                else:
                    if label_type == "dwLabel":
                        gt_colors = [
                            'blue',         # Water
                            'green',         # Trees
                            'lightgreen',        # Grass
                            'darkblue',     # Flooded vegetation
                            'yellow',    # Crops
                            'darkgreen',    # Shrub and scrub
                            'red',      # Built
                            'grey',      # Bare
                            'lightblue',         # Snow and Ice
                            'white',          # Uklassifisert
                        ]

                        gt_cmap = colors.ListedColormap(gt_colors)   
                    elif label_type == "grunnkartLabel":
                        gt_colors = [
                            'blue',         # Water
                            'green',         # Trees
                            'lightgreen',        # Grass
                            'darkblue',     # Flooded vegetation
                            'yellow',    # Crops
                            'darkgreen',    # Shrub and scrub
                            'red',      # Built
                            'grey',      # Bare
                            'white',          # Uklassifisert
                        ]
                        
                        gt_cmap = colors.ListedColormap(gt_colors)

                    
                
                mask_np = masks[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(image_rgb)
                axes[0].set_title("Satellite Image (RGB)")
                axes[1].imshow(mask_np, cmap=gt_cmap, vmin=0, vmax=NUM_CLASSES)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pred_np, cmap=gt_cmap, vmin=0, vmax=NUM_CLASSES)
                axes[2].set_title("Model Prediction")
                
                for ax in axes:
                    ax.axis("off")
                
                plt.savefig(os.path.join(model_predictions_dir, f'prediction_{images_shown}.png'))
                print(f"Saved prediction_{images_shown}.png")
                plt.close()
                
                images_shown += 1
                
model_dir = "models/"

# Function to load a model based on its name. Used during training and evaluation.
def load_model(model_name, device):
    if "DynamicNorway" in model_name:
        model = DynamicNorway(b=2, m=1.5, num_classes=NUM_CLASSES)
    elif "UNet" in model_name:
        model = CustomUNet(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Model {model_name} not recognized")
        
        
    state_dict = torch.load(f"{model_dir}{model_name}", map_location=device)

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"Model {model_name} loaded successfully")
    
    return model

def filter_ignore_label(preds, targets, ignore_label=99):
    valid_mask = targets != ignore_label
    return preds[valid_mask], targets[valid_mask]
