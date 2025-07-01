import torch
import os
import pandas as pd
import numpy as np 
from data_processing.preprocessing import create_dataloaders 
from utility import load_model 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from constants import PATCH_SHAPE, BATCH_SIZE, DATA_DIR, dw_class_mapping, grunn_class_mapping, LABEL

# Some of this script is outdated, but the metric functions are still used.

# Constants to control the evaluation process
EVALUATE_MODELS = False # True if you want to evaluate all models in the `models/` folder
if LABEL == "grunnkartLabel":
    class_mapping = grunn_class_mapping
elif LABEL == "dwLabel":
    class_mapping = dw_class_mapping

results_csv = "results/model_evaluation_results.csv"
os.makedirs("results", exist_ok=True) 

def compute_overall_accuracy(preds, masks):
    correct = (preds == masks).sum().item()
    total = masks.size if isinstance(masks, np.ndarray) else masks.numel()
    return (correct / total) * 100 if total > 0 else 0


def compute_iou(preds, masks, class_mapping):
    results = []

    for class_id in class_mapping.keys():
        if class_id == 99:
            continue
        intersection = ((preds == class_id) & (masks == class_id)).sum().item()
        union = ((preds == class_id) | (masks == class_id)).sum().item()
        iou_score = intersection / union if union > 0 else float("nan")

        results.append({
            "Class ID": class_id,
            "Class Name": class_mapping.get(class_id, "Unknown"),
            "IoU Score": iou_score
        })

    return results

def compute_mean_iou(iou_per_class):
    iou_scores = [entry["IoU Score"] for entry in iou_per_class if not np.isnan(entry["IoU Score"])]
    
    if len(iou_scores) == 0:
        return 0  
    
    return np.mean(iou_scores) 

def precision_recall_f1(preds, masks, ignore_class=99):
    y_true_all = []
    y_pred_all = []

    for y_true, y_pred in zip(masks, preds):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Mask out the ignore_class from both y_true and y_pred
        valid_mask = (y_true_flat != ignore_class) & (y_pred_flat != ignore_class)
        y_true_flat = y_true_flat[valid_mask]
        y_pred_flat = y_pred_flat[valid_mask]

        y_true_all.extend(y_true_flat)
        y_pred_all.extend(y_pred_flat)

    precision = precision_score(y_true_all, y_pred_all, average='macro', pos_label=1)
    recall = recall_score(y_true_all, y_pred_all, average='macro', pos_label=1)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', pos_label=1)

    return precision, recall, f1
     

def get_model_names():
    model_dir = "models/"
    models = []

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist!")

    for model_name in os.listdir(model_dir):
        if model_name.endswith(".pth"):  # Ensure only .pth files are included
            models.append((model_name, os.path.join(model_dir, model_name))) 

    return models  


# Evaluate all models in `models/` folder and save results to CSV.
def evaluate_models(class_mapping):
    _, _, test_loader = create_dataloaders(DATA_DIR, BATCH_SIZE, PATCH_SHAPE, label_type=LABEL)

    unique_values = set()
    for batch in test_loader:
        images, masks = batch
        unique_values.update(masks.unique().tolist())
        break

    print("Unique values in masks:", unique_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_csv = "results/model_evaluation_results.csv"
    os.makedirs("results", exist_ok=True)

    # Load existing results (to avoid re-evaluating models)
    if os.path.exists(results_csv):
        existing_results = pd.read_csv(results_csv)

        if existing_results.empty or "Model Name" not in existing_results:
            evaluated_models = set()
        else:
            existing_results["Model Name"] = existing_results["Model Name"].astype(str).str.strip("[]").replace("'", "")
            evaluated_models = set(existing_results["Model Name"])
    else:
        existing_results = pd.DataFrame(columns=["Model Name","Test Accuracy (%)","Mean IoU (%)","Precision","Recall","F1","IoU per Class"])
        evaluated_models = set()


    models_to_evaluate = get_model_names()
    results = []

    for model_name, model_path in models_to_evaluate:
        torch.cuda.empty_cache()  # Free up memory
        if model_name in evaluated_models:
            print(f"Model `{model_name}` has already been evaluated. Skipping...")
            continue  

        print(f"Loading `{model_name}`...")
        model = load_model(model_name, device)  
        model = model.to(device)
        model.eval()

        all_preds, all_masks = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_masks.append(masks.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        accuracy = compute_overall_accuracy(all_preds, all_masks)
        iou_per_class = compute_iou(all_preds, all_masks, class_mapping)
        mean_iou = compute_mean_iou(iou_per_class)
        precision, recall, f1 = precision_recall_f1(all_preds, all_masks)

        print(f"`{model_name}` - Accuracy: {accuracy:.2f}% | mIoU: {mean_iou:.2f}%")

        iou_per_class_formatted = [(entry["Class ID"], entry["IoU Score"]) for entry in iou_per_class]

        results = [[model_name, round(accuracy, 2), round(mean_iou, 2), round(precision, 2), round(recall, 2), round(f1, 2), iou_per_class_formatted]]
        results_df = pd.DataFrame(results, columns=["Model Name","Test Accuracy (%)","Mean IoU (%)","Precision","Recall","F1","IoU per Class"])
        updated_results_df = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results_df.to_csv(results_csv, index=False)
        existing_results = updated_results_df
    
def plot_training_history(train_losses, val_losses, train_mean_ious, val_mean_ious, train_accs, val_accs, model_name):
    save_path = f"models/{model_name}.png"
    os.makedirs("models", exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    # Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
 
    # Mean IoU Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_mean_ious, label="Training Mean IoU", marker="o")
    plt.plot(epochs, val_mean_ious, label="Validation Mean IoU", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Mean IoU")
    plt.title("Training & Validation Mean IoU")
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_accs, label="Training Accuracy", marker="o")
    plt.plot(epochs, val_accs, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    plt.show()
    
if EVALUATE_MODELS:
    # Only run this if you want to evaluate the models
    print("Evaluating models...")
    evaluate_models(class_mapping)
