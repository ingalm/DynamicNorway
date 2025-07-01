import random
import numpy as np
import time
import os
import torch
import torch.optim as optim
from data_processing.preprocessing import create_dataloaders
import torch.nn as nn
from architectures.CustomUNet import CustomUNet
from utility import load_model, filter_ignore_label
from evaluation import compute_overall_accuracy, compute_iou, compute_mean_iou
from torch.utils.tensorboard import SummaryWriter
from architectures.DynamicNorway import DynamicNorway
from constants import NUM_CLASSES, EPOCHS, BATCH_SIZE, PATCH_SHAPE, DATA_DIR, grunn_class_mapping, dw_class_mapping, MODEL_NAME, LABEL
from sklearn.metrics import precision_score, recall_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###
# Command line tool to view the logs
# tensorboard --logdir=logs
###

label_type = LABEL
if label_type == "grunnkartLabel":
    class_mapping = grunn_class_mapping
elif label_type == "dwLabel":
    class_mapping = dw_class_mapping
else:
    raise ValueError(f"Unknown label type: {label_type}")

train_loader, val_loader, test_loader = create_dataloaders(
    DATA_DIR,
    batch_size=BATCH_SIZE,
    patch_shape=PATCH_SHAPE,
    label_type=label_type,
    transform_train=True 
)

def get_loss_function(class_weights=None):
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=99)
    else:
        return nn.CrossEntropyLoss(ignore_index=99)

def get_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr)

def train_model(model_name, model, train_loader, val_loader, class_mapping, num_epochs=EPOCHS, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    optimizer = get_optimizer(model, lr)
    
    # Uncomment one of the following lines to choose the method for calculating class weights
    # criterion = get_loss_function(class_weights=get_class_weights(train_loader, device))
    criterion = get_loss_function()
    # criterion = get_loss_function(class_weights=get_custom_built_class_weight(device))

    # Set up logging (unique log dir for each model)
    log_dir = f"logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_iou, running_accuracy = 0.0, [], 0.0
        start_time = time.time()

        for batch in train_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            
            # Check for invalid labels in mask
            assert masks.max() < NUM_CLASSES or masks[masks >= NUM_CLASSES].unique().tolist() == [99], \
                f"Invalid label in mask: {masks.unique().tolist()}"
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 

            preds = torch.argmax(outputs, dim=1)
            iou_per_class = compute_iou(preds.cpu().numpy(), masks.cpu().numpy(), class_mapping)
            mean_iou = compute_mean_iou(iou_per_class)
            accuracy = compute_overall_accuracy(preds.cpu().numpy(), masks.cpu().numpy())

            running_iou.append(mean_iou)
            running_accuracy += accuracy

        avg_train_loss = running_loss / len(train_loader)
        avg_train_iou = sum(running_iou) / len(running_iou)
        avg_train_acc = running_accuracy / len(train_loader)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("IoU/Train", avg_train_iou, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)

        epoch_time = time.time() - start_time

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f'[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Mean IoU: {avg_train_iou:.4f}, Accuracy: {avg_train_acc:.2f}%, Time: {epoch_time:.2f}s', flush=True)

        model.eval()
        val_loss, val_iou, val_accuracy = 0.0, [], 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                iou_per_class = compute_iou(preds.cpu().numpy(), masks.cpu().numpy(), class_mapping)
                mean_iou = compute_mean_iou(iou_per_class)
                accuracy = compute_overall_accuracy(preds.cpu().numpy(), masks.cpu().numpy())

                val_iou.append(mean_iou)
                val_accuracy += accuracy

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = sum(val_iou) / len(val_iou)
        avg_val_acc = val_accuracy / len(val_loader)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("IoU/Validation", avg_val_iou, epoch)
        writer.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f'models/{model_name}_best_model.pth')
            print(f"Best model saved at epoch {epoch+1}", flush=True)

    # Close TensorBoard writer
    writer.close()


def evaluate_model(model_name, test_loader, class_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(f"{model_name}_best_model.pth", device)
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    all_ious = []
    all_preds = []
    all_targets = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == masks).sum().item()
            total += masks.numel()

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())

            iou_per_class = compute_iou(preds.cpu().numpy(), masks.cpu().numpy(), class_mapping)
            mean_iou = compute_mean_iou(iou_per_class)
            all_ious.append(mean_iou)

            if batch_idx % 20 == 0:
                batch_acc = (correct / total) * 100
                print(f"[Batch {batch_idx+1}/{len(test_loader)}] Accuracy: {batch_acc:.2f}%", flush=True)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    valid_preds, valid_targets = filter_ignore_label(all_preds, all_targets)

    accuracy = (valid_preds == valid_targets).sum() / len(valid_targets) * 100
    precision = precision_score(valid_targets, valid_preds, average='macro', zero_division=0)
    recall = recall_score(valid_targets, valid_preds, average='macro', zero_division=0)
    f1 = f1_score(valid_targets, valid_preds, average='macro', zero_division=0)

    mean_iou = sum(all_ious) / len(all_ious)
    per_class_iou = compute_iou(valid_preds.reshape(1, -1), valid_targets.reshape(1, -1), class_mapping)

    eval_time = time.time() - start_time

    print(f'\nTest Accuracy: {accuracy:.2f}%')
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro): {recall:.4f}')
    print(f'F1 Score (macro): {f1:.4f}')
    print(f'Evaluation Time: {eval_time:.2f}s\n')

    print("Per-Class IoU:")
    for entry in per_class_iou:
        print(f"{entry['Class Name']}: {entry['IoU Score']:.4f}")

    # Save results to a text file
    os.makedirs("logs", exist_ok=True)
    results_path = os.path.join("logs", f"{model_name}_test_metrics.txt")
    with open(results_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Precision (macro): {precision:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")
        f.write(f"F1 Score (macro): {f1:.4f}\n\n")
        f.write("Per-Class IoU:\n")
        for entry in per_class_iou:
            f.write(f"{entry['Class Name']}: {entry['IoU Score']:.4f}\n")
        

# Define models to test
models_to_test = [
    (MODEL_NAME, DynamicNorway(num_classes=NUM_CLASSES))
]

# Train and evaluate each model
for model_name, model in models_to_test:
    print(f'Training {model_name}...', flush=True)
    train_model(model_name, model, train_loader, val_loader, class_mapping=class_mapping)
    print(f'Evaluating {model_name}...', flush=True)
    evaluate_model(model_name, test_loader, class_mapping=class_mapping)


