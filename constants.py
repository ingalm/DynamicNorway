NUM_CLASSES = 8
EPOCHS = 300
BATCH_SIZE = 16
PATCH_SHAPE = (170, 170)
LABEL = "grunnkartLabel"  # Change to "dwLabel" for the other dataset
DATA_DIR = "data/NIBIO_s2_summer_2024"  # Change to the appropriate dataset directory
dataset_name = DATA_DIR.split("/")[-1]

MODEL_NAME = f"UNet{LABEL}_dataset{dataset_name}_ps{PATCH_SHAPE[0]}_e{EPOCHS}_bs{BATCH_SIZE}"

TRAIN_IMAGE_DIR = 'data/split_images/tra_scene'
TRAIN_MASK_DIR = 'data/split_masks/tra_truth'
VAL_IMAGE_DIR = 'data/split_images/val_scene'
VAL_MASK_DIR = 'data/split_masks/val_truth'
TEST_IMAGE_DIR = 'data/split_images/test_scene'
TEST_MASK_DIR = 'data/split_masks/test_truth'

# Train-validation-test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

grunn_class_mapping = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Shrub and scrub",
    6: "Built",
    7: "Bare",
    99: "Unclassified",
}


dw_class_mapping = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Shrub and scrub",
    6: "Built",
    7: "Bare",
    8: "Snow and Ice",
    99: "Unclassified",
}