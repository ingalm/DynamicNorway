from matplotlib import pyplot as plt
from constants import DATA_DIR, BATCH_SIZE, PATCH_SHAPE, grunn_class_mapping, dw_class_mapping
from data_processing.preprocessing import create_dataloaders


train_loader_grunn, val_loader_grunn, test_loader_grunn = create_dataloaders(DATA_DIR, BATCH_SIZE, PATCH_SHAPE, label_type='grunnkartLabel')
train_loader_dw, val_loader_dw, test_loader_dw = create_dataloaders(DATA_DIR, BATCH_SIZE, PATCH_SHAPE, label_type='dwLabel')


# compute normalized class distribution
def class_distribution(loaders, class_mapping):
    class_counts = {class_id: 0 for class_id in class_mapping.keys()}
    total_pixels = 0

    for loader in loaders:
        for batch in loader:
            _, masks = batch
            for class_id in class_mapping.keys():
                class_counts[class_id] += (masks == class_id).sum().item()
            total_pixels += masks.numel()

    return {class_id: count / total_pixels for class_id, count in class_counts.items()}

grunn_loaders = [train_loader_grunn, val_loader_grunn, test_loader_grunn]
dw_loaders = [train_loader_dw, val_loader_dw, test_loader_dw]

# Get distributions
grunn_dist = class_distribution(grunn_loaders, grunn_class_mapping)
dw_dist = class_distribution(dw_loaders, dw_class_mapping)

all_class_ids = sorted(set(grunn_class_mapping.keys()).union(dw_class_mapping.keys()))
all_labels = [grunn_class_mapping.get(class_id) or dw_class_mapping.get(class_id) for class_id in all_class_ids]

# Build aligned value lists
grunn_values = [grunn_dist.get(class_id, 0) for class_id in all_class_ids]
dw_values = [dw_dist.get(class_id, 0) for class_id in all_class_ids]

# Plot
def save_comparison_plot(class_ids, labels, grunn_values, dw_values):
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], grunn_values, width=width, label='Grunnkart', color='royalblue')
    plt.bar([i + width/2 for i in x], dw_values, width=width, label='Dynamic World', color='orange')

    plt.xlabel('Class')
    plt.ylabel('Proportion of Pixels')
    plt.title('Class Distribution Comparison: Grunnkart vs Dynamic World')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/class_distribution_comparison_whole_dataset.png')
    plt.show()

# Save the plot
# save_comparison_plot(all_class_ids, all_labels, grunn_values, dw_values)


# DW-training set class counts calculated on google earth engine
# These counts are based on the complete Dynamic World training dataset, not just the subset from Norway.
complete_dw_class_counts = {
    0: 423216643, # water
    1: 1916992784, # trees
    2: 115735650, # grass
    3: 120387253, #flooded vegetation
    4: 635436788, # crops
    5: 1681675702, # shrub_and_scrub
    6: 232655591, # built
    7: 169496667, # bare
    8: 146607591, # snow and ice
    # 10: 24195641 # clouds

}

total_complete_dw_pixels = sum(complete_dw_class_counts.values())
# Normalize the counts to proportions
complete_dw_class_counts = {k: v / total_complete_dw_pixels for k, v in complete_dw_class_counts.items()}

complete_dw_values = list(complete_dw_class_counts.values())

# Get distributions

# Plot
def save_dw_comparison_plot(class_ids, labels, complete_dw_values, dw_values):
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], complete_dw_values, width=width, label='Complete DW Training Data', color='royalblue')
    plt.bar([i + width/2 for i in x], dw_values, width=width, label='DW Training Data from Norway', color='orange')

    plt.xlabel('Class')
    plt.ylabel('Proportion of Pixels')
    plt.title('Class Distribution Comparison: Complete DW vs Subset from Norway')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/class_distribution_comparison_CompleteDW_vs_Norway_subset.png')
    plt.show()

save_dw_comparison_plot(all_class_ids, all_labels, complete_dw_values, dw_values)