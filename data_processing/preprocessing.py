import os
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader, random_split
import random
import torchvision.transforms.functional as TF
from constants import TRAIN_RATIO, VAL_RATIO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ImageDataset(Dataset):

    S2_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9] # Sentinel-2 bands to use

    def __init__(self, data_dir, patch_shape, label_type="grunnkartLabel", transform=None):
        self.data_dir = data_dir
        self.patch_size = patch_shape
        self.transform = transform
        self.label_type = label_type 

        # Get all TIFF files in the directory
        self.image_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.tif') and not f.startswith('.')
        ])

        if not self.image_files:
            raise ValueError(f"No TIFF files found in {data_dir}")

        # Generate patches
        self.patches = self.generate_patches()
    

    def generate_patches(self):
        patches = []

        for image_path in self.image_files:
            with rasterio.open(image_path) as src:
                image = np.stack([src.read(b) for b in self.S2_BAND_INDICES], axis=0)

                # Read the ground truth mask
                if self.label_type == "dwLabel":
                    mask = src.read(10)
                    # Account for unclassified pixels. 
                    # Some "ghost" pixels have the value 10, which is outside the range of the mask. 
                    # These will be counted as unclassified.
                    mask[mask == 0] = 100
                    mask[mask == 10] = 100
                    mask = mask - 1
                elif self.label_type == "grunnkartLabel":
                    # For the NIBIO_s2_summer_2024 dataset theh grunnkart label is in band 10, the other datasets has grunnkartlabel in band 11
                    mask = src.read(11) 
                    mask[mask == 0] = 100
                    mask = mask - 1

                else:
                    raise ValueError(f"Invalid label_type '{self.label_type}'. Choose 'dwLabel' or 'grunnkartLabel'.")

            image = torch.tensor(image.astype(np.float32), dtype=torch.float32)
            mask = torch.tensor(mask.astype(np.int64), dtype=torch.long)

            # Ensure image and mask have matching spatial dimensions
            assert image.shape[1:] == mask.shape, f"Size mismatch: {image.shape} vs {mask.shape}"

            # Generate patches
            img_patches, mask_patches = self._split_into_patches(image, mask)
            for img_patch, mask_patch in zip(img_patches, mask_patches):
                patches.append((img_patch, mask_patch))

        return patches


    # Split the images into patches of size patch_size
    def _split_into_patches(self, image, mask):
        _, img_h, img_w = image.shape  
        patch_h, patch_w = self.patch_size  

        img_patches = []
        mask_patches = []

        for y in range(0, img_h, patch_h):
            for x in range(0, img_w, patch_w):
                if y + patch_h > img_h or x + patch_w > img_w:
                    continue

                img_patch = image[:, y:y+patch_h, x:x+patch_w]  
                mask_patch = mask[y:y+patch_h, x:x+patch_w]

                img_patches.append(img_patch)
                mask_patches.append(mask_patch)

        return img_patches, mask_patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image_patch, mask_patch = self.patches[idx]

        if self.transform:
            image_patch, mask_patch = self.transform(image_patch, mask_patch)

        return image_patch, mask_patch

class DeterministicAugmenter:
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, image, mask):
        return random_flip_and_rotation(image, mask, generator=self.generator)

# Deterministic random flip and rotation using torch.Generator.
def random_flip_and_rotation(image, mask, generator=None):
    if generator is None:
        generator = torch.Generator().manual_seed(SEED)  # fallback to global seed if not passed

    # Horizontal flip
    if torch.rand(1, generator=generator).item() > 0.5:
        image = torch.flip(image, [2])
        mask = torch.flip(mask, [1])

    # Vertical flip
    if torch.rand(1, generator=generator).item() > 0.5:
        image = torch.flip(image, [1])
        mask = torch.flip(mask, [0])

    # Rotation
    angle = int(torch.randint(0, 4, (1,), generator=generator).item()) * 90

    if image.dim() == 3:
        image = TF.rotate(image, angle)
    else:
        raise ValueError(f"Expected image tensor shape [C, H, W], got: {image.shape}")

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
        mask = TF.rotate(mask, angle)
        mask = mask.squeeze(0)
    else:
        raise ValueError(f"Expected mask tensor shape [H, W], got: {mask.shape}")

    return image, mask


def create_dataloaders(data_dir, batch_size, patch_shape, label_type="grunnkartLabel", transform_train=True, split_file="splits.npz"):
    split_path = os.path.join(data_dir, split_file)

    base_dataset = ImageDataset(
        data_dir=data_dir, 
        patch_shape=patch_shape, 
        label_type=label_type,
        transform=DeterministicAugmenter(SEED) if transform_train else None
    )

    if os.path.exists(split_path):
        splits = np.load(split_path)
        train_indices = splits['train']
        val_indices = splits['val']
        test_indices = splits['test']
    else:
        dataset_size = len(base_dataset)
        train_size = int(TRAIN_RATIO * dataset_size)
        val_size = int(VAL_RATIO * dataset_size)
        test_size = dataset_size - train_size - val_size

        generator = torch.Generator().manual_seed(SEED)
        train_subset, val_subset, test_subset = random_split(
            base_dataset, [train_size, val_size, test_size], generator=generator
        )

        train_indices = train_subset.indices
        val_indices = val_subset.indices
        test_indices = test_subset.indices

        np.savez(split_path, train=train_indices, val=val_indices, test=test_indices)

    # For val and test: use dataset without transform
    clean_dataset = ImageDataset(
        data_dir=data_dir,
        patch_shape=patch_shape,
        label_type=label_type,
        transform=None
    )

    train_set = torch.utils.data.Subset(base_dataset, train_indices)
    val_set = torch.utils.data.Subset(clean_dataset, val_indices)
    test_set = torch.utils.data.Subset(clean_dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
