import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, ScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandFlipd, RandRotate90d, Resized, ToTensord
)

class Nifti2DSliceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, slice_axis=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.slice_axis = slice_axis

        # Match image and mask by basename
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
        self.pairs = []
        for f in self.image_files:
            base = f.replace(".nii.gz", "")
            mask_f = f"{base}_gt.nii.gz"
            if os.path.exists(os.path.join(mask_dir, mask_f)):
                self.pairs.append((f, mask_f))
            else:
                print(f"⚠️ Missing mask for {f}, skipping.")

        # Precompute (image, mask, slice_idx) map
        self.slice_map = []
        for img_f, mask_f in self.pairs:
            img = nib.load(os.path.join(image_dir, img_f)).get_fdata()
            num_slices = img.shape[self.slice_axis]
            for s in range(num_slices):
                self.slice_map.append((img_f, mask_f, s))

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        img_f, mask_f, slice_idx = self.slice_map[idx]
        img = nib.load(os.path.join(self.image_dir, img_f)).get_fdata().astype(np.float32)
        mask = nib.load(os.path.join(self.mask_dir, mask_f)).get_fdata().astype(np.int64)

        img_2d = np.take(img, slice_idx, axis=self.slice_axis)
        mask_2d = np.take(mask, slice_idx, axis=self.slice_axis)

        img_2d = np.expand_dims(img_2d, axis=0)   # (1, H, W)
        mask_2d = np.expand_dims(mask_2d, axis=0) # (1, H, W)

        sample = {"image": img_2d, "mask": mask_2d, "imagefilename": img_f}
        if self.transform:
            sample = self.transform(sample)
        return sample["image"], sample["mask"], sample["imagefilename"]

def get_train_transforms(img_size=224):
    return Compose([
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="area"),
        Resized(keys=["mask"], spatial_size=(img_size, img_size), mode="nearest"),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
        RandGaussianNoised(keys=["image"], mean=0.0, std=0.05, prob=0.3),
        ToTensord(keys=["image", "mask"])
    ])

def get_val_transforms(img_size=224):
    return Compose([
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="area"),
        Resized(keys=["mask"], spatial_size=(img_size, img_size), mode="nearest"),
        ToTensord(keys=["image", "mask"])
    ])