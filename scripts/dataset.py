# dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CurveDataset(Dataset):
    """
    Loads an image and its corresponding mask (curve).
    image_dir: Path to folder with images
    mask_dir:  Path to folder with binary masks
    list_of_images: list of image filenames (e.g. ["IMG_001.jpg", "IMG_002.jpg", ...])
    """
    def __init__(self, list_of_images, image_dir, mask_dir, img_size=(512,512), transform=None):
        self.image_names = list_of_images
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        filename = self.image_names[idx]
        img_path = os.path.join(self.image_dir, filename)

        # Construct mask filename, e.g. "IMG_001_mask.png"
        basename, _ = os.path.splitext(filename)
        mask_filename = basename + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load image
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image {img_path}")

        # Load mask
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"Cannot read mask {mask_path}")

        # Resize
        H, W = self.img_size
        image_bgr = cv2.resize(image_bgr, (W, H), interpolation=cv2.INTER_AREA)
        mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0,1]
        image_bgr = image_bgr.astype(np.float32) / 255.0

        # Convert mask to [0,1]
        mask_gray = mask_gray.astype(np.float32) / 255.0

        # (H,W,3) -> (3,H,W)
        image_bgr = np.transpose(image_bgr, (2, 0, 1))  # shape: (3,H,W)

        # (H,W) -> (1,H,W)
        mask_gray = np.expand_dims(mask_gray, axis=0)   # shape: (1,H,W)

        # If you have data augmentations, apply them here using albumentations, etc.
        # For now, we skip or do a placeholder:
        # if self.transform:
        #     augmented = self.transform(image=image_bgr, mask=mask_gray)
        #     image_bgr = augmented['image']
        #     mask_gray = augmented['mask']

        # Convert to torch Tensors
        image_tensor = torch.from_numpy(image_bgr).float()   # (3,H,W)
        mask_tensor = torch.from_numpy(mask_gray).float()    # (1,H,W)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': filename
        }
