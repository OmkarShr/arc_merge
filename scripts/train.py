# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import CurveDataset
from model import UNetSmall

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a small U-Net to segment curves.")
    parser.add_argument('--image_dir', type=str, default='data/images', help='Path to images')
    parser.add_argument('--mask_dir', type=str, default='data/masks', help='Path to binary masks')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='models/best_model.pt')
    parser.add_argument('--img_size', nargs=2, type=int, default=[512,512], help='(H,W) to resize images and masks')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Collect all images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts)]
    if not all_images:
        raise RuntimeError(f"No images found in {args.image_dir}")
    all_images.sort()

    # 2. Split train/val
    train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

    # 3. Create Datasets
    train_dataset = CurveDataset(
        list_of_images=train_files,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=tuple(args.img_size),
        transform=None
    )
    val_dataset = CurveDataset(
        list_of_images=val_files,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=tuple(args.img_size),
        transform=None
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Create model, loss, optimizer
    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    # 5. Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)  # (B,3,H,W)
            masks  = batch['mask'].to(device)   # (B,1,H,W)

            # Forward
            outputs = model(images)  # (B,1,H/2,W/2) if not upsampled
            # If the output is smaller, you might need to downsample the mask to match or upsample
            # For simplicity, let's also downsample the mask by 2:
            # But if you want full resolution, see note below.
            if outputs.shape[-1] != masks.shape[-1] or outputs.shape[-2] != masks.shape[-2]:
                # downsample the mask to match the model output size
                # (this is just a quick fix for the half-resolution output)
                masks = nn.functional.interpolate(masks, size=outputs.shape[-2:], mode='nearest')

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks  = batch['mask'].to(device)

                outputs = model(images)
                # Adjust mask dimension if needed
                if outputs.shape[-1] != masks.shape[-1] or outputs.shape[-2] != masks.shape[-2]:
                    masks = nn.functional.interpolate(masks, size=outputs.shape[-2:], mode='nearest')

                val_loss = criterion(outputs, masks)
                val_loss_sum += val_loss.item()

        val_loss_avg = val_loss_sum / len(val_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss_avg:.4f}")

        # Save best
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), args.save_path)
            print(f"[INFO] Saved best model with Val Loss: {val_loss_avg:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
