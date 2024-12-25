# infer_batch.py
import os
import torch
import cv2
import numpy as np
from model import UNetSmall

def load_model(model_path, device):
    model = UNetSmall(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path, img_size=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    original_size = image.shape[:2]  # H, W
    image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    image_resized = image_resized.astype(np.float32) / 255.0  # Normalize to [0,1]
    image_tensor = torch.from_numpy(np.transpose(image_resized, (2, 0, 1))).unsqueeze(0)  # (1, 3, H, W)
    return image_tensor, original_size

def postprocess_mask(pred_mask, original_size):
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
    pred_mask_resized = cv2.resize(pred_mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    binary_mask = (pred_mask_resized >= 0.5).astype(np.uint8)  # Threshold at 0.5
    return binary_mask

def visualize_and_save(image_path, binary_mask, output_mask_path, output_overlay_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    cv2.imwrite(output_mask_path, binary_mask * 255)

    overlay = original_image.copy()
    overlay[binary_mask == 1] = [0, 255, 0]
    cv2.imwrite(output_overlay_path, overlay)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference on a set of images using a trained U-Net.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pt)")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the directory of test images")
    parser.add_argument('--output_dir', type=str, default="outputs/", help="Path to save the predicted masks and overlays")
    parser.add_argument('--img_size', nargs=2, type=int, default=[512, 512], help="Resize image to this size (H, W)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)
        output_mask_path = os.path.join(args.output_dir, f"{os.path.splitext(image_file)[0]}_pred_mask.png")
        output_overlay_path = os.path.join(args.output_dir, f"{os.path.splitext(image_file)[0]}_overlay.png")

        try:
            image_tensor, original_size = preprocess_image(image_path, img_size=tuple(args.img_size))
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                pred_mask = torch.sigmoid(model(image_tensor))

            binary_mask = postprocess_mask(pred_mask, original_size)
            visualize_and_save(image_path, binary_mask, output_mask_path, output_overlay_path)
            print(f"[INFO] Processed {image_file}. Saved mask: {output_mask_path}, overlay: {output_overlay_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {image_file}: {e}")

if __name__ == "__main__":
    main()
