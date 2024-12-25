#!/usr/bin/env python3
import os
import csv
import cv2
import numpy as np

def parse_vgg_polyline(spatial_coords_str):
    """
    The VGG CSV has a field like:
    [6, x1, y1, x2, y2, x3, y3, ...]
    where 6 indicates a polyline shape ID.
    We parse it into a list of (x, y) points.
    """
    # Strip brackets
    coords_str = spatial_coords_str.strip('[]')
    coords_vals = coords_str.split(',')
    
    # The first value is shape_id=6 => polyline
    # The rest are x,y pairs
    shape_id = int(coords_vals[0].strip())
    if shape_id != 6:
        # Not a polyline, skip or raise an error
        return None
    
    float_vals = [float(v.strip()) for v in coords_vals[1:]]
    
    # Group them into pairs
    points = []
    for i in range(0, len(float_vals), 2):
        x = float_vals[i]
        y = float_vals[i+1]
        points.append((x, y))
    return points

def create_mask_for_image(img_path, mask_path, points):
    """
    Creates a mask of the same size as the image,
    then draws the polyline defined by 'points' in white on black background.
    """
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] Could not load image {img_path}")
        return
    H, W = image.shape[:2]
    
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Convert float coords to int
    pts = np.array(points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))  # shape: (N,1,2) for cv2.polylines
    
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=2)
    
    # Save the mask
    cv2.imwrite(mask_path, mask)
    # Optionally display debug info
    # print(f"[INFO] Saved mask to {mask_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create binary masks from VGG CSV polylines.")
    parser.add_argument('--image_dir', type=str, default='data/images', help='Directory of images')
    parser.add_argument('--csv_path', type=str, default='data/annotations.csv', help='Path to VGG annotations CSV')
    parser.add_argument('--mask_dir', type=str, default='data/masks', help='Where to save generated masks')
    args = parser.parse_args()
    
    os.makedirs(args.mask_dir, exist_ok=True)
    
    # Read CSV
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip VGG header lines that start with '#'
        # We'll parse lines containing the shape info
        for row in reader:
            if len(row) < 5:
                continue
            if row[0].startswith('#'):
                continue
            
            # row[0] might be some annotation ID
            # row[1] is something like '["IMG-0001-00001.jpg"]'
            # row[4] is the spatial_coordinates column that has [6, x1, y1, ...]
            file_list_str = row[1]  # e.g. '["IMG-0001-00001.jpg"]'
            spatial_coords_str = row[4]
            
            # Parse file_list_str: it might look like '["IMG-0001-00001.jpg"]'
            # We'll strip the brackets/quotes to get the filename
            file_list_str = file_list_str.strip('[]').strip('"')
            # If there's an extra pair of quotes, handle them
            if file_list_str.startswith('"'):
                file_list_str = file_list_str.strip('"')
            
            # Now we have something like 'IMG-0001-00001.jpg'
            filename = file_list_str
            
            # Parse the polyline
            points = parse_vgg_polyline(spatial_coords_str)
            if points is None:
                continue  # not a polyline or invalid shape
            
            img_path = os.path.join(args.image_dir, filename)
            mask_filename = os.path.splitext(filename)[0] + '_mask.png'
            mask_path = os.path.join(args.mask_dir, mask_filename)
            
            create_mask_for_image(img_path, mask_path, points)
    
    print("[INFO] Finished creating masks.")

if __name__ == '__main__':
    main()
