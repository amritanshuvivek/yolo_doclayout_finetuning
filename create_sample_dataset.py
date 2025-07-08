import os
import cv2
import numpy as np
from pathlib import Path

# Create directories
os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/labels', exist_ok=True)
os.makedirs('data/val/images', exist_ok=True)
os.makedirs('data/val/labels', exist_ok=True)

def create_sample_image(output_path):
    """Create a simple image with some colored rectangles."""
    img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255  # White background
    
    # Add some colored rectangles
    cv2.rectangle(img, (100, 100), (500, 200), (0, 0, 255), 2)  # Title (red)
    cv2.rectangle(img, (100, 250), (900, 800), (0, 255, 0), 2)  # Text (green)
    cv2.rectangle(img, (550, 100), (900, 250), (255, 0, 0), 2)  # Figure (blue)
    
    cv2.imwrite(output_path, img)
    return img

def create_label_file(image_path, label_path):
    """Create a sample label file with some annotations."""
    # Get image dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Create YOLO format annotations (class_id x_center y_center width height)
    # Normalized to [0, 1]
    annotations = [
        [1, 300/w, 150/h, 400/w, 100/h],  # Title
        [0, 500/w, 525/h, 800/w, 550/h],   # Text
        [2, 725/w, 175/h, 350/w, 150/h]    # Figure
    ]
    
    # Write to file
    with open(label_path, 'w') as f:
        for ann in annotations:
            line = ' '.join(str(x) for x in ann) + '\n'
# Create training samples
for i in range(10):
    img_path = f'data/train/images/sample_{i}.jpg'
    label_path = f'data/train/labels/sample_{i}.txt'
    create_sample_image(img_path)
    create_label_file(img_path, label_path)

# Create validation samples
for i in range(2):
    img_path = f'data/val/images/val_{i}.jpg'
    label_path = f'data/val/labels/val_{i}.txt'
    create_sample_image(img_path)
    create_label_file(img_path, label_path)

print("Created sample dataset in data/{train,val}/{images,labels}/")
print("You can now run: python train.py")
