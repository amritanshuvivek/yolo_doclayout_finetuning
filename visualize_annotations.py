#!/usr/bin/env python3
"""Visualize document layout annotations.

This script provides functionality to visualize bounding box annotations on document
images. It supports both individual images and directories of images with their
corresponding YOLO-format annotations.

Example:
    Visualize a single image:
    $ python visualize_annotations.py --image data/train/images/doc1.jpg

    Visualize a random image from a directory:
    $ python visualize_annotations.py --image-dir data/train/images --random
"""

# Standard library
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Local
from config import get_class_names, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("visualization.log")],
)
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255],
                   thickness=tf, lineType=cv2.LINE_AA)

def visualize_annotations(image_path, label_path, class_names):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Read and process labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = [x.strip().split() for x in f.read().strip().splitlines()]
            
        # Draw boxes
        for label in labels:
            if len(label) >= 5:  # class, x, y, w, h
                class_id = int(label[0])
                x_center, y_center, width, height = map(float, label[1:5])
                
                # Convert from normalized to pixel coordinates
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                class_name = class_names.get(class_id, str(class_id))
                plot_one_box([x1, y1, x2, y2], img, label=class_name)
    
    return img

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize document layout annotations.')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--label', type=str, help='Path to label file (optional)')
    parser.add_argument('--image-dir', type=str, default='data/train/images', 
                       help='Directory containing images (used if --image not specified)')
    parser.add_argument('--label-dir', type=str, default='data/train/labels',
                       help='Directory containing labels (used if --label not specified)')
    parser.add_argument('--output', type=str, default='visualization.jpg',
                       help='Output path for visualization')
    parser.add_argument('--random', action='store_true',
                       help='Select a random image from the directory')
    return parser.parse_args()

def main():
    args = parse_args()
    class_names = get_class_names()
    
    # Handle input paths
    if args.image:
        image_path = args.image
        label_path = args.label or Path(args.image).with_suffix('.txt')
    else:
        # Get all image files in directory
        image_files = list(Path(args.image_dir).glob('*.[jJ][pP][gG]')) + \
                     list(Path(args.image_dir).glob('*.[pP][nN][gG]'))
        
        if not image_files:
            print(f"No images found in {args.image_dir}")
            return
            
        if args.random:
            image_path = str(random.choice(image_files))
        else:
            # Print available images and let user choose
            print("Available images:")
            for i, img_file in enumerate(image_files):
                print(f"{i+1}. {img_file.name}")
            selection = int(input("Select image number: ")) - 1
            image_path = str(image_files[selection])
            
        label_path = Path(args.label_dir) / f"{Path(image_path).stem}.txt"
    
    # Visualize
    try:
        img = visualize_annotations(image_path, str(label_path), class_names)
        
        # Save and show
        cv2.imwrite(args.output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to {args.output}")
        
        # Display the result
        cv2.imshow('Visualization', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()
