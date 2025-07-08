#!/usr/bin/env python3
"""Validate document layout dataset.

This script performs comprehensive validation of a document layout dataset,
ensuring the integrity of the directory structure, file formats, and annotations.
It checks for common issues like missing files, invalid annotations, and
inconsistent data.

Example:
    Validate the entire dataset:
    $ python validate_dataset.py --data-dir data

    Validate only the training split:
    $ python validate_dataset.py --data-dir data --split train
"""

# Standard library
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Third-party
import cv2
import numpy as np

# Local
from config import get_class_names, get_num_classes, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validate_dataset.log")
    ]
)
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]

class ValidationResult(Enum):
    """Possible validation outcomes."""
    VALID = auto()
    INVALID = auto()
    SKIPPED = auto()

@dataclass
class ValidationError:
    """Container for validation error information."""
    file_path: PathLike
    error_type: str
    message: str
    severity: str = "error"  # 'error', 'warning', or 'info'

def get_image_files(directory: Path) -> Set[str]:
    """Get set of image file stems from directory."""
    return {
        f.stem for f in directory.iterdir() 
        if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
    }

def get_label_files(directory: Path) -> Set[str]:
    """Get set of label file stems from directory."""
    return {f.stem for f in directory.glob('*.txt')}

def validate_annotation(label_file: Path, num_classes: int) -> List[str]:
    """Validate a single annotation file.
    
    Args:
        label_file: Path to the label file
        num_classes: Number of classes in the dataset
        
    Returns:
        List of error messages, empty if no errors
    """
    errors = []
    
    try:
        with open(label_file, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                
                # Check number of parts
                if len(parts) != 5:
                    errors.append(f"Line {i}: Expected 5 values, got {len(parts)}")
                    continue
                
                # Check class ID
                try:
                    class_id = int(parts[0])
                    if not (0 <= class_id < num_classes):
                        errors.append(f"Line {i}: Class ID {class_id} out of range [0, {num_classes-1}]")
                except ValueError:
                    errors.append(f"Line {i}: Invalid class ID '{parts[0]}'")
                
                # Check coordinates
                try:
                    coords = list(map(float, parts[1:5]))
                    if not all(0 <= c <= 1 for c in coords):
                        errors.append(f"Line {i}: Coordinates out of bounds [0, 1]: {coords}")
                except ValueError:
                    errors.append(f"Line {i}: Invalid coordinate values")
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
    
    return errors

def validate_dataset(data_dir: Path, dataset_type: str = 'dataset') -> Tuple[bool, Dict[str, Any]]:
    """Validate the dataset structure and annotations.
    
    Args:
        data_dir: Path to the dataset directory
        dataset_type: Type of dataset (train/val/test) for logging
        
    Returns:
        Tuple of (is_valid, results) where results contains detailed validation info
    """
    results = {
        'valid': True,
        'images_dir': data_dir / 'images',
        'labels_dir': data_dir / 'labels',
        'missing_images': [],
        'missing_labels': [],
        'annotation_errors': {},
        'stats': {
            'total_images': 0,
            'total_labels': 0,
            'valid_labels': 0,
            'invalid_labels': 0,
            'total_boxes': 0
        }
    }
    
    # Check directory structure
    if not results['images_dir'].exists():
        raise FileNotFoundError(f"Images directory not found: {results['images_dir']}")
    if not results['labels_dir'].exists():
        raise FileNotFoundError(f"Labels directory not found: {results['labels_dir']}")
    
    # Get file lists
    image_files = get_image_files(results['images_dir'])
    label_files = get_label_files(results['labels_dir'])
    
    results['stats']['total_images'] = len(image_files)
    results['stats']['total_labels'] = len(label_files)
    
    # Check for missing files
    results['missing_labels'] = list(image_files - label_files)
    results['missing_images'] = list(label_files - image_files)
    
    if results['missing_labels']:
        results['valid'] = False
        print(f"❌ {len(results['missing_labels'])} images missing labels")
    
    if results['missing_images']:
        results['valid'] = False
        print(f"❌ {len(results['missing_images'])} labels missing images")
    
    # Validate annotations
    for label_file in results['labels_dir'].glob('*.txt'):
        errors = validate_annotation(label_file, get_num_classes())
        if errors:
            results['valid'] = False
            results['annotation_errors'][label_file.name] = errors
            results['stats']['invalid_labels'] += 1
        else:
            results['stats']['valid_labels'] += 1
            
        # Count total boxes
        with open(label_file, 'r') as f:
            results['stats']['total_boxes'] += sum(1 for _ in f)
    
    return results['valid'], results

def print_validation_results(results: Dict[str, Any]) -> None:
    """Print formatted validation results."""
    print("\n" + "="*50)
    print(f"Dataset Validation Results")
    print("="*50)
    
    # Basic stats
    stats = results['stats']
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Images: {stats['total_images']}")
    print(f"  - Label files: {stats['total_labels']}")
    print(f"  - Bounding boxes: {stats['total_boxes']}")
    print(f"  - Valid labels: {stats['valid_labels']}")
    print(f"  - Invalid labels: {stats['invalid_labels']}")
    
    # Missing files
    if results['missing_labels']:
        print(f"\n❌ Missing labels for {len(results['missing_labels'])} images:")
        for i, img in enumerate(results['missing_labels'][:5], 1):
            print(f"  {i}. {img}")
        if len(results['missing_labels']) > 5:
            print(f"  ... and {len(results['missing_labels']) - 5} more")
    
    if results['missing_images']:
        print(f"\n❌ Missing images for {len(results['missing_images'])} labels:")
        for i, lbl in enumerate(results['missing_images'][:5], 1):
            print(f"  {i}. {lbl}")
        if len(results['missing_images']) > 5:
            print(f"  ... and {len(results['missing_images']) - 5} more")
    
    # Annotation errors
    if results['annotation_errors']:
        print(f"\n❌ Found errors in {len(results['annotation_errors'])} label files:")
        for i, (file, errors) in enumerate(list(results['annotation_errors'].items())[:3], 1):
            print(f"  {i}. {file}:")
            for err in errors[:3]:
                print(f"     - {err}")
            if len(errors) > 3:
                print(f"     ... and {len(errors) - 3} more errors")
        if len(results['annotation_errors']) > 3:
            print(f"  ... and {len(results['annotation_errors']) - 3} more files with errors")
    
    # Final result
    if results['valid']:
        print("\n✅ Dataset is valid!")
    else:
        print("\n❌ Dataset validation failed!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate document layout dataset.')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Root directory containing train/val/test subdirectories')
    parser.add_argument('--split', type=str, choices=['all', 'train', 'val', 'test'], 
                      default='all', help='Which dataset split to validate')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    print(f"🔍 Validating dataset in: {data_dir.absolute()}")
    print(f"Using class names: {get_class_names()}")
    
    splits_to_validate = []
    if args.split == 'all':
        splits_to_validate = ['train', 'val', 'test']
    else:
        splits_to_validate = [args.split]
    
    all_valid = True
    
    for split in splits_to_validate:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"⚠️  {split} directory not found: {split_dir}")
            continue
            
        print(f"\n{'='*30} {split.upper()} {'='*30}")
        try:
            is_valid, results = validate_dataset(split_dir, split)
            print_validation_results(results)
            if not is_valid:
                all_valid = False
        except Exception as e:
            print(f"❌ Error validating {split}: {e}")
            all_valid = False
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())
