import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from train_utils import DocumentLayoutDataset
import yaml

def test_data_loading():
    # Load config
    config_path = 'config/training_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Test training data loading
    print("\n=== Testing Training Data Loading ===")
    train_dataset = DocumentLayoutDataset(
        img_dir=os.path.join(data_config['train_dir'], 'images'),
        label_dir=os.path.join(data_config['train_dir'], 'labels'),
        img_size=data_config['image_size'],
        augment=False
    )
    
    print(f"Found {len(train_dataset)} training samples")
    if len(train_dataset) > 0:
        sample_img, sample_target = train_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample target keys: {sample_target.keys()}")
        print(f"Number of boxes: {len(sample_target['boxes'])}")
        if len(sample_target['boxes']) > 0:
            print(f"First box coordinates: {sample_target['boxes'][0]}")
            print(f"First box label: {sample_target['labels'][0]}")
    
    # Test validation data loading
    print("\n=== Testing Validation Data Loading ===")
    val_dataset = DocumentLayoutDataset(
        img_dir=os.path.join(data_config['val_dir'], 'images'),
        label_dir=os.path.join(data_config['val_dir'], 'labels'),
        img_size=data_config['image_size'],
        augment=False
    )
    
    print(f"Found {len(val_dataset)} validation samples")
    if len(val_dataset) > 0:
        sample_img, sample_target = val_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample target keys: {sample_target.keys()}")
        print(f"Number of boxes: {len(sample_target['boxes'])}")
        if len(sample_target['boxes']) > 0:
            print(f"First box coordinates: {sample_target['boxes'][0]}")
            print(f"First box label: {sample_target['labels'][0]}")

if __name__ == "__main__":
    test_data_loading()
