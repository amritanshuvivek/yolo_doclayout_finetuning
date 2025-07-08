#!/usr/bin/env python3
"""
Test script for the DocumentLayoutDataset class.
"""

import os
import sys
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from train_utils import DocumentLayoutDataset

class TestDocumentLayoutDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        # Create a small test dataset
        cls.test_dir = Path("tests/test_data")
        cls.img_dir = cls.test_dir / "images"
        cls.label_dir = cls.test_dir / "labels"
        
        # Create directories
        cls.img_dir.mkdir(parents=True, exist_ok=True)
        cls.label_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy image and label file
        cls.create_dummy_data()
        
    @classmethod
    def create_dummy_data(cls):
        """Create a dummy image and label file for testing."""
        import cv2
        import numpy as np
        
        # Create a dummy image
        img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
        img_path = cls.img_dir / "test.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Create a dummy label file (YOLO format)
        label_path = cls.label_dir / "test.txt"
        with open(label_path, 'w') as f:
            # Format: class_id x_center y_center width height
            f.write("0 0.5 0.5 0.2 0.2\n")  # A text box in the center
            f.write("1 0.1 0.1 0.15 0.1\n")  # A title in the top-left
    
    def test_dataset_loading(self):
        """Test that the dataset loads correctly."""
        dataset = DocumentLayoutDataset(
            img_dir=str(self.img_dir),
            label_dir=str(self.label_dir),
            img_size=1024,
            augment=False
        )
        
        # Check dataset length
        self.assertEqual(len(dataset), 1)
        
        # Check item access
        img, target = dataset[0]
        
        # Check image shape and type
        self.assertEqual(img.shape, (3, 1024, 1024))
        self.assertIsInstance(img, torch.Tensor)
        
        # Check target structure
        self.assertIn('boxes', target)
        self.assertIn('labels', target)
        self.assertEqual(len(target['boxes']), 2)  # Two objects in our test data
        self.assertEqual(len(target['labels']), 2)
    
    def test_dataloader(self):
        """Test that the dataloader works with the dataset."""
        dataset = DocumentLayoutDataset(
            img_dir=str(self.img_dir),
            label_dir=str(self.label_dir),
            img_size=1024,
            augment=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Check that we can iterate through the dataloader
        batch = next(iter(dataloader))
        self.assertEqual(len(batch), 2)  # Images and targets
        self.assertEqual(len(batch[0]), 1)  # Batch size of 1
        self.assertEqual(len(batch[1]), 1)  # Corresponding targets
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()
