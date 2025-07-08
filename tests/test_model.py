#!/usr/bin/env python3
"""
Test script for the YOLOv10 document layout model.
"""

import unittest
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from doclayout_yolo import YOLOv10

class TestYOLOv10Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model_path = 'doclayout_yolo_docstructbench_imgsz1024.pt'
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        try:
            model = YOLOv10(self.model_path).to(self.device)
            self.assertIsNotNone(model)
            
            # Check model is in evaluation mode
            self.assertFalse(model.training)
            
            # Check model device
            self.assertEqual(next(model.parameters()).device, self.device)
            
        except Exception as e:
            self.fail(f"Model loading failed with error: {str(e)}")
    
    def test_model_inference(self):
        """Test that the model can perform inference."""
        try:
            # Initialize model
            model = YOLOv10(self.model_path).to(self.device)
            model.eval()
            
            # Create a dummy input
            dummy_input = torch.randn(1, 3, 1024, 1024).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output format
            self.assertIsInstance(output, dict)
            self.assertIn('pred_boxes', output)
            self.assertIn('scores', output)
            self.assertIn('labels', output)
            
        except Exception as e:
            self.fail(f"Model inference failed with error: {str(e)}")
    
    def test_model_training_mode(self):
        """Test that the model can be switched to training mode."""
        try:
            model = YOLOv10(self.model_path).to(self.device)
            model.train()
            self.assertTrue(model.training)
            
            # Switch back to eval mode
            model.eval()
            self.assertFalse(model.training)
            
        except Exception as e:
            self.fail(f"Model training mode test failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
