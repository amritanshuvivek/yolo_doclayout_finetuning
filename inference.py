import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import argparse
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

from doclayout_yolo import YOLOv10
from train_utils import setup_logging, load_config

# Setup logging
logger = setup_logging(log_dir='logs', log_file='inference.log')

class LayoutDetector:
    def __init__(self, model_path, config_path='config/training_config.yaml'):
        """Initialize the layout detector with a trained model.
        
        Args:
            model_path: Path to the trained model weights
            config_path: Path to the training configuration file
        """
        # Load and validate config
        self.config = load_config(config_path)
        
        # Set device
        use_cuda = self.config['model'].get('device', 'cuda') == 'cuda'
        self.device = get_device(use_cuda=use_cuda)
        
        # Load model
        try:
            self.model = YOLOv10(model_path).to(self.device)
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Get class names from config
        self.class_names = self.config['model']['class_names']
        logger.info(f"Using {len(self.class_names)} classes")
    
    def preprocess(self, image_path):
        """Preprocess the input image."""
        # Read and resize image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Resize maintaining aspect ratio
        target_size = self.config['data']['image_size']
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Pad to make it square
        image = cv2.resize(image, (new_w, new_h))
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = image
        
        # Normalize
        image = padded.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image, (w, h), (new_w / w, new_h / h)
    
    def detect(self, image_path, conf_threshold=0.5, iou_threshold=0.45):
        """Detect layout elements in an image."""
        # Preprocess
        image, orig_size, scale = self.preprocess(image_path)
        image = image.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image)
        
        # Process outputs
        results = []
        for output in outputs.xyxy[0]:
            x1, y1, x2, y2, conf, cls = output.cpu().numpy()
            
            if conf < conf_threshold:
                continue
            
            # Scale boxes back to original image size
            x1 = int(x1 / scale[0])
            y1 = int(y1 / scale[1])
            x2 = int(x2 / scale[0])
            y2 = int(y2 / scale[1])
            
            class_name = self.class_names.get(int(cls), str(int(cls)))
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class': class_name,
                'class_id': int(cls)
            })
        
        # Apply NMS
        results = self.non_max_suppression(results, iou_threshold)
        
        return results
    
    def non_max_suppression(self, boxes, iou_threshold):
        """Apply non-maximum suppression to remove overlapping boxes."""
        if not boxes:
            return []
        
        # Convert to numpy array for easier manipulation
        boxes = np.array([[
            box['bbox'][0], box['bbox'][1], 
            box['bbox'][2], box['bbox'][3], 
            box['confidence'], 
            box['class_id']
        ] for box in boxes])
        
        # Sort by confidence (descending)
        boxes = boxes[boxes[:, 4].argsort()[::-1]]
        
        keep = []
        while boxes.size > 0:
            # Keep the box with highest confidence
            keep.append(boxes[0])
            
            if boxes.shape[0] == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = self.box_iou(
                boxes[0:1, :4],  # Current box
                boxes[1:, :4]     # All other boxes
            )
            
            # Remove boxes with IoU > threshold
            boxes = boxes[1:][ious[0] <= iou_threshold]
        
        # Convert back to list of dicts
        results = []
        for box in keep:
            results.append({
                'bbox': box[:4].astype(int).tolist(),
                'confidence': float(box[4]),
                'class': self.class_names.get(int(box[5]), str(int(box[5]))),
                'class_id': int(box[5])
            })
            
        return results
    
    @staticmethod
    def box_iou(box1, box2):
        """Calculate Intersection over Union between two sets of boxes."""
        # box1: (n, 4), box2: (m, 4)
        # Returns: (n, m) iou matrix
        
        # Get coordinates of intersections
        x1 = np.maximum(box1[:, None, 0], box2[:, 0])
        y1 = np.maximum(box1[:, None, 1], box2[:, 1])
        x2 = np.minimum(box1[:, None, 2], box2[:, 2])
        y2 = np.minimum(box1[:, None, 3], box2[:, 3])
        
        # Calculate areas
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        # Calculate IoU
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        
        return iou

def visualize_detections(image_path, detections, output_path='output.jpg'):
    """Visualize detection results on the image."""
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define colors for different classes
    colors = {
        'text': (255, 0, 0),      # Red
        'title': (0, 255, 0),     # Green
        'figure': (0, 0, 255),    # Blue
        'table': (255, 255, 0),   # Cyan
        'list': (255, 0, 255),    # Magenta
        'header': (0, 255, 255),  # Yellow
        'footer': (128, 0, 128),  # Purple
        'caption': (0, 128, 128), # Teal
        'formula': (128, 128, 0), # Olive
        'page_number': (128, 0, 0) # Maroon
    }
    
    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Get color for this class
        color = colors.get(class_name, (128, 128, 128))  # Default to gray
        
        # Draw rectangle
        cv2.rectangle(
            image, 
            (x1, y1), (x2, y2), 
            color, 2
        )
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            image, 
            (x1, y1 - 20), (x1 + w, y1), 
            color, -1
        )
        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    # Save the result
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to {output_path}")
    return image

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Layout Detection Inference')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Path to save the output visualization')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for detection')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='IOU threshold for NMS')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = LayoutDetector(args.model)
    
    # Run detection
    detections = detector.detect(
        args.image,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Print results
    print(f"Detected {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class']} (conf: {det['confidence']:.2f}): {det['bbox']}")
    
    # Visualize results
    visualize_detections(args.image, detections, args.output)

if __name__ == "__main__":
    main()
