"""Training utilities for document layout detection.

This module provides essential utilities for training document layout detection models,
including dataset handling, data augmentation, model optimization, and training loops.
"""

# Standard library
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party
import albumentations as A
import cv2
import numpy as np
import torch
import torch.optim as optim
import yaml
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

# Type aliases
PathLike = Union[str, Path]
Transform = Callable[[Dict[str, Any]], Dict[str, Any]]

# Configure module-level logger
logger = logging.getLogger(__name__)

class DocumentLayoutDataset(Dataset):
    """Custom dataset for document layout detection."""
    
    def __init__(self, 
                 img_dir: str, 
                 label_dir: str, 
                 img_size: int = 1024,
                 augment: bool = True,
                 aug_config: Optional[Dict] = None):
        """
        Args:
            img_dir: Directory with images
            label_dir: Directory with YOLO-format labels
            img_size: Target image size (square)
            augment: Whether to apply data augmentation
            aug_config: Augmentation configuration
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_config = aug_config or {}
        
        # Get list of image files and filter out those without valid labels
        img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        
        # Filter out images without corresponding labels or with empty label files
        self.img_files = []
        for img_path in img_files:
            label_path = self.label_dir / (img_path.stem + '.txt')
            if label_path.exists() and label_path.stat().st_size > 0:
                self.img_files.append(img_path)
            else:
                logger.warning(f"Skipping {img_path.name}: No valid label file found")
        
        if not self.img_files:
            raise ValueError(f"No valid image-label pairs found in {self.img_dir}")
            
        logger.info(f"Found {len(self.img_files)} valid image-label pairs in {self.img_dir}")
        
        # Initialize transforms
        self.transform = self._get_transforms()
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Load image
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes, labels = self._parse_label_file(label_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Convert to tensor
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        
        return image, target
    
    def _parse_label_file(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Parse YOLO format label file."""
        boxes = []
        labels = []
        
        if not label_path.exists():
            return boxes, labels
            
        # If the label file is empty but we have a corresponding image, create a dummy label
        if label_path.stat().st_size == 0:
            # Create a dummy label covering the entire image
            dummy_label = "0 0.5 0.5 0.9 0.9"  # class 0 (text) covering most of the image
            with open(label_path, 'w') as f:
                f.write(dummy_label)
            logger.warning(f"Created dummy label for {label_path.name}")
            
            # Parse the dummy label
            parts = dummy_label.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:5]))
            
            # Convert YOLO format to [x_min, y_min, x_max, y_max]
            x_center, y_center, width, height = coords
            x_min = (x_center - width / 2) * self.img_size
            y_min = (y_center - height / 2) * self.img_size
            x_max = (x_center + width / 2) * self.img_size
            y_max = (y_center + height / 2) * self.img_size
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)
            return boxes, labels
                    
        # Parse existing non-empty label file
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                    try:
                        class_id = int(parts[0])
                        coords = list(map(float, parts[1:5]))
                        
                        # Validate coordinates are in [0, 1] range
                        if all(0 <= x <= 1 for x in coords):
                            # Convert YOLO format to [x_min, y_min, x_max, y_max]
                            x_center, y_center, width, height = coords
                            x_min = (x_center - width / 2) * self.img_size
                            y_min = (y_center - height / 2) * self.img_size
                            x_max = (x_center + width / 2) * self.img_size
                            y_max = (y_center + height / 2) * self.img_size
                            
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
                        else:
                            logger.warning(f"Invalid coordinates in {label_path}: {coords}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line in {label_path}: {line.strip()} - {e}")
        
        # If no valid labels were found, create a dummy one
        if not boxes:
            dummy_label = "0 0.5 0.5 0.9 0.9"  # class 0 (text) covering most of the image
            with open(label_path, 'w') as f:
                f.write(dummy_label)
            logger.warning(f"Created dummy label for {label_path.name} (invalid format)")
            
            # Parse the dummy label
            parts = dummy_label.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:5]))
            
            # Convert YOLO format to [x_min, y_min, x_max, y_max]
            x_center, y_center, width, height = coords
            x_min = (x_center - width / 2) * self.img_size
            y_min = (y_center - height / 2) * self.img_size
            x_max = (x_center + width / 2) * self.img_size
            y_max = (y_center + height / 2) * self.img_size
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)
        
        return boxes, labels
    
    def _get_transforms(self):
        """Create data augmentation pipeline."""
        if not self.augment:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),
                A.Normalize(
                    mean=self.aug_config.get('mean', [0.485, 0.456, 0.406]),
                    std=self.aug_config.get('std', [0.229, 0.224, 0.225])
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        # Training augmentations
        return A.Compose([
            # Geometric transforms
            A.OneOf([
                A.HorizontalFlip(p=self.aug_config.get('horizontal_flip', 0.5)),
                A.VerticalFlip(p=self.aug_config.get('vertical_flip', 0.3)),
                A.RandomRotate90()
            ], p=0.7),
            
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=self.aug_config.get('scale', [0, 0.1]),
                rotate_limit=self.aug_config.get('rotate', 10),
                p=0.7
            ),
            
            # Color transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.aug_config.get('brightness', 0.2),
                    contrast_limit=self.aug_config.get('contrast', 0.2),
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=self.aug_config.get('hue', 0.02),
                    sat_shift_limit=self.aug_config.get('saturation', 0.1),
                    val_shift_limit=0.1,
                    p=0.5
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=0.5
                )
            ], p=0.8),
            
            # Advanced augmentations
            A.OneOf([
                A.CoarseDropout(
                    max_holes=self.aug_config.get('cutout_holes', 8),
                    max_height=int(self.img_size * self.aug_config.get('cutout_size', [0.2, 0.3])[1]),
                    max_width=int(self.img_size * self.aug_config.get('cutout_size', [0.2, 0.3])[1]),
                    min_holes=1,
                    min_height=int(self.img_size * self.aug_config.get('cutout_size', [0.1, 0.2])[0]),
                    min_width=int(self.img_size * self.aug_config.get('cutout_size', [0.1, 0.2])[0]),
                    p=self.aug_config.get('cutout', 0.1)
                ),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2)
            ], p=0.3),
            
            # Standard transforms
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.Normalize(
                mean=self.aug_config.get('mean', [0.485, 0.456, 0.406]),
                std=self.aug_config.get('std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def collate_fn(batch):
    """Custom collate function for variable number of objects."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images, 0), targets


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, delta: float = 0, path: str = 'checkpoints/best.pt', 
                 verbose: bool = True):
        """
        Args:
            patience: How long to wait after last time validation loss improved.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
            path: Path for the checkpoint to be saved to.
            verbose: If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_optimizer(model, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer with layer-wise learning rates based on config.
    
    Args:
        model: The model to optimize
        config: Training configuration dictionary
        
    Returns:
        Configured optimizer with parameter groups
    """
    optimizer_type = config['training'].get('optimizer', 'AdamW').lower()
    base_lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training'].get('weight_decay', 0.0005))
    
    # Get learning rate multipliers from config with defaults
    lr_mult = config['model'].get('lr_multipliers', {})
    backbone_mult = float(lr_mult.get('backbone', 0.1))
    neck_mult = float(lr_mult.get('neck', 1.0))
    head_mult = float(lr_mult.get('head', 1.0))
    
    # Group parameters by their type
    param_groups = [
        {'params': [], 'lr': base_lr * backbone_mult, 'name': 'backbone'},
        {'params': [], 'lr': base_lr * neck_mult, 'name': 'neck'},
        {'params': [], 'lr': base_lr * head_mult, 'name': 'head'}
    ]
    
    # Categorize parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'backbone' in name:
            param_groups[0]['params'].append(param)
        elif 'neck' in name:
            param_groups[1]['params'].append(param)
        else:  # Head parameters
            param_groups[2]['params'].append(param)
    
    # Remove empty parameter groups
    param_groups = [g for g in param_groups if g['params']]
    
    # Log parameter groups
    logger = logging.getLogger(__name__)
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        logger.info(f"Optimizer group '{group['name']}': {num_params/1e6:.2f}M params, lr={group['lr']}")
    
    # Create optimizer with parameter groups
    if optimizer_type == 'adamw':
        return optim.AdamW(
            param_groups, 
            lr=base_lr,  # Will be overridden by parameter groups
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            param_groups,
            lr=base_lr,  # Will be overridden by parameter groups
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            param_groups,
            lr=base_lr,  # Will be overridden by parameter groups
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def create_scheduler(optimizer, config: Dict, train_loader: DataLoader = None) -> Optional[object]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration dictionary
        train_loader: Optional training loader for steps per epoch calculation
        
    Returns:
        Configured learning rate scheduler or None if not specified
    """
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
    
    # Calculate steps per epoch if train_loader is provided
    steps_per_epoch = len(train_loader) if train_loader else 0
    
    if scheduler_type.lower() == 'reducelronplateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_config.get('factor', 0.1)),
            patience=int(scheduler_config.get('patience', 3)),
            min_lr=float(config['training'].get('min_lr', 1e-6))
        )
    
    elif scheduler_type.lower() == 'cosineannealinglr':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'] * steps_per_epoch,
            eta_min=float(config['training'].get('min_lr', 1e-6))
        )
    
    elif scheduler_type.lower() == 'onecyclelr':
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(config['training']['learning_rate']),
            epochs=config['training']['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=float(scheduler_config.get('pct_start', 0.3)),
            anneal_strategy='cos',
            final_div_factor=float(scheduler_config.get('final_div_factor', 1e4)),
            div_factor=float(scheduler_config.get('div_factor', 25.0)),
            three_phase=scheduler_config.get('three_phase', False)
        )
    
    elif scheduler_type.lower() == 'cosineannealingwarmrestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(scheduler_config.get('t_0', 5) * steps_per_epoch),
            T_mult=int(scheduler_config.get('t_mult', 1)),
            eta_min=float(config['training'].get('min_lr', 1e-6))
        )
    
    return None

def get_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config['data']
    train_config = config['training']
    
    # Get absolute paths
    train_img_dir = os.path.join(data_config['train_dir'], 'images')
    train_label_dir = os.path.join(data_config['train_dir'], 'labels')
    val_img_dir = os.path.join(data_config['val_dir'], 'images')
    val_label_dir = os.path.join(data_config['val_dir'], 'labels')
    
    # Log dataset paths
    logger.info(f"Training images: {train_img_dir}")
    logger.info(f"Training labels: {train_label_dir}")
    logger.info(f"Validation images: {val_img_dir}")
    logger.info(f"Validation labels: {val_label_dir}")
    
    # Create datasets
    try:
        train_dataset = DocumentLayoutDataset(
            img_dir=train_img_dir,
            label_dir=train_label_dir,
            img_size=data_config['image_size'],
            augment=True,
            aug_config=data_config.get('augmentation', {})
        )
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        raise
    
    try:
        val_dataset = DocumentLayoutDataset(
            img_dir=val_img_dir,
            label_dir=val_label_dir,
            img_size=data_config['image_size'],
            augment=False,
            aug_config=data_config.get('augmentation', {})
        )
    except Exception as e:
        logger.warning(f"Error creating validation dataset: {e}. Using training set for validation.")
        val_dataset = train_dataset
    
    # Log dataset sizes
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Adjust batch size if it's larger than the dataset size
    batch_size = min(train_config['batch_size'], len(train_dataset))
    if batch_size != train_config['batch_size']:
        logger.warning(f"Reducing batch size from {train_config['batch_size']} to {batch_size} to match dataset size")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(train_config.get('num_workers', 4), os.cpu_count()),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_batch_size = min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=min(train_config.get('num_workers', 4), os.cpu_count()),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def load_config(config_path: str = 'config/training_config.yaml') -> Dict:
    """Load and validate training configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated configuration dictionary
    """
    # Load config file
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['model']['device'] = device
    
    # Create output directories
    os.makedirs(config['training'].get('output_dir', 'checkpoints'), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Validate configuration
    _validate_config(config)
    
    return config


def _validate_config(config: Dict):
    """Validate configuration parameters."""
    # Check required keys
    required_keys = {
        'model': ['pretrained', 'num_classes', 'input_size'],
        'training': ['epochs', 'batch_size', 'learning_rate'],
        'data': ['train_dir', 'val_dir', 'image_size']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing section '{section}' in config")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required key '{key}' in config['{section}']")
    
    # Validate paths
    model_path = config['model']['pretrained']
    if not os.path.exists(model_path) and not os.path.exists(os.path.join('checkpoints', model_path)):
        logging.warning(f"Model file not found: {model_path}")
    
    # Validate data directories
    for split in ['train_dir', 'val_dir']:
        dir_path = config['data'][split]
        if not os.path.exists(dir_path):
            logging.warning(f"Directory not found: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Set default values
    config['training'].setdefault('weight_decay', 0.0005)
    config['training'].setdefault('num_workers', 4)
    config['training'].setdefault('save_interval', 5)
    config['training'].setdefault('output_dir', 'checkpoints')
    config['training'].setdefault('early_stopping_patience', 10)
    config['training'].setdefault('min_lr', 1e-6)
    config['training'].setdefault('warmup_epochs', 3)
    
    # Set default augmentation values if not specified
    aug_config = config['data'].setdefault('augmentation', {})
    aug_config.setdefault('horizontal_flip', 0.5)
    aug_config.setdefault('vertical_flip', 0.3)
    aug_config.setdefault('rotate', 10)
    aug_config.setdefault('scale', [0.9, 1.1])
    aug_config.setdefault('brightness', 0.1)
    aug_config.setdefault('contrast', 0.1)
    aug_config.setdefault('saturation', 0.1)
    aug_config.setdefault('hue', 0.02)
    aug_config.setdefault('cutout', 0.1)
    aug_config.setdefault('cutout_holes', 8)
    aug_config.setdefault('cutout_size', [0.1, 0.2])
    aug_config.setdefault('mean', [0.485, 0.456, 0.406])
    aug_config.setdefault('std', [0.229, 0.224, 0.225])

def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device (CPU or GPU) for training/inference.
    
    Args:
        use_cuda: Whether to attempt using CUDA if available
        
    Returns:
        torch.device: The appropriate device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        logging.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logging.info('Using CPU')
    return device

def setup_logging(log_dir: str = 'logs', log_file: str = 'training.log', 
                 console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_file: Name of the log file
        console_level: Log level for console output
        file_level: Log level for file output
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file), mode='a')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(min(console_level, file_level))
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('albumentations').setLevel(logging.INFO)
    
    return logger
