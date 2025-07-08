#!/usr/bin/env python3
"""Document Layout Detection Training Script.

This script handles the end-to-end training of a YOLOv10-based document layout
detection model. It includes data loading, model training, validation, and
checkpointing with support for resuming training and early stopping.

Example:
    # Train with default configuration
    $ python train.py

    # Train with custom config and device
    $ python train.py --config config/custom_config.yaml --device cuda:0

    # Resume training from checkpoint
    $ python train.py --resume checkpoints/checkpoint_epoch10.pt
"""

# Standard library
import json
import logging
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local
from config import get_class_names
from doclayout_yolo import YOLOv10
from train_utils import (
    EarlyStopping,
    create_optimizer,
    create_scheduler,
    get_data_loaders,
    get_device,
    load_config,
    setup_logging,
)

# Type aliases
PathLike = Union[str, Path]

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable tokenizers parallelism (causes warnings)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)

class DocumentLayoutTrainer:
    """Document Layout Detection Trainer.
    
    Handles the training loop, validation, and model checkpointing for document
    layout detection using YOLOv10.
    """
    
    def __init__(self, config_path='config/training_config.yaml'):
        """Initialize the trainer.
        
        Args:
            config_path: Path to the training configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(
            log_dir='logs',
            log_file=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # Setup device
        use_cuda = self.config['model'].get('device', 'cuda') == 'cuda'
        self.device = get_device(use_cuda=use_cuda)
        self.logger.info(f"Using device: {self.device}")
        
        # Log configuration
        self.logger.info("Training configuration:")
        self.logger.info(json.dumps(self.config, indent=2, default=str))
        
        # Initialize model
        self.model = self._init_model()
        
        # Setup data loaders
        self.train_loader, self.val_loader = get_data_loaders(self.config)
        
        # Training setup
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(
            self.optimizer, 
            self.config,
            train_loader=self.train_loader
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training'].get('early_stopping_patience', 10),
            delta=0,
            path=os.path.join(self.config['training']['output_dir'], 'best.pt'),
            verbose=True
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(self.device) else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def _init_model(self):
        """Initialize the YOLOv10 model."""
        try:
            model_path = self.config['model']['pretrained']
            self.logger.info(f"Loading model from {model_path}")
            
            # Try to load from absolute path first, then check checkpoints directory
            if not os.path.exists(model_path):
                checkpoint_path = os.path.join('checkpoints', model_path)
                if os.path.exists(checkpoint_path):
                    model_path = checkpoint_path
            
            # Load the model first
            model = YOLOv10(model_path).to(self.device)
            
            # Set the data configuration path as an attribute
            data_config = os.path.abspath('data.yaml')
            self.logger.info(f"Using data configuration from {data_config}")
            model.data = data_config  # Store the data config path for later use
            
            # Log model architecture
            self.logger.info(f"Model architecture:\n{model}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def train_epoch(self, epoch: int):
        """Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        # Initialize model with data configuration
        data_config = os.path.abspath('data.yaml')
        self.model.data = data_config
        
        # Set model to training mode
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Initialize metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move data to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Update metrics
            reduced_loss = loss.detach()
            losses.update(reduced_loss.item(), images.size(0))
            epoch_loss += reduced_loss.item()
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Log progress
            if batch_idx % self.config['training'].get('print_freq', 10) == 0:
                self.logger.info(
                    f"Epoch: [{epoch}][{batch_idx}/{num_batches}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"LR {self._get_learning_rate():.6f}"
                )
            
            self.global_step += 1
        
        # Update learning rate for epoch-based schedulers
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss / num_batches)
            else:
                self.scheduler.step()
        
        # Log epoch metrics
        avg_loss = epoch_loss / num_batches
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['learning_rates'].append(self._get_learning_rate())
        
        self.logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, LR: {self._get_learning_rate():.6f}")
        
        return avg_loss
    
    def validate(self, epoch: int = 0):
        """Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number (for logging)
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Initialize metrics
        batch_time = AverageMeter()
        losses = AverageMeter()
        
        with torch.no_grad():
            end = time.time()
            
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # Move data to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                
                # Update metrics
                reduced_loss = loss.detach()
                losses.update(reduced_loss.item(), images.size(0))
                val_loss += reduced_loss.item()
                
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Log progress
                if batch_idx % self.config['training'].get('print_freq', 10) == 0:
                    self.logger.info(
                        f"Val: [{epoch}][{batch_idx}/{num_batches}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})"
                    )
        
        # Calculate average validation loss
        avg_val_loss = val_loss / num_batches
        self.metrics['val_loss'].append(avg_val_loss)
        
        self.logger.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def train(self):
        """Run the training loop."""
        start_time = time.time()
        
        # Set the data configuration
        data_config = os.path.abspath('data.yaml')
        self.model.data = data_config
        self.logger.info(f"Using data configuration from: {data_config}")
        
        # Log start of training
        self.logger.info("Starting training...")
        self.logger.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"Number of epochs: {self.config['training']['epochs']}")
        
        # Training loop
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_loss = self.validate(epoch)
            
            # Update best model
            is_best = val_loss < self.best_metric
            if is_best:
                self.best_metric = val_loss
                self._save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if epoch % self.config['training'].get('save_interval', 5) == 0 or epoch == self.config['training']['epochs']:
                self._save_checkpoint(epoch, is_best=is_best)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Best Val Loss: {self.best_metric:.4f} | LR: {self._get_learning_rate():.6f}"
            )
            
            # Early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        final_model_path = os.path.join(
            self.config['training']['output_dir'],
            f'model_final_epoch{self.current_epoch}.pt'
        )
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Log training completion
        total_time = time.time() - start_time
        self.logger.info(
            f"Training completed in {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s"
        )
        
        return self.model

    def _get_learning_rate(self) -> float:
        """Get the current learning rate."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        # Save checkpoint
        checkpoint_dir = self.config['training']['output_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
        torch.save(state, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'model_best.pt')
            torch.save(state, best_path)
            self.logger.info(f"New best model saved to {best_path} with loss: {self.best_metric:.4f}")
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(state, latest_path)
        
        # Clean up old checkpoints (keep only the last 3)
        if epoch > 3:
            old_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch-3}.pt')
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Document Layout Detection Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use for training (overrides config)')
    
    return parser.parse_args()


def main():
    """Main function for training."""
    # Parse command line arguments
    args = parse_args()
    trainer = None
    
    try:
        # Initialize trainer
        trainer = DocumentLayoutTrainer(config_path=args.config)
        
        # Override device if specified
        if args.device:
            trainer.device = torch.device(args.device)
            trainer.model = trainer.model.to(trainer.device)
        
        # Resume from checkpoint if specified
        if args.resume and os.path.isfile(args.resume):
            trainer.logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            
            # Load model state
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                if trainer.scheduler is not None:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            trainer.current_epoch = checkpoint.get('epoch', 0) + 1
            trainer.global_step = checkpoint.get('global_step', 0)
            trainer.metrics = checkpoint.get('metrics', {
                'train_loss': [],
                'val_loss': [],
                'learning_rates': []
            })
            trainer.best_metric = checkpoint.get('best_metric', float('inf'))
            
            trainer.logger.info(f"Resumed training from epoch {trainer.current_epoch}")
        
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        if trainer is not None and hasattr(trainer, 'logger'):
            trainer.logger.info("Training interrupted. Saving model...")
            if hasattr(trainer, 'current_epoch') and trainer.current_epoch > 0:  # Only save if training has started
                trainer._save_checkpoint(trainer.current_epoch, is_best=False)
        else:
            print("Training interrupted before trainer was fully initialized.")
        return 1
    except Exception as e:
        if trainer is not None and hasattr(trainer, 'logger'):
            trainer.logger.error(f"Training failed: {str(e)}", exc_info=True)
        else:
            print(f"Training failed before trainer was fully initialized: {str(e)}")
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
