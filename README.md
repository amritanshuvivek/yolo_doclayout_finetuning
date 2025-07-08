# Document Layout Detection Model Fine-tuning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

This project provides tools for fine-tuning a YOLOv10-based document layout detection model on custom datasets. The model can detect various document elements such as text, titles, figures, tables, and more.

## ✨ Features

- Fine-tune YOLOv10 on custom document layout datasets
- Support for various data augmentations
- Multiple learning rate schedulers
- Early stopping and model checkpointing
- Mixed precision training
- Comprehensive logging and visualization
- **Utilities**: Dataset validation and model evaluation
- **Developer Friendly**: Type hints, pre-commit hooks, and comprehensive tests

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- Linux/macOS (Windows support untested)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/layout-model-fine-tuning.git
cd layout-model-fine-tuning
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠 Development Setup

1. Install development dependencies:
```bash
pip install -r requirements.txt  # Includes dev dependencies
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest
```

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for static type checking
- **flake8** for linting

These are automatically checked by pre-commit hooks. You can also run them manually:

```bash
black .
isort .
flake8
mypy .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Pull Request Guidelines

- Ensure your code passes all tests and linting checks
- Add tests for new features
- Update documentation as needed
- Keep pull requests focused on a single feature or bug fix
- Write clear commit messages

## Dataset Preparation

The dataset should be organized in the following structure:
```
layout_model_fine_tuning/
├── data/
│   ├── train/
│   │   ├── images/
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   └── labels/
│   │       ├── img1.txt
│   │       ├── img2.txt
│   │       └── ...
│   └── val/
│       ├── images/
│       │   ├── img1.jpg
│       │   ├── img2.jpg
│       │   └── ...
│       └── labels/
│           ├── img1.txt
│           ├── img2.txt
│           └── ...
```

Each label file should contain annotations in YOLO format:
```
class_id x_center y_center width height
```
where coordinates are normalized (0-1)

## Training

To start training:
```bash
python train_layout_model.py
```

The training logs will be saved in the `logs/` directory and checkpoints will be saved in the `checkpoints/` directory.

## Configuration

The training configuration can be found in `config/training_config.yaml`. You can modify the following parameters:
- `num_classes`: Number of classes in your dataset
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimizer
- `save_interval`: Interval for saving checkpoints

## Monitoring

- Training progress and metrics will be logged in `logs/training.log`
- Checkpoint models will be saved in `checkpoints/` directory
- Validation metrics will be tracked during training
