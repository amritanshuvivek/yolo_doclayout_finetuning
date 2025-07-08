#!/bin/bash

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/{train,val}/{images,labels}
mkdir -p checkpoints logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set execute permissions for scripts
chmod +x train.py
chmod +x inference.py
chmod +x validate_dataset.py
chmod +x visualize_annotations.py

echo "Setup complete!"
echo "Next steps:"
echo "1. Place your training images in data/train/images/"
echo "2. Place your validation images in data/val/images/"
echo "3. Add corresponding label files in the labels/ directories"
echo "4. Run: python train.py"
