#!/bin/bash

# Run all tests
echo "Running dataset tests..."
python -m tests.test_dataset

echo -e "\nRunning model tests..."
python -m tests.test_model

echo -e "\nAll tests completed."
