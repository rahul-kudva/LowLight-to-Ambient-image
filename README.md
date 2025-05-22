# Low-Light to Ambient Image Converter

This project enhances low-light images by converting them into well-lit, ambient-style images using a deep learning model.

## Model Training and Testing Guide

### Step 1: Prepare the Dataset

Organize your dataset in the following structure:

Data/
├── High/   # Ground truth ambient images
└── Low/    # Corresponding low-light input images

Do the same for your test dataset.

### Step 2: Preprocess the Dataset

1. Open data_process.py
2. Update the folder paths to match your dataset
3. Run the script to generate a .npz file from your images

### Step 3: Train and Test the Model

1. Open lod.py
2. Replace the dataset file path with your generated .npz file
3. Run the script to train and evaluate the model
   (Both training and testing logic are included in the same file)

## Running with Pretrained Model

To test using the provided pretrained model:

1. Navigate to the /Code directory
2. Ensure the following files are present:
   - test_dataset.npz
   - gen_model_009680.h5 (pretrained model)
3. Run the script:

   ./run.sh

## Provided Files

- test_dataset.npz: Sample test dataset
- gen_model_009680.h5: Pretrained model

These can be used directly with run.sh for quick testing or demonstration.
