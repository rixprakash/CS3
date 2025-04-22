# AI Image Detection Project

## Overview
This project implements a system for detecting AI-generated images using a combination of deep learning (CNN) and handcrafted image feature extraction. The system extracts a comprehensive set of features from images and employs multiple models to classify images as real or AI-generated.

## Key Components

### 1. Feature Extraction (`feature_extraction.py`)
- Extracts multiple categories of discriminative features:
  - Metadata features (dimensions, aspect ratios)
  - Color distribution patterns
  - Image complexity and edge characteristics
  - Noise patterns
  - Local texture patterns
  - Frequency domain features

### 2. Data Processing
- `preprocessing.py`: Handles image loading, normalization, and splitting
- `prepare_combined_features.py`: Combines and processes extracted features

### 3. Models
- CNN model: Based on EfficientNetB0, focused on visual patterns
- Feature-based models: Random Forest, Gradient Boosting, SVM
- Hybrid model: Combines CNN and handcrafted features

## Issue Fixed: Image Format for Edge Detection
We fixed an issue in the feature extraction process where Canny edge detection was failing due to incorrect image format. The Canny edge detection algorithm requires 8-bit unsigned integer images (CV_8U), but our grayscale conversion wasn't explicitly enforcing this format.

Solution:
```python
# Convert to grayscale for certain calculations
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ensure the image is 8-bit unsigned integer format for Canny
gray_uint8 = cv2.convertScaleAbs(gray)

# Edge detection using Canny
edges = cv2.Canny(gray_uint8, 100, 200)
```

## Workflow
1. Extract features from both real and AI-generated images
2. Combine features and split into train/validation/test sets
3. Train multiple models and evaluate their performance
4. Compare models to determine the most effective approach

## Usage

### Feature Extraction
```bash
python feature_extraction.py --data_dir DATA/path/to/images --output features.csv --workers 4
```

### Prepare Features for Model Training
```bash
python prepare_combined_features.py --real_features real_features.csv --fake_features fake_features.csv --output_dir prepared_data
```

### Train Feature-Based Model
```bash
python train_feature_model.py --data_dir prepared_data --output_dir model_results --model_type random_forest
```

### Train All Models
```bash
python main.py --model_type all --features_file combined_features.csv --output_dir final_results
```

## Team Members
- James Kulp (Leader)
- Rix Prakash
- Aymen Zouari

## Dataset
The project uses the DeepGuardDB dataset, which contains pairs of real and AI-generated images. Due to the large size of the dataset, the image files are not included in the git repository. To set up the dataset:

1. Download the DeepGuardDB dataset from IEEE Dataport:
   - URL: https://ieee-dataport.org/documents/deepguarddb-real-and-text-image-synthetic-images-dataset
   - Note: You'll need an IEEE Dataport account to download the dataset

2. Place the downloaded dataset in the following structure:
```
Project 3/
└── DATA/
    └── DeepGuardDB_v1/
        ├── DALLE_dataset/
        │   ├── fake/
        │   └── real/
        ├── GLIDE_dataset/
        │   ├── fake/
        │   └── real/
        ├── IMAGEN_dataset/
        │   ├── fake/
        │   └── real/
        ├── SD_dataset/
        │   ├── fake/
        │   └── real/
        └── json_files/
```

## Project Structure
```
Project 3/
├── DATA/              # Dataset storage (not in git)
├── SCRIPTS/           # Python scripts and notebooks
│   ├── EDAcode3.ipynb # Exploratory Data Analysis
│   ├── preprocessing.py
│   ├── model.py
│   └── organize_data.py
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup Instructions
1. Clone the repository:
```bash
git clone [repository-url]
cd "Project 3"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and set up the dataset as described in the Dataset section above.

5. Run the data organization script:
```bash
python SCRIPTS/organize_data.py
```

## Project Goals
- Develop a CNN-based model using EfficientNet architecture
- Implement frequency domain analysis for enhanced detection
- Achieve at least 80% accuracy in classification
- Evaluate performance across different image categories and generation techniques

## References
1. Y. Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple Domains," IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020
2. H. Ajder et al., "The State of Deepfakes: Landscape, Threats, and Impact," Deeptrace Labs, 2019
3. F. Marra et al., "Detection of GAN-Generated Fake Images over Social Networks," IEEE Conference on Multimedia Information Processing and Retrieval, 2019
4. DeepGuardDB Dataset, IEEE Dataport, 2023
5. M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019 