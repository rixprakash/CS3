# Feature Extraction Module for AI Image Detection

This module implements comprehensive feature extraction for distinguishing between real photographs and AI-generated images based on the findings from our exploratory data analysis.

## Overview

The feature extraction module extracts five key categories of discriminative features:

1. **Metadata Features**
   - Image dimensions, aspect ratios, and size ratios
   - These capture the tendency of AI generators to produce images with standard dimensions

2. **Color Distribution Features**
   - Statistical measures of color channels: mean, standard deviation, skewness, kurtosis
   - Color channel correlations
   - These capture differences in how AI models handle color spaces compared to real cameras

3. **Image Complexity Features**
   - Edge density, entropy, spatial complexity
   - Corner detection metrics
   - These measure the natural complexity gradients found in real photos vs. AI-generated images

4. **Noise Pattern Features**
   - Noise level, pattern consistency, spectrum analysis
   - These identify the characteristic noise "fingerprints" of different AI generators, which differ significantly from real camera noise

5. **Local Texture Features**
   - Local Binary Pattern (LBP) histograms
   - These capture how pixel neighborhoods relate to each other, revealing anomalies in AI-rendered textures

6. **Frequency Domain Features**
   - Fast Fourier Transform (FFT) analysis
   - Frequency band energy distribution
   - These reveal patterns in the frequency domain that are often invisible in the spatial domain

## Usage

### Basic Usage

```python
from feature_extraction import ImageFeatureExtractor

# Create a feature extractor
extractor = ImageFeatureExtractor()

# Extract features from an image
import cv2
image = cv2.imread('path/to/image.jpg')
features = extractor.extract_features(image)

# Print extracted features
for feature_name, value in features.items():
    print(f"{feature_name}: {value}")
```

### Batch Processing

```python
from feature_extraction import extract_all_features

# Extract features from all images in a directory
features_df = extract_all_features(
    data_dir='path/to/images',
    output_file='features.csv',
    n_workers=4  # Number of parallel workers
)
```

### Testing the Features

```bash
python test_feature_extraction.py --data_dir /path/to/dataset --samples 10
```

This will:
1. Randomly sample images from each generator and category
2. Extract all features from these samples
3. Generate comparison plots between real and AI-generated images
4. Save the results in the feature_test_results directory

## Feature Details

### Metadata Features
- `width`: Image width in pixels
- `height`: Image height in pixels
- `aspect_ratio`: Width divided by height
- `size_ratio`: Size relative to 1MP

### Color Features
- `color_std_r/g/b`: Standard deviation of each color channel
- `color_mean_r/g/b`: Mean value of each color channel
- `color_skew_r/g/b`: Skewness of each color channel
- `color_kurtosis_r/g/b`: Kurtosis of each color channel
- `color_correlation_rg/rb/gb`: Correlation between color channels

### Complexity Features
- `edge_density`: Proportion of edge pixels in the image
- `entropy`: Shannon entropy of the image
- `std_dev`: Standard deviation of pixel values
- `spatial_complexity`: Mean gradient magnitude
- `corner_density`: Density of corner features

### Noise Features
- `noise_level`: Overall amount of noise
- `noise_pattern_consistency`: Consistency of noise patterns across the image
- `noise_spectrum_ratio`: Ratio of high to low frequency components in noise
- `noise_structure`: Auto-correlation measurement of noise

### Texture Features
- `lbp_hist_X`: Bins of the Local Binary Pattern histogram

### Frequency Features
- `fft_ring_X`: Energy in concentric frequency bands
- `fft_std`: Standard deviation of frequency spectrum
- `fft_skew`: Skewness of frequency spectrum
- `fft_kurtosis`: Kurtosis of frequency spectrum
- `fft_energy`: Total energy in frequency domain

## Integration with Models

These features can be used in three ways:

1. **Feature-Only Models**: Train machine learning models directly on the extracted features
2. **CNN-Only Models**: Use traditional deep learning on the raw images
3. **Hybrid Models**: Combine both approaches by feeding both raw images and extracted features into a neural network

## References

This feature extraction approach is based on our exploratory data analysis findings, documented in:
- `EDA_Summary.md`: Basic EDA of the DeepGuardDB dataset
- `Advanced_Visualizations_Summary.md`: Advanced analysis of visual patterns in real vs. AI-generated images 