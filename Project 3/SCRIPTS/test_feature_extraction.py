import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_extraction import ImageFeatureExtractor, extract_features_from_image_path
import cv2
import random
from pathlib import Path

def plot_feature_comparison(real_features, fake_features, feature_name, title, output_dir=None):
    """
    Plot comparison between real and fake image features.
    
    Args:
        real_features (list): List of feature values from real images
        fake_features (list): List of feature values from fake images
        feature_name (str): Name of the feature
        title (str): Plot title
        output_dir (str): Directory to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Create histograms
    bins = np.linspace(
        min(min(real_features), min(fake_features)),
        max(max(real_features), max(fake_features)),
        30
    )
    
    plt.hist(real_features, bins=bins, alpha=0.5, label='Real', color='blue')
    plt.hist(fake_features, bins=bins, alpha=0.5, label='AI-Generated', color='red')
    
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{feature_name.replace('/', '_')}.png"))
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test feature extraction on sample images")
    parser.add_argument("--data_dir", default=None, help="Path to the DeepGuardDB dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample images per category")
    parser.add_argument("--output_dir", default="feature_test_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (one level up from SCRIPTS)
    project_dir = os.path.dirname(script_dir)
    
    # If data_dir is not provided, use the default
    if args.data_dir is None:
        args.data_dir = os.path.join(project_dir, 'DATA', 'DeepGuardDB_v1')
    
    output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature extractor
    extractor = ImageFeatureExtractor()
    
    # List of available generators
    generators = ['DALLE_dataset', 'SD_dataset', 'IMAGEN_dataset', 'GLIDE_dataset']
    
    all_real_features = []
    all_fake_features = []
    
    # Process samples from each generator
    for generator in generators:
        generator_dir = os.path.join(args.data_dir, generator)
        
        if not os.path.exists(generator_dir):
            print(f"Generator directory not found: {generator_dir}")
            continue
        
        # Get real and fake image directories
        real_dir = os.path.join(generator_dir, 'real')
        fake_dir = os.path.join(generator_dir, 'fake')
        
        # Get sample images
        real_images = []
        fake_images = []
        
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            real_images = [os.path.join(real_dir, f) for f in random.sample(real_files, min(args.samples, len(real_files)))]
        
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            fake_images = [os.path.join(fake_dir, f) for f in random.sample(fake_files, min(args.samples, len(fake_files)))]
        
        print(f"\nProcessing {generator}:")
        print(f"  - Real images: {len(real_images)}")
        print(f"  - Fake images: {len(fake_images)}")
        
        # Extract features
        for img_path in real_images:
            features = extract_features_from_image_path(img_path, extractor)
            if features:
                features['category'] = 'real'
                features['generator'] = generator
                all_real_features.append(features)
        
        for img_path in fake_images:
            features = extract_features_from_image_path(img_path, extractor)
            if features:
                features['category'] = 'fake'
                features['generator'] = generator
                all_fake_features.append(features)
    
    # Combine all features
    all_features = all_real_features + all_fake_features
    if not all_features:
        print("No features extracted. Check your dataset paths.")
        return
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save the features
    features_csv = os.path.join(output_dir, "sample_features.csv")
    features_df.to_csv(features_csv, index=False)
    print(f"\nSaved sample features to {features_csv}")
    
    # Generate comparison plots for a few key features
    print("\nGenerating feature comparison plots...")
    
    # Select interesting features from each category
    plot_features = [
        # Metadata features
        'aspect_ratio', 'size_ratio',
        # Color features
        'color_std_r', 'color_correlation_rg',
        # Complexity features
        'edge_density', 'entropy', 'spatial_complexity',
        # Noise features
        'noise_level', 'noise_spectrum_ratio',
        # Frequency features
        'fft_std', 'fft_energy'
    ]
    
    # Plot selected features
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for feature in plot_features:
        if feature in features_df.columns:
            real_values = features_df[features_df['category'] == 'real'][feature].dropna().tolist()
            fake_values = features_df[features_df['category'] == 'fake'][feature].dropna().tolist()
            
            if real_values and fake_values:
                plot_feature_comparison(
                    real_values, 
                    fake_values, 
                    feature, 
                    f"{feature} Distribution: Real vs AI-Generated",
                    plots_dir
                )
    
    # Display some sample images with their features
    sample_images_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    for idx, row in features_df.head(10).iterrows():
        img_path = row['image_path']
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (512, 512))
            category = row['category']
            generator = row['generator']
            
            # Create a figure with image and key features
            plt.figure(figsize=(12, 8))
            
            # Display the image
            plt.subplot(1, 2, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"{category.capitalize()} Image ({Path(generator).stem})")
            plt.axis('off')
            
            # Display key features
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.text(0.1, 0.95, "Key Features:", fontsize=14, fontweight='bold')
            
            y_pos = 0.9
            for feature in plot_features:
                if feature in row:
                    plt.text(0.1, y_pos, f"{feature}: {row[feature]:.4f}", fontsize=10)
                    y_pos -= 0.05
            
            # Save the figure
            img_name = Path(img_path).stem
            output_path = os.path.join(sample_images_dir, f"{category}_{generator.split('_')[0]}_{img_name}.png")
            plt.savefig(output_path)
            plt.close()
    
    print(f"\nSample images with features saved to {sample_images_dir}")
    print(f"Feature comparison plots saved to {plots_dir}")
    print("\nDone!")

if __name__ == "__main__":
    main() 