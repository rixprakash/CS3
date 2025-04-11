#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis for DeepGuardDB Dataset
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
from collections import Counter

# Set plot style
plt.style.use('ggplot')
sns.set_palette('bright')

# Define path to dataset
base_path = "../DATA/DeepGuardDB_v1/"
output_dir = "eda_results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image directories
dataset_models = ['SD_dataset', 'DALLE_dataset', 'IMAGEN_dataset', 'GLIDE_dataset']
categories = ['real', 'fake']

def count_images(path):
    """Count number of images in a directory"""
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])

def analyze_dataset_structure():
    """Analyze and visualize dataset structure"""
    print("Analyzing dataset structure...")
    
    # Create a dataframe with image counts
    counts = []
    for model in dataset_models:
        for category in categories:
            path = os.path.join(base_path, model, category)
            count = count_images(path)
            counts.append({
                'Model': model.split('_')[0], 
                'Category': category, 
                'Count': count
            })
    
    count_df = pd.DataFrame(counts)
    print(count_df)
    
    # Visualize image counts
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Count', hue='Category', data=count_df)
    plt.title('Number of Images by Model and Category', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    
    # Add counts above bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():,}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_structure.png'), dpi=300)
    plt.close()

def analyze_metadata():
    """Analyze dataset metadata"""
    print("Analyzing metadata...")
    
    # Load metadata from the main JSON file
    try:
        json_path = os.path.join(base_path, 'json_files', 'DeepGuardDB.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            
        # Convert to DataFrame
        metadata_df = pd.DataFrame(metadata)
        print(f"Total number of entries: {len(metadata_df)}")
        
        # Platform distribution
        platform_counts = metadata_df['platform'].value_counts()
        print("\nDistribution by platform:")
        for platform, count in platform_counts.items():
            print(f"{platform}: {count} ({count/len(metadata_df)*100:.2f}%)")
        
        # Visualize distribution by platform
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='platform', data=metadata_df, palette='viridis')
        plt.title('Distribution of Images by Platform', fontsize=16)
        plt.xlabel('Platform', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add counts above bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():,}\n({p.get_height()/len(metadata_df)*100:.1f}%)', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'platform_distribution.png'), dpi=300)
        plt.close()
        
        return metadata_df
    except Exception as e:
        print(f"Error analyzing metadata: {e}")
        return None

def tokenize_text(text):
    """Simple tokenization for prompts"""
    if not isinstance(text, str):
        return []
    
    # Split by spaces and remove punctuation
    tokens = text.lower().split()
    tokens = [token.strip('.,;:!?()-"\'') for token in tokens]
    
    # Remove empty tokens and common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'of'}
    tokens = [token for token in tokens if token and token not in stop_words]
    
    return tokens

def analyze_prompts(metadata_df):
    """Analyze prompts used to generate the fake images"""
    if metadata_df is None:
        return
    
    print("Analyzing prompts...")
    
    # Combine all prompts
    all_tokens = []
    for prompt in metadata_df['prompts']:
        all_tokens.extend(tokenize_text(prompt))
    
    # Count word frequencies
    word_freq = Counter(all_tokens)
    most_common_words = word_freq.most_common(20)
    
    # Create dataframe for visualization
    word_freq_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    word_freq_df = word_freq_df.sort_values('Frequency')
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Word', data=word_freq_df, palette='viridis')
    plt.title('Most Common Words in Image Generation Prompts', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Word', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'common_words.png'), dpi=300)
    plt.close()
    
    # Analyze prompts by platform
    platform_tokens = {}
    for platform in metadata_df['platform'].unique():
        platform_prompts = metadata_df[metadata_df['platform'] == platform]['prompts']
        tokens = []
        for prompt in platform_prompts:
            tokens.extend(tokenize_text(prompt))
        platform_tokens[platform] = tokens
    
    # Get top words by platform
    top_words_by_platform = {}
    for platform, tokens in platform_tokens.items():
        counter = Counter(tokens)
        top_words_by_platform[platform] = counter.most_common(10)
    
    # Plot top words by platform
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (platform, words) in enumerate(top_words_by_platform.items()):
        if i < 4:  # Ensure we don't exceed available axes
            df = pd.DataFrame(words, columns=['Word', 'Frequency']).sort_values('Frequency')
            sns.barplot(x='Frequency', y='Word', data=df, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Top Words in {platform.upper()} Prompts', fontsize=14)
            axes[i].set_xlabel('Frequency', fontsize=12)
            axes[i].set_ylabel('Word', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'platform_words.png'), dpi=300)
    plt.close()

def display_sample_images():
    """Display sample images from each model and category"""
    print("Generating sample image visualizations...")
    
    for model in dataset_models:
        try:
            real_path = os.path.join(base_path, f'{model}/real')
            fake_path = os.path.join(base_path, f'{model}/fake')
            
            if not os.path.exists(real_path) or not os.path.exists(fake_path):
                continue
                
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Skip if either category has no images
            if not real_files or not fake_files:
                continue
                
            # Sample at most 4 images from each category
            n_samples = min(4, len(real_files), len(fake_files))
            real_samples = random.sample(real_files, n_samples)
            fake_samples = random.sample(fake_files, n_samples)
            
            # Create figure
            fig, axes = plt.subplots(2, n_samples, figsize=(16, 8))
            
            # Plot real images
            for i, img_file in enumerate(real_samples):
                img_path = os.path.join(real_path, img_file)
                try:
                    img = Image.open(img_path)
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'Real Image', fontsize=10)
                    axes[0, i].axis('off')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            # Plot fake images
            for i, img_file in enumerate(fake_samples):
                img_path = os.path.join(fake_path, img_file)
                try:
                    img = Image.open(img_path)
                    axes[1, i].imshow(img)
                    axes[1, i].set_title(f'Fake Image ({model.split("_")[0]})', fontsize=10)
                    axes[1, i].axis('off')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            plt.suptitle(f'Sample Images from {model.split("_")[0]} Dataset', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'samples_{model.split("_")[0]}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error generating sample visualization for {model}: {e}")

def analyze_image_properties():
    """Analyze image dimensions, aspect ratios, and file sizes"""
    print("Analyzing image properties...")
    
    # Function to get image properties
    def get_image_properties(img_path):
        try:
            img = Image.open(img_path)
            width, height = img.size
            aspect_ratio = width / height
            file_size = os.path.getsize(img_path) / 1024  # in KB
            return {
                'width': width, 
                'height': height, 
                'aspect_ratio': aspect_ratio, 
                'file_size': file_size
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    # Sample images to analyze (limit to 50 per category for performance)
    image_properties = []
    for model in dataset_models:
        for category in categories:
            path = os.path.join(base_path, model, category)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                sample_size = min(50, len(files))
                
                if sample_size > 0:
                    sample_files = random.sample(files, sample_size)
                    
                    for file in sample_files:
                        img_path = os.path.join(path, file)
                        props = get_image_properties(img_path)
                        if props:
                            props['model'] = model.split('_')[0]
                            props['category'] = category
                            image_properties.append(props)
    
    # Convert to DataFrame
    properties_df = pd.DataFrame(image_properties)
    if properties_df.empty:
        print("No image properties collected.")
        return
    
    print(f"Collected properties for {len(properties_df)} images.")
    
    # Distribution of image dimensions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Width distribution
    sns.histplot(data=properties_df, x='width', hue='category', bins=20, kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Image Widths', fontsize=14)
    axes[0].set_xlabel('Width (pixels)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    
    # Height distribution
    sns.histplot(data=properties_df, x='height', hue='category', bins=20, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Image Heights', fontsize=14)
    axes[1].set_xlabel('Height (pixels)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_dimensions.png'), dpi=300)
    plt.close()
    
    # Aspect ratio distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='aspect_ratio', hue='category', data=properties_df)
    plt.title('Aspect Ratio Distribution by Model and Category', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Aspect Ratio (width/height)', fontsize=14)
    plt.axhline(y=1, color='red', linestyle='--')
    plt.ylim(0.5, 2.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_ratios.png'), dpi=300)
    plt.close()
    
    # File size distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='file_size', hue='category', data=properties_df)
    plt.title('File Size Distribution by Model and Category', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('File Size (KB)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'file_sizes.png'), dpi=300)
    plt.close()

def main():
    """Main function to run all analyses"""
    print("Starting DeepGuardDB Exploratory Data Analysis")
    
    # Run all analyses
    analyze_dataset_structure()
    metadata_df = analyze_metadata()
    analyze_prompts(metadata_df)
    display_sample_images()
    analyze_image_properties()
    
    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 