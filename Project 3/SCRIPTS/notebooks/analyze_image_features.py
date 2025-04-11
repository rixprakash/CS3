import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import random

# Paths
DATA_DIR = '../DATA/DeepGuardDB_v1'
OUTPUT_DIR = './feature_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting image feature analysis...")

def load_images(source_type, max_samples=20):
    """Load sample images of the specified source type (real or specific AI model)"""
    image_paths = []
    
    if source_type == 'real':
        # Collect real images from all model directories
        for model in ['SD_dataset', 'DALLE_dataset', 'IMAGEN_dataset', 'GLIDE_dataset']:
            real_dir = os.path.join(DATA_DIR, model, 'real')
            if os.path.exists(real_dir):
                paths = glob.glob(os.path.join(real_dir, '*.jpg')) + \
                        glob.glob(os.path.join(real_dir, '*.jpeg')) + \
                        glob.glob(os.path.join(real_dir, '*.png'))
                image_paths.extend(paths)
    else:
        # Get fake images from the specific model directory
        fake_dir = os.path.join(DATA_DIR, f'{source_type}_dataset', 'fake')
        if os.path.exists(fake_dir):
            image_paths = glob.glob(os.path.join(fake_dir, '*.jpg')) + \
                          glob.glob(os.path.join(fake_dir, '*.jpeg')) + \
                          glob.glob(os.path.join(fake_dir, '*.png'))
    
    # Randomly sample to max_samples
    if len(image_paths) > max_samples:
        image_paths = random.sample(image_paths, max_samples)
    
    print(f"Loading {len(image_paths)} images for {source_type}")
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append((np.array(img), path))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return images

def analyze_color_histograms(samples):
    """Analyze color distributions across different image sources"""
    plt.figure(figsize=(15, 10))
    categories = list(samples.keys())
    colors = ['r', 'g', 'b']
    
    for i, color_name in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(3, 1, i+1)
        
        for category in categories:
            all_values = []
            for img, _ in samples[category]:
                if len(img.shape) == 3 and img.shape[2] >= 3:  # Ensure RGB
                    channel_values = img[:,:,i].flatten()
                    # Sample to reduce computation
                    if len(channel_values) > 1000:
                        indices = np.random.choice(len(channel_values), 1000, replace=False)
                        channel_values = channel_values[indices]
                    all_values.extend(channel_values)
            
            plt.hist(all_values, bins=50, alpha=0.5, density=True, label=category)
        
        plt.title(f'{color_name} Channel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'color_histograms.png'))
    plt.close()
    print("Color histogram analysis completed")

def analyze_image_dimensions(samples):
    """Analyze image dimensions and aspect ratios"""
    widths = {category: [] for category in samples.keys()}
    heights = {category: [] for category in samples.keys()}
    aspect_ratios = {category: [] for category in samples.keys()}
    
    for category in samples:
        for img, _ in samples[category]:
            if len(img.shape) == 3:
                h, w = img.shape[0], img.shape[1]
            else:
                h, w = img.shape
            
            widths[category].append(w)
            heights[category].append(h)
            aspect_ratios[category].append(w/h)
    
    # Plot dimensions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for category in samples:
        plt.scatter(widths[category], heights[category], alpha=0.7, label=category)
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Image Dimensions by Category')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for category in samples:
        plt.hist(aspect_ratios[category], bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Frequency')
    plt.title('Aspect Ratio Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_dimensions.png'))
    plt.close()
    print("Image dimension analysis completed")

def analyze_image_complexity(samples):
    """Analyze image complexity using simple metrics"""
    complexity_metrics = {category: [] for category in samples.keys()}
    
    for category in samples:
        for img, _ in samples[category]:
            try:
                # Convert to grayscale if needed
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    # Simple grayscale conversion
                    gray = np.mean(img[:,:,:3], axis=2).astype(np.uint8)
                else:
                    gray = img
                
                # Calculate basic entropy (std dev as simple measure)
                complexity = np.std(gray)
                
                # Calculate gradient magnitude (simple edge measure)
                gy, gx = np.gradient(gray.astype(float))
                edge_magnitude = np.sqrt(gx**2 + gy**2)
                edge_mean = np.mean(edge_magnitude)
                
                complexity_metrics[category].append((complexity, edge_mean))
            except Exception as e:
                print(f"Error analyzing complexity: {e}")
    
    # Plot complexity metrics
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    for category in samples:
        values = [x[0] for x in complexity_metrics[category]]
        plt.hist(values, bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Standard Deviation (Complexity)')
    plt.ylabel('Frequency')
    plt.title('Image Complexity Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for category in samples:
        values = [x[1] for x in complexity_metrics[category]]
        plt.hist(values, bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Mean Edge Magnitude')
    plt.ylabel('Frequency')
    plt.title('Edge Complexity Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_complexity.png'))
    plt.close()
    print("Image complexity analysis completed")

def analyze_noise_patterns(samples):
    """Analyze noise patterns in different categories"""
    noise_metrics = {category: [] for category in samples.keys()}
    
    for category in samples:
        for img, _ in samples[category]:
            try:
                # Convert to grayscale if needed
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    # Simple grayscale conversion
                    gray = np.mean(img[:,:,:3], axis=2).astype(np.uint8)
                else:
                    gray = img
                
                # Apply simple box blur
                from scipy.ndimage import uniform_filter
                blurred = uniform_filter(gray, size=5)
                
                # Calculate noise as difference between original and blurred
                noise = gray.astype(float) - blurred.astype(float)
                
                # Calculate noise metrics
                noise_mean = np.mean(np.abs(noise))
                noise_std = np.std(noise)
                
                noise_metrics[category].append((noise_mean, noise_std))
            except Exception as e:
                print(f"Error analyzing noise: {e}")
    
    # Plot noise metrics
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    for category in samples:
        values = [x[0] for x in noise_metrics[category]]
        plt.hist(values, bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Mean Absolute Noise')
    plt.ylabel('Frequency')
    plt.title('Noise Level Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for category in samples:
        values = [x[1] for x in noise_metrics[category]]
        plt.hist(values, bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Noise Variability Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'noise_patterns.png'))
    plt.close()
    print("Noise pattern analysis completed")

def analyze_local_patterns(samples):
    """Analyze local texture patterns in different categories"""
    pattern_metrics = {category: [] for category in samples.keys()}
    
    for category in samples:
        for img, _ in samples[category]:
            try:
                # Convert to grayscale if needed
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    # Simple grayscale conversion
                    gray = np.mean(img[:,:,:3], axis=2).astype(np.uint8)
                else:
                    gray = img
                
                # Resize for faster processing
                from PIL import Image
                gray_img = Image.fromarray(gray)
                resized = gray_img.resize((100, 100))
                gray = np.array(resized)
                
                # Calculate local binary pattern-like features
                # (simplified version - just looking at local differences)
                local_diff = np.zeros_like(gray, dtype=float)
                
                for i in range(1, gray.shape[0]-1):
                    for j in range(1, gray.shape[1]-1):
                        # Get the center pixel and its neighbors
                        center = gray[i, j]
                        neighbors = [
                            gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                            gray[i, j-1], gray[i, j+1],
                            gray[i+1, j-1], gray[i+1, j], gray[i+1, j+1]
                        ]
                        # Calculate the mean absolute difference
                        local_diff[i, j] = np.mean([abs(n - center) for n in neighbors])
                
                # Calculate metrics on local differences
                local_mean = np.mean(local_diff)
                local_std = np.std(local_diff)
                
                pattern_metrics[category].append((local_mean, local_std))
            except Exception as e:
                print(f"Error analyzing local patterns: {e}")
    
    # Plot local pattern metrics
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    for category in samples:
        values = [x[0] for x in pattern_metrics[category]]
        plt.hist(values, bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Mean Local Difference')
    plt.ylabel('Frequency')
    plt.title('Local Texture Pattern Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for category in samples:
        values = [x[1] for x in pattern_metrics[category]]
        plt.hist(values, bins=20, alpha=0.5, density=True, label=category)
    plt.xlabel('Standard Deviation of Local Differences')
    plt.ylabel('Frequency')
    plt.title('Local Pattern Variability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'local_patterns.png'))
    plt.close()
    print("Local pattern analysis completed")

def main():
    # Define categories to analyze
    categories = ['real', 'SD', 'DALLE', 'IMAGEN', 'GLIDE']
    
    # Load sample images for each category
    samples = {}
    for category in categories:
        if category == 'real':
            samples[category] = load_images('real', max_samples=50)
        else:
            samples[category] = load_images(category, max_samples=20)
    
    # Run analyses
    print("\n1. Analyzing color histograms...")
    analyze_color_histograms(samples)
    
    print("\n2. Analyzing image dimensions and aspect ratios...")
    analyze_image_dimensions(samples)
    
    print("\n3. Analyzing image complexity...")
    analyze_image_complexity(samples)
    
    print("\n4. Analyzing noise patterns...")
    analyze_noise_patterns(samples)
    
    print("\n5. Analyzing local texture patterns...")
    analyze_local_patterns(samples)
    
    print("\nAll analyses completed. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main() 