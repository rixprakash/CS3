import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try importing packages and provide helpful error messages
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV (cv2) is not installed or not found in the Python path.")
    print("Please install it with: pip install opencv-python")
    print("If already installed, make sure you're using the correct Python environment.")
    exit(1)

try:
    import pandas as pd
    import seaborn as sns
except ImportError:
    print("ERROR: pandas or seaborn not installed.")
    print("Please install them with: pip install pandas seaborn")
    exit(1)

try:
    from scipy.fftpack import fft2, fftshift
except ImportError:
    print("ERROR: scipy not installed.")
    print("Please install it with: pip install scipy")
    exit(1)

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn not installed.")
    print("Please install it with: pip install scikit-learn")
    exit(1)

try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage.color import rgb2gray
except ImportError:
    print("ERROR: scikit-image not installed.")
    print("Please install it with: pip install scikit-image")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: tqdm not installed. Progress bars will not be shown.")
    # Create a simple replacement for tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Paths
DATA_DIR = '../DATA/DeepGuardDB_v1'
SD_DIR = os.path.join(DATA_DIR, 'SD_dataset')
DALLE_DIR = os.path.join(DATA_DIR, 'DALLE_dataset')
IMAGEN_DIR = os.path.join(DATA_DIR, 'IMAGEN_dataset')
GLIDE_DIR = os.path.join(DATA_DIR, 'GLIDE_dataset')
JSON_DIR = os.path.join(DATA_DIR, 'json_files')
OUTPUT_DIR = './advanced_viz_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sample_images(max_samples=100):
    """Load sample images from each category for analysis"""
    samples = {
        'real': [],
        'SD': [],
        'DALLE': [],
        'IMAGEN': [],
        'GLIDE': []
    }
    
    # Load real images
    real_paths = []
    for model_dir in [SD_DIR, DALLE_DIR, IMAGEN_DIR, GLIDE_DIR]:
        real_dir = os.path.join(model_dir, 'real')
        if os.path.exists(real_dir):
            paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            real_paths.extend(paths[:max_samples//4])  # Balance across directories
    
    # Subsample to max_samples
    if len(real_paths) > max_samples:
        np.random.shuffle(real_paths)
        real_paths = real_paths[:max_samples]
    
    for path in tqdm(real_paths, desc="Loading real images"):
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            samples['real'].append((img, path))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    # Load AI-generated images
    for model, model_dir in [('SD', SD_DIR), ('DALLE', DALLE_DIR), 
                             ('IMAGEN', IMAGEN_DIR), ('GLIDE', GLIDE_DIR)]:
        fake_dir = os.path.join(model_dir, 'fake')
        if os.path.exists(fake_dir):
            paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(paths) > max_samples:
                np.random.shuffle(paths)
                paths = paths[:max_samples]
            
            for path in tqdm(paths, desc=f"Loading {model} images"):
                try:
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    samples[model].append((img, path))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    print(f"Loaded samples: Real={len(samples['real'])}, SD={len(samples['SD'])}, "
          f"DALLE={len(samples['DALLE'])}, IMAGEN={len(samples['IMAGEN'])}, GLIDE={len(samples['GLIDE'])}")
    
    return samples

def visualize_color_distributions(samples):
    """
    Analyze and visualize color distributions across different image sources
    """
    plt.figure(figsize=(18, 12))
    color_channels = ['Red', 'Green', 'Blue']
    categories = list(samples.keys())
    
    for i, channel in enumerate(range(3)):
        plt.subplot(3, 1, i+1)
        
        for category in categories:
            channel_values = []
            for img, _ in samples[category]:
                # Extract the channel and flatten
                channel_data = img[:,:,channel].flatten()
                channel_values.extend(channel_data)
            
            # Sample to reduce computation if needed
            if len(channel_values) > 10000:
                channel_values = np.random.choice(channel_values, 10000, replace=False)
            
            # Plot histogram
            sns.kdeplot(channel_values, label=category)
        
        plt.title(f'{color_channels[channel]} Channel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'color_distributions.png'))
    plt.close()
    print("Color distribution visualization saved.")

def analyze_frequency_domain(samples):
    """
    Analyze images in frequency domain using FFT to detect patterns
    """
    plt.figure(figsize=(20, 15))
    categories = list(samples.keys())
    rows, cols = len(categories), 3
    
    # Calculate average magnitude spectrum for each category
    for i, category in enumerate(categories):
        magnitudes = []
        
        for img, _ in samples[category][:20]:  # Use a subset for faster computation
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply FFT
            f = fft2(gray)
            fshift = fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # Normalize
            magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / \
                               (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
            
            magnitudes.append(magnitude_spectrum)
        
        # Average magnitude
        avg_magnitude = np.mean(magnitudes, axis=0)
        
        # Plot
        plt.subplot(rows, cols, i*cols + 1)
        plt.imshow(avg_magnitude, cmap='viridis')
        plt.title(f'{category} - Avg Frequency Spectrum')
        plt.colorbar()
        
        # Plot horizontal profile (middle row)
        plt.subplot(rows, cols, i*cols + 2)
        middle_row = avg_magnitude.shape[0] // 2
        plt.plot(avg_magnitude[middle_row, :])
        plt.title(f'{category} - Horizontal Profile')
        
        # Plot vertical profile (middle column)
        plt.subplot(rows, cols, i*cols + 3)
        middle_col = avg_magnitude.shape[1] // 2
        plt.plot(avg_magnitude[:, middle_col])
        plt.title(f'{category} - Vertical Profile')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'frequency_domain_analysis.png'))
    plt.close()
    print("Frequency domain analysis visualization saved.")

def extract_texture_features(samples):
    """
    Extract and visualize GLCM texture features from images
    """
    # Features to extract
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    # Prepare data structure
    feature_data = []
    
    for category in samples:
        for img, path in tqdm(samples[category], desc=f"Extracting {category} texture features"):
            # Convert to grayscale
            gray = rgb2gray(img)
            
            # Rescale to 0-255 and convert to uint8
            gray = (gray * 255).astype(np.uint8)
            
            # Compute GLCM
            distances = [1, 3, 5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            try:
                glcm = graycomatrix(gray, distances=distances, angles=angles, 
                                   levels=256, symmetric=True, normed=True)
                
                # Calculate properties
                feature_dict = {
                    'category': category,
                    'file': os.path.basename(path)
                }
                
                for prop in properties:
                    feature_dict[prop] = np.mean(graycoprops(glcm, prop))
                
                feature_data.append(feature_dict)
            except Exception as e:
                print(f"Error extracting GLCM features for {path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_data)
    
    # Visualize
    fig, axes = plt.subplots(len(properties), 1, figsize=(15, 20))
    
    for i, prop in enumerate(properties):
        sns.boxplot(x='category', y=prop, data=df, ax=axes[i])
        axes[i].set_title(f'{prop.capitalize()} Distribution by Category')
        axes[i].set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'texture_features.png'))
    plt.close()
    
    # Create pair plot of features
    plt.figure(figsize=(15, 15))
    sns.pairplot(df, hue='category', vars=properties)
    plt.savefig(os.path.join(OUTPUT_DIR, 'texture_features_pairplot.png'))
    plt.close()
    
    print("Texture feature visualization saved.")
    
    return df

def edge_detection_analysis(samples):
    """
    Apply edge detection and analyze edge characteristics
    """
    plt.figure(figsize=(20, 15))
    categories = list(samples.keys())
    
    # Create subplots
    fig, axes = plt.subplots(len(categories), 3, figsize=(15, 5*len(categories)))
    
    # Edge detection parameters
    canny_thresholds = [(50, 150), (100, 200), (150, 250)]
    
    for i, category in enumerate(categories):
        # Use a representative sample
        edge_maps = []
        edge_densities = []
        
        # Process multiple images and average results
        for img, _ in samples[category][:10]:  # Use a subset
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection with medium threshold
            edges = cv2.Canny(gray, 100, 200)
            edge_maps.append(edges)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_densities.append(edge_density)
        
        # Average edge map
        avg_edge_map = np.mean(np.array(edge_maps), axis=0)
        
        # Display sample image, edge map, and edge density histogram
        if len(samples[category]) > 0:
            sample_img = samples[category][0][0]  # First sample image
            axes[i, 0].imshow(sample_img)
            axes[i, 0].set_title(f'{category} Sample')
        
        axes[i, 1].imshow(avg_edge_map, cmap='gray')
        axes[i, 1].set_title(f'{category} Avg Edge Map')
        
        axes[i, 2].hist(edge_densities, bins=10, alpha=0.7)
        axes[i, 2].set_title(f'{category} Edge Density')
        axes[i, 2].set_xlim(0, 0.5)  # Adjust as needed
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'edge_detection_analysis.png'))
    plt.close()
    
    # Plot edge density comparison
    plt.figure(figsize=(12, 8))
    edge_density_data = []
    
    for category in categories:
        for img, _ in samples[category][:20]:  # Use a subset
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_density_data.append({
                'category': category,
                'edge_density': edge_density
            })
    
    df = pd.DataFrame(edge_density_data)
    sns.boxplot(x='category', y='edge_density', data=df)
    plt.title('Edge Density Comparison')
    plt.savefig(os.path.join(OUTPUT_DIR, 'edge_density_comparison.png'))
    plt.close()
    
    print("Edge detection analysis visualization saved.")
    
    return df

def tsne_visualization(samples, texture_features_df=None):
    """
    Create t-SNE visualization of images based on their features
    """
    # Extract features from images
    features = []
    labels = []
    file_paths = []
    
    print("Extracting features for t-SNE visualization...")
    
    # First, try to use texture features if available
    if texture_features_df is not None:
        feature_cols = [col for col in texture_features_df.columns 
                        if col not in ['category', 'file']]
        
        X = texture_features_df[feature_cols].values
        y = texture_features_df['category'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create DataFrame for visualization
        tsne_df = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'category': y
        })
        
        # Visualize
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='category', data=tsne_df, palette='tab10', s=100, alpha=0.7)
        plt.title('t-SNE Visualization of Texture Features')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_texture_features.png'))
        plt.close()
        
        print("t-SNE visualization of texture features saved.")
    
    # If there's not enough texture feature data, extract simple features from images
    else:
        for category in samples:
            for img, path in tqdm(samples[category], desc=f"Processing {category}"):
                # Simple features: downsampled image, color histograms
                try:
                    # Resize for faster processing
                    small_img = cv2.resize(img, (32, 32))
                    
                    # Extract color histograms
                    hist_features = []
                    for i in range(3):  # RGB channels
                        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        hist_features.extend(hist)
                    
                    # Combine features
                    combined_features = np.concatenate([small_img.flatten(), np.array(hist_features)])
                    features.append(combined_features)
                    labels.append(category)
                    file_paths.append(path)
                except Exception as e:
                    print(f"Error extracting features from {path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create DataFrame for visualization
        tsne_df = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'category': y,
            'file': file_paths
        })
        
        # Visualize
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='category', data=tsne_df, palette='tab10', s=100, alpha=0.7)
        plt.title('t-SNE Visualization of Image Features')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_image_features.png'))
        plt.close()
        
        print("t-SNE visualization of image features saved.")

def noise_level_analysis(samples):
    """
    Analyze noise patterns in different image categories
    """
    # Calculate noise metrics for each image
    noise_data = []
    
    for category in samples:
        for img, path in tqdm(samples[category], desc=f"Analyzing noise in {category}"):
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Apply Gaussian blur to remove details (adjust kernel size as needed)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                
                # Calculate noise as difference between original and blurred
                noise = gray.astype(np.float32) - blur.astype(np.float32)
                
                # Calculate noise metrics
                noise_mean = np.mean(np.abs(noise))
                noise_std = np.std(noise)
                noise_energy = np.sum(noise**2) / (gray.shape[0] * gray.shape[1])
                
                noise_data.append({
                    'category': category,
                    'file': os.path.basename(path),
                    'noise_mean': noise_mean,
                    'noise_std': noise_std,
                    'noise_energy': noise_energy
                })
            except Exception as e:
                print(f"Error analyzing noise in {path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(noise_data)
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    metrics = ['noise_mean', 'noise_std', 'noise_energy']
    titles = ['Mean Absolute Noise', 'Noise Standard Deviation', 'Noise Energy']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.boxplot(x='category', y=metric, data=df, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'noise_analysis.png'))
    plt.close()
    
    # Create noise distribution plots
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        for category in df['category'].unique():
            category_data = df[df['category'] == category][metric]
            sns.kdeplot(category_data, label=category)
        plt.title(titles[i])
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'noise_distributions.png'))
    plt.close()
    
    print("Noise level analysis visualization saved.")
    
    return df

def main():
    print("Loading sample images...")
    samples = load_sample_images(max_samples=100)
    
    print("\n1. Visualizing color distributions...")
    visualize_color_distributions(samples)
    
    print("\n2. Analyzing frequency domain patterns...")
    analyze_frequency_domain(samples)
    
    print("\n3. Extracting and visualizing texture features...")
    texture_df = extract_texture_features(samples)
    
    print("\n4. Performing edge detection analysis...")
    edge_df = edge_detection_analysis(samples)
    
    print("\n5. Creating t-SNE visualization...")
    tsne_visualization(samples, texture_df)
    
    print("\nAll visualizations completed and saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main() 