import numpy as np
import cv2
from skimage import feature
from scipy import fftpack
from scipy import stats
import os
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from concurrent.futures import ThreadPoolExecutor

class ImageFeatureExtractor:
    """
    A comprehensive feature extractor for distinguishing real from AI-generated images.
    
    This class implements extraction of five key discriminative feature categories:
    1. Metadata features (dimensions, aspect ratios)
    2. Color distribution patterns
    3. Image complexity and edge characteristics
    4. Noise patterns
    5. Local texture patterns
    
    Both spatial and frequency domain analyses are included.
    """
    
    def __init__(self, include_metadata=True, include_color=True, 
                 include_complexity=True, include_noise=True, 
                 include_texture=True, include_frequency=True):
        """
        Initialize the feature extractor with options to include/exclude feature types.
        
        Args:
            include_metadata (bool): Whether to include metadata features
            include_color (bool): Whether to include color distribution features
            include_complexity (bool): Whether to include complexity features
            include_noise (bool): Whether to include noise pattern features
            include_texture (bool): Whether to include texture features
            include_frequency (bool): Whether to include frequency domain features
        """
        self.include_metadata = include_metadata
        self.include_color = include_color
        self.include_complexity = include_complexity
        self.include_noise = include_noise
        self.include_texture = include_texture
        self.include_frequency = include_frequency
        
        # Feature names for each category
        self.metadata_features = ['width', 'height', 'aspect_ratio', 'size_ratio']
        self.color_features = ['color_std_r', 'color_std_g', 'color_std_b', 
                              'color_mean_r', 'color_mean_g', 'color_mean_b',
                              'color_skew_r', 'color_skew_g', 'color_skew_b',
                              'color_kurtosis_r', 'color_kurtosis_g', 'color_kurtosis_b',
                              'color_correlation_rg', 'color_correlation_rb', 'color_correlation_gb']
        self.complexity_features = ['edge_density', 'entropy', 'std_dev', 
                                   'spatial_complexity', 'corner_density']
        self.noise_features = ['noise_level', 'noise_pattern_consistency', 
                              'noise_spectrum_ratio', 'noise_structure']
        self.texture_features = ['lbp_hist_' + str(i) for i in range(26)]
        self.frequency_features = ['fft_ring_' + str(i) for i in range(10)] + \
                                 ['fft_std', 'fft_skew', 'fft_kurtosis', 'fft_energy']
        
    def get_feature_names(self):
        """
        Get the names of all features that will be extracted.
        
        Returns:
            list: Names of all features
        """
        feature_names = []
        if self.include_metadata:
            feature_names.extend(self.metadata_features)
        if self.include_color:
            feature_names.extend(self.color_features)
        if self.include_complexity:
            feature_names.extend(self.complexity_features)
        if self.include_noise:
            feature_names.extend(self.noise_features)
        if self.include_texture:
            feature_names.extend(self.texture_features)
        if self.include_frequency:
            feature_names.extend(self.frequency_features)
        return feature_names
        
    def extract_features(self, image):
        """
        Extract all enabled features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Extract each feature type if enabled
        if self.include_metadata:
            metadata_feats = self.extract_metadata_features(image)
            features.update(metadata_feats)
            
        if self.include_color:
            color_feats = self.extract_color_features(image)
            features.update(color_feats)
            
        if self.include_complexity:
            complexity_feats = self.extract_complexity_features(image)
            features.update(complexity_feats)
            
        if self.include_noise:
            noise_feats = self.extract_noise_features(image)
            features.update(noise_feats)
            
        if self.include_texture:
            texture_feats = self.extract_texture_features(image)
            features.update(texture_feats)
            
        if self.include_frequency:
            freq_feats = self.extract_frequency_features(image)
            features.update(freq_feats)
            
        return features
    
    def extract_metadata_features(self, image):
        """
        Extract metadata features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of metadata features
        """
        height, width = image.shape[:2]
        aspect_ratio = width / height
        size_ratio = (width * height) / (1000 * 1000)  # Size relative to 1MP
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'size_ratio': size_ratio
        }
    
    def extract_color_features(self, image):
        """
        Extract color distribution features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of color features
        """
        # Separate color channels
        b, g, r = cv2.split(image)
        
        # Calculate statistics for each channel
        features = {}
        
        # Standard deviation (color variability)
        features['color_std_r'] = np.std(r)
        features['color_std_g'] = np.std(g)
        features['color_std_b'] = np.std(b)
        
        # Mean (color bias)
        features['color_mean_r'] = np.mean(r)
        features['color_mean_g'] = np.mean(g)
        features['color_mean_b'] = np.mean(b)
        
        # Skewness (asymmetry in color distribution)
        features['color_skew_r'] = stats.skew(r.flatten())
        features['color_skew_g'] = stats.skew(g.flatten())
        features['color_skew_b'] = stats.skew(b.flatten())
        
        # Kurtosis (peakedness of color distribution)
        features['color_kurtosis_r'] = stats.kurtosis(r.flatten())
        features['color_kurtosis_g'] = stats.kurtosis(g.flatten())
        features['color_kurtosis_b'] = stats.kurtosis(b.flatten())
        
        # Color channel correlations (color consistency)
        features['color_correlation_rg'] = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        features['color_correlation_rb'] = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        features['color_correlation_gb'] = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        
        return features
    
    def extract_complexity_features(self, image):
        """
        Extract complexity and edge features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of complexity features
        """
        # Convert to grayscale for certain calculations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure the image is 8-bit unsigned integer format for Canny
        gray_uint8 = cv2.convertScaleAbs(gray)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray_uint8, 100, 200)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Image entropy (measure of randomness)
        entropy = shannon_entropy(gray)
        
        # Standard deviation of pixel values (overall contrast)
        std_dev = np.std(gray)
        
        # Spatial complexity using gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        spatial_complexity = np.mean(gradient_magnitude)
        
        # Corner detection (Harris corner detector)
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corner_density = np.sum(corners > 0.01 * corners.max()) / (gray.shape[0] * gray.shape[1])
        
        return {
            'edge_density': edge_density,
            'entropy': entropy,
            'std_dev': std_dev,
            'spatial_complexity': spatial_complexity,
            'corner_density': corner_density
        }
    
    def extract_noise_features(self, image):
        """
        Extract noise pattern features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of noise features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply a denoising filter
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Calculate the noise component
        noise = gray.astype(np.int16) - denoised.astype(np.int16)
        
        # Noise level (overall amount of noise)
        noise_level = np.std(noise)
        
        # Noise pattern consistency (variance of local noise std)
        block_size = 16
        local_noise_std = []
        for i in range(0, gray.shape[0] - block_size, block_size):
            for j in range(0, gray.shape[1] - block_size, block_size):
                block = noise[i:i+block_size, j:j+block_size]
                local_noise_std.append(np.std(block))
        noise_pattern_consistency = np.var(local_noise_std)
        
        # Noise spectrum analysis (using FFT)
        noise_fft = np.abs(fftpack.fft2(noise))
        noise_fft_shifted = fftpack.fftshift(noise_fft)
        center_y, center_x = noise_fft_shifted.shape[0] // 2, noise_fft_shifted.shape[1] // 2
        radius = min(center_y, center_x) // 2
        
        # Create a mask for high-frequency components
        y, x = np.ogrid[:noise_fft_shifted.shape[0], :noise_fft_shifted.shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask_hf = dist_from_center > radius
        mask_lf = dist_from_center <= radius
        
        # Calculate ratio of high to low frequency components
        hf_energy = np.sum(noise_fft_shifted[mask_hf])
        lf_energy = np.sum(noise_fft_shifted[mask_lf])
        noise_spectrum_ratio = hf_energy / (lf_energy + 1e-10)
        
        # Noise structure (auto-correlation measurement)
        noise_flat = noise.flatten()
        noise_structure = np.correlate(noise_flat, noise_flat, mode='same')[len(noise_flat)//2:len(noise_flat)//2+100].std()
        
        return {
            'noise_level': noise_level,
            'noise_pattern_consistency': noise_pattern_consistency,
            'noise_spectrum_ratio': noise_spectrum_ratio,
            'noise_structure': noise_structure
        }
    
    def extract_texture_features(self, image):
        """
        Extract local texture pattern features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of texture features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to a standard size to ensure consistent LBP features
        gray = cv2.resize(gray, (256, 256))
        
        # Parameters for LBP
        radius = 3
        n_points = 24
        
        # Compute LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute the histogram of LBP values
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize the histogram
        hist = hist.astype("float") / (hist.sum() + 1e-10)
        
        # Create feature dictionary with LBP histogram bins
        features = {}
        for i, value in enumerate(hist):
            features[f'lbp_hist_{i}'] = value
            
        return features
    
    def extract_frequency_features(self, image):
        """
        Extract frequency domain features from an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            dict: Dictionary of frequency domain features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D FFT
        f = fftpack.fft2(gray)
        fshift = fftpack.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # Create concentric ring masks to analyze frequency bands
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_radius = min(center_y, center_x)
        ring_features = {}
        
        # Analyze 10 frequency rings
        num_rings = 10
        for i in range(num_rings):
            inner_radius = i * max_radius / num_rings
            outer_radius = (i + 1) * max_radius / num_rings
            mask = (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
            ring_mean = np.mean(magnitude_spectrum[mask])
            ring_features[f'fft_ring_{i}'] = ring_mean
        
        # Statistical properties of the frequency spectrum
        flat_spectrum = magnitude_spectrum.flatten()
        fft_std = np.std(flat_spectrum)
        fft_skew = stats.skew(flat_spectrum)
        fft_kurtosis = stats.kurtosis(flat_spectrum)
        
        # Energy of the spectrum
        fft_energy = np.sum(np.abs(f)**2) / (gray.shape[0] * gray.shape[1])
        
        # Combine all frequency features
        features = ring_features.copy()
        features.update({
            'fft_std': fft_std,
            'fft_skew': fft_skew,
            'fft_kurtosis': fft_kurtosis,
            'fft_energy': fft_energy
        })
        
        return features


def extract_features_from_image_path(image_path, extractor):
    """
    Extract features from an image specified by its path.
    
    Args:
        image_path (str): Path to the image file
        extractor (ImageFeatureExtractor): The feature extractor to use
        
    Returns:
        dict: Dictionary of extracted features, or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is not None:
            features = extractor.extract_features(image)
            features['image_path'] = image_path
            return features
        return None
    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None


def extract_features_batch(image_paths, extractor, n_workers=4):
    """
    Extract features from a batch of images in parallel.
    
    Args:
        image_paths (list): List of paths to image files
        extractor (ImageFeatureExtractor): The feature extractor to use
        n_workers (int): Number of worker threads to use
        
    Returns:
        pandas.DataFrame: DataFrame of extracted features
    """
    # Extract features in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            lambda path: extract_features_from_image_path(path, extractor),
            image_paths
        ))
    
    # Filter out None results (failed extractions)
    results = [r for r in results if r is not None]
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        # Return empty DataFrame with the right columns
        columns = ['image_path'] + extractor.get_feature_names()
        return pd.DataFrame(columns=columns)


def extract_all_features(data_dir, output_file=None, include_all=True, n_workers=4):
    """
    Extract features from all images in a directory structure.
    
    Args:
        data_dir (str): Directory containing the images
        output_file (str): Path to save the output CSV file (optional)
        include_all (bool): Whether to include all feature types
        n_workers (int): Number of worker threads
        
    Returns:
        pandas.DataFrame: DataFrame of extracted features
    """
    # Create feature extractor
    extractor = ImageFeatureExtractor(
        include_metadata=include_all or True,
        include_color=include_all or True,
        include_complexity=include_all or True, 
        include_noise=include_all or True,
        include_texture=include_all or True,
        include_frequency=include_all or True
    )
    
    # Find all images in the directory
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images in {data_dir}")
    
    # Extract features
    features_df = extract_features_batch(image_paths, extractor, n_workers)
    
    # Save to file if specified
    if output_file and not features_df.empty:
        features_df.to_csv(output_file, index=False)
        print(f"Saved features to {output_file}")
    
    return features_df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from images")
    parser.add_argument("--data_dir", required=True, help="Directory containing images")
    parser.add_argument("--output", default="image_features.csv", help="Output CSV file")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Extract features from all images in the directory
    extract_all_features(args.data_dir, args.output, n_workers=args.workers) 