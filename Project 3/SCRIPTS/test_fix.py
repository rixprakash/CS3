import os
import cv2
import argparse
from feature_extraction import ImageFeatureExtractor

def process_single_image(image_path):
    """
    Process a single image and extract features
    
    Args:
        image_path (str): Path to the image file
    """
    print(f"Processing: {image_path}")
    
    # Create feature extractor
    extractor = ImageFeatureExtractor()
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return False
            
        # Extract features
        features = extractor.extract_features(image)
        
        # Print a few sample features
        print("Extraction successful!")
        print(f"Sample features:")
        sample_keys = list(features.keys())[:5]  # Print first 5 features
        for key in sample_keys:
            print(f"  {key}: {features[key]}")
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_multiple_images(data_dir, num_images=5):
    """
    Process multiple images from a directory
    
    Args:
        data_dir (str): Directory containing images
        num_images (int): Number of images to process
    """
    print(f"Testing feature extraction on up to {num_images} images from {data_dir}")
    
    # Create feature extractor
    extractor = ImageFeatureExtractor()
    
    # Get list of image files
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:num_images]
    
    # Process each image
    for image_file in image_files:
        try:
            image_path = os.path.join(data_dir, image_file)
            print(f"\nProcessing: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
                
            # Extract features
            features = extractor.extract_features(image)
            
            # Print a few sample features
            print("Extraction successful!")
            print(f"Sample features:")
            sample_keys = list(features.keys())[:10]  # Print first 10 features
            for key in sample_keys:
                print(f"  {key}: {features[key]}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the fixed feature extraction code")
    parser.add_argument("--image", help="Path to a single image to process")
    parser.add_argument("--dir", help="Directory containing images to process")
    parser.add_argument("--num", type=int, default=5, help="Number of images to process from directory")
    
    args = parser.parse_args()
    
    if args.image:
        process_single_image(args.image)
    elif args.dir:
        process_multiple_images(args.dir, args.num)
    else:
        # Default test case
        data_dir = "../DATA/DeepGuardDB_v1/SD_dataset/real"
        process_multiple_images(data_dir) 