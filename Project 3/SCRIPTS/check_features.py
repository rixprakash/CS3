import os
import time
import pandas as pd
import argparse

def check_features(real_features_path, fake_features_path, interval=60, max_checks=30):
    """
    Check if feature files exist and display basic information.
    
    Args:
        real_features_path (str): Path to real image features file
        fake_features_path (str): Path to fake image features file
        interval (int): Check interval in seconds
        max_checks (int): Maximum number of checks
    """
    print(f"Checking for feature files every {interval} seconds (max {max_checks} checks)...")
    
    check_count = 0
    while check_count < max_checks:
        real_exists = os.path.exists(real_features_path)
        fake_exists = os.path.exists(fake_features_path)
        
        print(f"\nCheck {check_count + 1}/{max_checks}:")
        print(f"Real features file: {'EXISTS' if real_exists else 'NOT FOUND'}")
        print(f"Fake features file: {'EXISTS' if fake_exists else 'NOT FOUND'}")
        
        if real_exists:
            try:
                real_df = pd.read_csv(real_features_path)
                print(f"  - Real features: {len(real_df)} samples, {len(real_df.columns)} columns")
                if len(real_df) > 0:
                    print(f"  - First few columns: {', '.join(real_df.columns[:5])}")
            except Exception as e:
                print(f"  - Error reading real features file: {str(e)}")
        
        if fake_exists:
            try:
                fake_df = pd.read_csv(fake_features_path)
                print(f"  - Fake features: {len(fake_df)} samples, {len(fake_df.columns)} columns")
                if len(fake_df) > 0:
                    print(f"  - First few columns: {', '.join(fake_df.columns[:5])}")
            except Exception as e:
                print(f"  - Error reading fake features file: {str(e)}")
        
        if real_exists and fake_exists:
            print("\nBoth feature files are ready!")
            
            # Check if both files have data
            try:
                real_df = pd.read_csv(real_features_path)
                fake_df = pd.read_csv(fake_features_path)
                
                if len(real_df) > 0 and len(fake_df) > 0:
                    print("\nFeature extraction completed successfully!")
                    print(f"Real features: {len(real_df)} samples")
                    print(f"Fake features: {len(fake_df)} samples")
                    print("\nNext steps:")
                    print("1. Run prepare_combined_features.py to combine and process the features")
                    print("2. Train models using train_feature_model.py or main.py")
                    return True
                else:
                    print("Warning: One or both feature files appear to be empty.")
            except Exception as e:
                print(f"Error analyzing feature files: {str(e)}")
        
        check_count += 1
        if check_count < max_checks and not (real_exists and fake_exists):
            print(f"\nWaiting {interval} seconds for next check...")
            time.sleep(interval)
    
    print("\nMax checks reached. Feature extraction may still be in progress.")
    print("You can run this script again later to check the status.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check status of feature extraction")
    parser.add_argument("--real_features", default="../DATA/real_features.csv", 
                      help="Path to real image features CSV")
    parser.add_argument("--fake_features", default="../DATA/fake_features.csv", 
                      help="Path to fake image features CSV")
    parser.add_argument("--interval", type=int, default=60, 
                      help="Check interval in seconds")
    parser.add_argument("--max_checks", type=int, default=30, 
                      help="Maximum number of checks")
    
    args = parser.parse_args()
    
    check_features(args.real_features, args.fake_features, args.interval, args.max_checks) 