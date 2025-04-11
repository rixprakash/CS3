import os
import shutil
from pathlib import Path

def organize_dataset(source_dir, target_dir):
    """
    Organize the DeepGuardDB dataset into the project structure.
    
    Args:
        source_dir (str): Path to the downloaded DeepGuardDB_v1 directory
        target_dir (str): Path to the project's DATA directory
    """
    # Create necessary directories
    os.makedirs(os.path.join(target_dir, 'images', 'real'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images', 'fake'), exist_ok=True)
    
    # Dictionary of dataset folders
    datasets = ['DALLE_dataset', 'GLIDE_dataset', 'IMAGEN_dataset', 'SD_dataset']
    
    # Copy images
    for dataset in datasets:
        # Copy real images
        real_src = os.path.join(source_dir, dataset, 'real')
        real_dst = os.path.join(target_dir, 'images', 'real')
        
        if os.path.exists(real_src):
            print(f"Copying real images from {dataset}...")
            for img in os.listdir(real_src):
                src_path = os.path.join(real_src, img)
                dst_path = os.path.join(real_dst, img)
                if os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
        
        # Copy fake images
        fake_src = os.path.join(source_dir, dataset, 'fake')
        fake_dst = os.path.join(target_dir, 'images', 'fake')
        
        if os.path.exists(fake_src):
            print(f"Copying fake images from {dataset}...")
            for img in os.listdir(fake_src):
                src_path = os.path.join(fake_src, img)
                dst_path = os.path.join(fake_dst, img)
                if os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
    
    # Copy JSON files
    json_src = os.path.join(source_dir, 'json_files')
    if os.path.exists(json_src):
        print("Copying JSON files...")
        for json_file in os.listdir(json_src):
            src_path = os.path.join(json_src, json_file)
            dst_path = os.path.join(target_dir, json_file)
            if os.path.isfile(src_path) and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (one level up from SCRIPTS)
    project_dir = os.path.dirname(script_dir)
    
    # Set up source and target directories
    source_dir = input("Please enter the full path to the DeepGuardDB_v1 directory: ")
    target_dir = os.path.join(project_dir, 'DATA')
    
    # Organize the dataset
    print(f"Organizing dataset from {source_dir} to {target_dir}")
    organize_dataset(source_dir, target_dir)
    print("Dataset organization complete!") 