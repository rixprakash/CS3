# AI Image Detection Project

## Project Overview
This project aims to develop a classification model that can distinguish between AI-generated images and authentic photographs with at least 80% accuracy across multiple image categories and generation techniques.

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