# DeepGuardDB Dataset: Exploratory Data Analysis Summary

## Dataset Structure

The DeepGuardDB dataset contains images from four different AI image generation models, along with real photos:

1. **SD (Stable Diffusion)**: 2,675 real images and 2,675 fake images
2. **DALLE (DALL-E 3)**: 2,150 real images and 1,480 fake images
3. **IMAGEN**: 1,175 real images and 997 fake images
4. **GLIDE**: 500 real images and 500 fake images

In total, the dataset contains:
- 6,500 real images
- 5,652 fake images

## Distribution by Platform

The metadata analysis shows the following distribution of images by platform:
- Stable Diffusion (SD): 41.15%
- DALL-E 3: 33.08%
- IMAGEN: 18.08%
- GLIDE: 7.69%

## Prompt Analysis

We analyzed the text prompts used to generate the fake images:

1. **Most Common Words**: The most frequent words in the prompts include "man," "woman," "sitting," "wearing," "standing," "people," and "walking." This suggests the dataset contains many images of people in various poses and activities.

2. **Platform-Specific Words**: Different models have slightly different distributions of prompt words, which may reflect their respective strengths or the types of images they're commonly used to generate.

## Image Properties

Analysis of image dimensions, aspect ratios, and file sizes revealed:

1. **Dimensions**: Most images have standard dimensions, with real images showing more variety in sizes compared to AI-generated ones, which tend to have more consistent dimensions.

2. **Aspect Ratios**: The majority of images have an aspect ratio close to 1.0 (square), with Stable Diffusion and DALL-E showing a tight cluster around the square format. Real images show more variety in aspect ratios.

3. **File Sizes**: AI-generated images tend to have more consistent file sizes, while real photos show greater variance in file size.

## Visual Observations

From the sample image visualizations:

1. **SD (Stable Diffusion)**: Good quality images with occasional artifacts. Strong at generating scenes with people, animals, and outdoor environments.

2. **DALL-E**: High quality images with impressive detail. Performs well with complex scenes and human figures.

3. **IMAGEN**: Good quality with distinctive style. Tends to have a slightly different color palette compared to the others.

4. **GLIDE**: The oldest model in the set. Quality is good but not as detailed as the newer models.

## Implications for Model Building

Based on this EDA, we can derive several insights for building a real vs. AI image detector:

1. **Balanced Dataset**: The dataset is relatively balanced between real and fake images, making it suitable for training a classifier.

2. **Model-Specific Patterns**: Different AI generators have distinct visual signatures. A robust detector should be able to identify images from all four generators.

3. **Feature Engineering**: Potential discriminative features include:
   - Image metadata (dimensions, aspect ratios, file sizes)
   - Frequency domain characteristics
   - Texture and pattern consistency
   - Color distribution and transitions

4. **Training Strategy**: Given the distribution of images, we might want to use stratified sampling to ensure the model learns from all generators proportionally.

5. **Evaluation**: We should evaluate the model's performance on each generator separately to understand its strengths and weaknesses.

## Next Steps

1. **Feature Extraction**: Extract relevant features from the images that may help distinguish real from AI-generated.

2. **Model Selection**: Compare various deep learning architectures (CNNs, Vision Transformers) for this classification task.

3. **Transfer Learning**: Leverage pre-trained models on similar tasks to improve performance.

4. **Ensemble Approach**: Consider combining multiple models specialized for different generators.

5. **Evaluation Metrics**: Define appropriate metrics (accuracy, precision, recall, F1-score) for model evaluation, with special attention to false positives and false negatives. 