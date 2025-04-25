# 🎵 What Makes a Genre

## 📌 Overview
This project explores the possibility of classifying music genres solely based on song lyrics using **Natural Language Processing (NLP)** and **Machine Learning** techniques. We used two main approaches:
1. **Traditional ML Models:** TF-IDF vectorization with Logistic Regression and Neural Networks.
2. **Deep Learning Models:** BERT embeddings fine-tuned for genre classification.

## 🛠️ Software & Platforms
- **Python** (Pandas, NumPy, Scikit-Learn, TensorFlow, Matplotlib, Seaborn)
- **Jupyter Notebook**
- **Google Colab** (for cloud-based computation)
- **GitHub** (for version control and collaboration)
- **Operating System:** Windows / Mac / Linux

## 📁 Project Structure
```
📂 Project 1
│── 📄 README.md                    # Project documentation
│── 📄 LICENSE.md                   # MIT License
│── 📄 requirements.txt             # List of project dependencies
│
│── 📂 DATA/                        # Dataset and Data Appendix
│   │── Data Appendix Project 1.pdf   # Detailed information about the dataset
│   │── processed_data.csv            # Cleaned dataset ready for analysis
│   │── tcc_ceds_music.csv            # Original dataset
│   │── train_data.csv                # 80% of processed_data used for training models
│   │── test_data.csv                 # 20% of processed_data used for testing models
│
│── 📂 OUTPUT/                      # Generated results and reports
│   │
│   │── 📂 MODELS/                   # Trained models and configurations
│   │   │── config.json                 # Model configuration
│   │   │── link to Hugging Face Model Hub repository  # Reference link to model on Hugging Face
│   │   │── model.safetensors            # Trained model weights in safetensors format
│   │   │── special_tokens_map.json      # Special tokens mapping for tokenizer
│   │   │── tokenizer_config.json        # Tokenizer configuration
│   │   │── vocab.txt                    # Vocabulary used by tokenizer
│   │
│   │── 📂 confusion matrices using BERT/
│   │   │── LR_confusion_matrix_BERT.png   # Confusion Matrix for Logistic Regression using BERT
│   │   │── NN_confusion_matrix_BERT.png   # Confusion Matrix for Neural Network using BERT
│   │
│   │── 📂 confusion matrices using TF-IDF/
│   │   │── LR_confusion_matrix_TF-IDF.png # Confusion Matrix for Logistic Regression using TF-IDF
│   │   │── NN_confusion_matrix_TF-IDF.png # Confusion Matrix for Neural Network using TF-IDF
│   │
│   │── PCA_visualization.png              # PCA visualization of BERT embeddings
│   │── X_embeddings.npy                   # Saved BERT embeddings for lyrics
│   │── genre_classification_report_using_BERT.pdf   # Classification report for BERT
│   │── genre_classification_report_using_TF-IDF_&_Logistic Regression.pdf # Classification report for TF-IDF & LogReg
│   │── label_encoder.pkl                  # Label encoder for genre classification
│   │── logistic_regression_bert.pkl        # Trained Logistic Regression model using BERT
│   │── neural_network_bert.h5              # Trained Neural Network model using BERT
│   │── test_predictions.csv                # Predictions made on test data
│   │── tfidf_vectorizer.pkl                # TF-IDF vectorizer for text features
│   │── y.npy                              # Encoded genre labels
│
│── 📂 SCRIPTS/                     # All scripts for training, testing, and visualizations
│   │
│   │── 📂 Testing Models scripts/  # Scripts to test different models
│   │   │── test_bert_model.ipynb      # Jupyter Notebook to test BERT model performance
│   │   │── test_ml_model.ipynb        # Jupyter Notebook to test ML models performance
│   │
│   │── 📂 visualization scripts/  # Scripts for generating visualizations
│   │   │── visualization_for_ml_models.ipynb     # Visualizations for ML model performance
│   │   │── visualizations_for_bert_model_performance.ipynb  # Visualizations for BERT model performance
│   │
│   │── EDA.ipynb                     # Exploratory Data Analysis notebook
│   │── bert_based_classification.ipynb # BERT-based classification approach
│   │── extract_embeddings.py          # Extract BERT embeddings from lyrics
│   │── predict_genre_bert_model.py     # Predict genre using the BERT model
│   │── predict_genre_ml_models.py      # Predict genre using traditional ML models
│   │── preprocess_data.py              # Data cleaning and preprocessing
│   │── train_bert_model.py             # Train BERT-based genre classifier
│   │── train_ml_models.py              # Train ML models (Logistic Regression, Neural Network)
│   │── train_test_split.ipynb          # Split data into training and testing sets
│   │── utils.py                       # Helper functions and utilities

```

## 🔄 Reproducibility
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rixprakash/CS3.git
   cd "Project 1"
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess Data:**
   ```bash
   python SCRIPTS/preprocess_data.py
   ```
4. **Extract BERT Embeddings:**
   ```bash
   python SCRIPTS/extract_embeddings.py
   ```
5. **Train the Models:**
   - **Traditional ML Models:**
     ```bash
     python SCRIPTS/train_ml_models.py
     ```
   - **Fine-tune BERT Model:**
     ```bash
     python SCRIPTS/train_bert_model.py
     ```
6. **Visualizations:**
   ```bash
   jupyter notebook SCRIPTS/visualizations_for_bert_model_performance.ipynb
   ```

---
   - Confusion Matrices, PCA visualization, and Classification Reports can be found in the `OUTPUT/` folder.

## 📜 License
This project is licensed under the MIT License. See the `LICENSE.md` file for details.

## 📚 References
- Research papers on NLP and song lyric analysis
- Scikit-Learn, TensorFlow, and Hugging Face Transformers documentation
- GitHub repositories on music genre classification
- https://data.mendeley.com/datasets/3t9vbwxgr5/2
