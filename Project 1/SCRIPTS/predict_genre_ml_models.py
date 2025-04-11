import os
import torch
import numpy as np
import pandas as pd
import joblib
import requests
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model

# Ensure GPU is used if available
device = "cuda" if torch.cuda.is_available() else "cpu"

##############################
# 1. Helper Function to Download Files
##############################
GITHUB_BASE_URL = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/OUTPUT/"
GITHUB_DATA_URL = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/DATA/test_data.csv"

FILES_TO_DOWNLOAD = [
    "logistic_regression_bert.pkl",
    "neural_network_bert.h5",
    "label_encoder.pkl",
    "X_embeddings.npy",
    "y.npy"
]

def download_file(filename, url):
    """Downloads a file from GitHub repository if it's missing."""
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename} from GitHub...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ {filename} downloaded successfully!")
        else:
            raise FileNotFoundError(f"❌ Failed to download {filename}")

# Ensure required model files exist
for file in FILES_TO_DOWNLOAD:
    download_file(file, GITHUB_BASE_URL + file)

# Ensure test dataset exists
download_file("test_data.csv", GITHUB_DATA_URL)

##############################
# 2. Load Pretrained BERT Model
##############################
print("📥 Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
print("✅ BERT model loaded successfully!")

##############################
# 3. Load Trained Models & Label Encoder
##############################
print("📥 Loading trained models...")
logistic_model = joblib.load("logistic_regression_bert.pkl")
nn_model = load_model("neural_network_bert.h5")
label_encoder = joblib.load("label_encoder.pkl")
print("✅ Models loaded successfully!")

##############################
# 4. Load Test Data
##############################
print("📥 Loading test dataset...")
df_test = pd.read_csv("test_data.csv")
test_lyrics = df_test["cleaned_lyrics"].tolist()  # Lyrics to classify
true_labels = label_encoder.transform(df_test["genre"])  # Convert genre names to encoded labels
print(f"✅ Loaded {len(test_lyrics)} test samples!")

##############################
# 5. Convert Lyrics to BERT Embeddings (Batch Processing)
##############################
def get_bert_embeddings_batch(texts, batch_size=16):
    """Converts a batch of lyrics into BERT embeddings with proper GPU handling."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]  # Get batch of lyrics
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        tokens = {key: val.to(device) for key, val in tokens.items()}

        with torch.no_grad():
            output = bert_model(**tokens)
        
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

print("🔄 Converting test lyrics to BERT embeddings...")
X_test_embeddings = get_bert_embeddings_batch(test_lyrics)
print("✅ BERT embeddings extracted!")

##############################
# 6. Make Predictions on Test Data
##############################
print("🎯 Making predictions...")

# Predict with Logistic Regression
logreg_preds = logistic_model.predict(X_test_embeddings)

# Predict with Neural Network
nn_preds = np.argmax(nn_model.predict(X_test_embeddings), axis=1)

##############################
# 7. Evaluate Model Performance
##############################
print("\n📊 Generating Classification Reports...\n")

# Generate reports
logreg_report = classification_report(true_labels, logreg_preds, target_names=label_encoder.classes_)
nn_report = classification_report(true_labels, nn_preds, target_names=label_encoder.classes_)

print("\n🔍 Logistic Regression Report:\n", logreg_report)
print("\n🔍 Neural Network Report:\n", nn_report)

# Ensure OUTPUT directory exists
os.makedirs("OUTPUT", exist_ok=True)

# Save reports
with open("OUTPUT/genre_classification_report_BERT.txt", "w") as f:
    f.write("Logistic Regression Classification Report:\n")
    f.write(logreg_report + "\n\n")
    f.write("Neural Network Classification Report:\n")
    f.write(nn_report)
print("✅ Classification reports saved to OUTPUT/genre_classification_report_BERT.txt!")

##############################
# 8. Generate & Save Confusion Matrices
##############################
def save_confusion_matrices():
    """Creates and saves confusion matrices."""
    print("📊 Creating confusion matrices...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Logistic Regression Confusion Matrix
    sns.heatmap(confusion_matrix(true_labels, logreg_preds), annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0])
    axes[0].set_title("Confusion Matrix - Logistic Regression")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Neural Network Confusion Matrix
    sns.heatmap(confusion_matrix(true_labels, nn_preds), annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[1])
    axes[1].set_title("Confusion Matrix - Neural Network")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("OUTPUT/confusion_matrix_BERT_test.png")
    plt.close()
    print("✅ Confusion matrices saved to OUTPUT/confusion_matrix_BERT_test.png!")

save_confusion_matrices()

print("\n🎯 Model Testing Complete! Results saved in OUTPUT/ directory.")
