#pip install transformers torch scikit-learn joblib requests pandas numpy
import os
import torch
import joblib
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ✅ Load Model and Tokenizer from Hugging Face
model_name = "AZ0202/bert-genre-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ✅ Move Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Download test dataset
TEST_CSV_URL = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/DATA/test_data.csv"
TEST_CSV_PATH = "test_data.csv"

response = requests.get(TEST_CSV_URL)
with open(TEST_CSV_PATH, "wb") as f:
    f.write(response.content)

# ✅ Load Test Data
df_test = pd.read_csv(TEST_CSV_PATH)
print(f"✅ Loaded Test Data! Shape: {df_test.shape}")

# ✅ Download and Load Label Encoder
LABEL_ENCODER_URL = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/OUTPUT/label_encoder.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

response = requests.get(LABEL_ENCODER_URL)
with open(LABEL_ENCODER_PATH, "wb") as f:
    f.write(response.content)

label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ✅ Function to Predict Genre for One Sample
def predict_genre(lyrics):
    inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU if available

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class  # Returns index of predicted genre


# ✅ Convert All Test Lyrics to Predictions
df_test["predicted_label"] = df_test["cleaned_lyrics"].apply(predict_genre)

# ✅ Convert Labels from Index to Genre Names
df_test["predicted_genre"] = df_test["predicted_label"].apply(lambda x: label_encoder.classes_[x])
df_test["true_label"] = label_encoder.transform(df_test["genre"])

# ✅ Print Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(df_test["true_label"], df_test["predicted_label"], target_names=label_encoder.classes_))
