"""
This script fine-tunes a BERT model for genre classification using labeled song lyrics.

Workflow:
1. Downloads the `train_data.csv` dataset containing cleaned song lyrics and their corresponding genres.
2. Preprocesses the data:
   - Loads the label encoder to convert genre labels into numerical values.
   - Tokenizes lyrics using the BERT tokenizer.
3. Splits the dataset into training and validation sets (90/10 split).
4. Initializes a BERT-based model (`bert-base-uncased`) for sequence classification.
5. Trains the model using the Hugging Face `Trainer` API with:
   - Early stopping to prevent overfitting.
   - Checkpoint saving every 500 steps.
   - Logging and evaluation metrics.
6. Saves the fine-tuned model and tokenizer for future inference.

This script is essential for building a robust deep learning model that directly leverages BERT's
text representation capabilities.
"""

# make sure:
# pip install transformers datasets torch scikit-learn accelerate
#!pip install transformers datasets torch scikit-learn accelerate joblib requests
#!pip install datasets

import os
import pandas as pd
import torch
import joblib
import requests
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split

#############################
# 1. Load & Inspect the Data
#############################
GITHUB_TRAIN_URL = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/DATA/train_data.csv"
GITHUB_LABEL_ENCODER_URL = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/OUTPUT/label_encoder.pkl"

# Download function
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename} from GitHub...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ {filename} downloaded successfully!")
        else:
            raise FileNotFoundError(f"❌ Failed to download {filename}")

# Download dataset file
download_file(GITHUB_TRAIN_URL, "train_data.csv")

# Load data using pandas
train_df = pd.read_csv("train_data.csv")

print("✅ Train Data Loaded! Shape:", train_df.shape)

#############################
# 2. Preprocess the Data
#############################
# Ensure column names are stripped of whitespace
train_df.columns = train_df.columns.str.strip()

# Download label encoder (if missing)
LABEL_ENCODER_PATH = "label_encoder.pkl"
download_file(GITHUB_LABEL_ENCODER_URL, LABEL_ENCODER_PATH)

# Load label encoder
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Convert genres to numerical labels
train_df["labels"] = label_encoder.transform(train_df["genre"])

# Drop the old "genre" column
train_df = train_df.drop(columns=["genre"])

print("✅ Genre Mapping:", dict(enumerate(label_encoder.classes_)))

#############################
# 3. Split into Train & Validation Sets
#############################
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["cleaned_lyrics"], train_df["labels"], test_size=0.1, random_state=42
)

train_df = pd.DataFrame({"cleaned_lyrics": train_texts, "labels": train_labels})
val_df = pd.DataFrame({"cleaned_lyrics": val_texts, "labels": val_labels})

#############################
# 4. Create the Hugging Face Datasets
#############################
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["cleaned_lyrics"], truncation=True, padding="max_length", max_length=512)

# Convert Pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove 'cleaned_lyrics' AFTER tokenization
train_dataset = train_dataset.remove_columns(["cleaned_lyrics"])
val_dataset = val_dataset.remove_columns(["cleaned_lyrics"])

print("✅ Train Dataset Example:", train_dataset[0])

#############################
# 5. Load the BERT Model for Classification
#############################
num_labels = len(label_encoder.classes_)  # Dynamically determine number of labels
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#############################
# 6. Define Training Arguments & Trainer with Early Stopping
#############################
# Ensure logging directory exists
os.makedirs("./OUTPUT/logs", exist_ok=True)

training_args = TrainingArguments(
    output_dir="./OUTPUT/MODELS/bert_genre_model",
    evaluation_strategy="steps",        # Evaluate every 'eval_steps'
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir="./OUTPUT/logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Early stopping callback: stop if no improvement after 2 evaluations
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Include validation dataset
    callbacks=[early_stopping_callback]
)

#############################
# 7. Train the Model
#############################
print("🚀 Training the model...")
trainer.train()
print("✅ Training completed!")

#############################
# 8. Save the Fine-Tuned Model and Tokenizer
#############################

# Ensure the output directory exists
os.makedirs("OUTPUT/MODELS/bert_genre_finetuned", exist_ok=True)

model.save_pretrained("OUTPUT/MODELS/bert_genre_finetuned")
tokenizer.save_pretrained("OUTPUT/MODELS/bert_genre_finetuned")
print("✅ Model and tokenizer saved successfully!")
