"""
This script trains two traditional machine learning models (Logistic Regression and a Neural Network)
on pre-extracted BERT embeddings for genre classification.

Workflow:
1. Downloads precomputed BERT embeddings (`X_embeddings.npy`), labels (`y.npy`), and the label encoder.
2. Splits the dataset into training and testing sets (80/20 split).
3. Trains a Logistic Regression model and evaluates it on the test set.
4. Trains a Neural Network model with multiple dense layers and dropout regularization.
5. Generates classification reports and confusion matrices to evaluate model performance.
6. Saves both trained models for future inference.

This script is useful for quickly training and evaluating lightweight ML models without the need
to fine-tune BERT directly.
"""

import numpy as np
import joblib
import tensorflow as tf
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load Data from GitHub
X_url = "https://github.com/rixprakash/Jarheads/raw/main/Project%201/OUTPUT/X_embeddings.npy"  # Use /raw/
y_url = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/OUTPUT/y.npy"
label_url = "https://raw.githubusercontent.com/rixprakash/Jarheads/main/Project%201/OUTPUT/label_encoder.pkl"

# Function to Download Large Files from GitHub
def download_file(url, filename):
    """Download large files from GitHub LFS."""
    if not os.path.exists(filename) or os.path.getsize(filename) < 1024:  # Avoid downloading again
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename} successfully!")
        else:
            print(f"Failed to download {filename}. Please download manually from {url}")
    else:
        print(f"{filename} already exists. Skipping download.")

# Download files
download_file(X_url, "X_embeddings.npy")
download_file(y_url, "y.npy")
download_file(label_url, "label_encoder.pkl")

# Load Data
try:
    X = np.load("X_embeddings.npy", allow_pickle=True)
    y = np.load("y.npy", allow_pickle=True)
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"Loaded embeddings! X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression Model
print("\nTraining Logistic Regression Model...")
logreg_model = LogisticRegression(max_iter=5000, solver="lbfgs")
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_test)
print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_logreg))

# Save Logistic Regression Model
joblib.dump(logreg_model, "logistic_regression_bert.pkl")
print("Logistic Regression model saved!")

# Step 4: Train Neural Network Model
def build_neural_network(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

num_classes = len(label_encoder.classes_)

print("\nTraining Neural Network...")
nn_model = build_neural_network(X_train.shape[1], num_classes)

history = nn_model.fit(X_train, y_train, epochs=15, batch_size=4, validation_data=(X_test, y_test), verbose=1)

y_pred_nn = np.argmax(nn_model.predict(X_test), axis=1)
print("\nNeural Network Results:")
print(classification_report(y_test, y_pred_nn))

# Save Neural Network Model
nn_model.save("neural_network_bert.h5")
print("Neural Network model saved!")

# Step 5: Plot Confusion Matrices
def plot_confusion_matrix(y_test, y_pred, model_name, label_encoder):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

plot_confusion_matrix(y_test, y_pred_logreg, "Logistic Regression (BERT)", label_encoder)
plot_confusion_matrix(y_test, y_pred_nn, "Neural Network (BERT)", label_encoder)

print("\n Training Complete! Models and confusion matrices are ready.")
