"""
Train KNN Classifier for Handwritten Digit Recognition
Matches: digit_recognition.ipynb by procoder369

Dataset: Scikit-learn 8x8 Digits (1,797 samples, 10 classes 0-9)
Model:   KNeighborsClassifier(n_neighbors=5)
Split:   80% train / 20% test, random_state=42
Saves:   model.pkl
"""

import os
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_save_model(model_path="model.pkl"):
    print("==================================================")
    print("   TRAINING KNN DIGIT RECOGNITION CLASSIFIER")
    print("   (Matching digit_recognition.ipynb)")
    print("==================================================")

    # 1. Load dataset (8x8 handwritten digits) — same as notebook
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features (8x8 pixels)")
    print(f"Classes: {np.unique(y)}")

    # 2. Train/Test Split — exact same as notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # 3. KNN Classifier — exact same as notebook
    knn = KNeighborsClassifier(n_neighbors=5)

    # 4. Train and evaluate — same as notebook
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 5. Save model
    joblib.dump(knn, model_path)
    print(f"\nModel saved to '{model_path}' ({os.path.getsize(model_path)} bytes)")
    print("==================================================\n")
    return knn

if __name__ == "__main__":
    train_and_save_model()
