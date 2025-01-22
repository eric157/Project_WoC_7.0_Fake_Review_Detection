import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import CHECKPOINT_1_OUTPUT_DIR, PREPROCESSED_DATA_FILE, RANDOM_STATE

def load_and_prepare_data():
    try:
        df = pd.read_csv(os.path.join(CHECKPOINT_1_OUTPUT_DIR, PREPROCESSED_DATA_FILE))
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {PREPROCESSED_DATA_FILE} was not found in {CHECKPOINT_1_OUTPUT_DIR}. Ensure checkpoint 1 preprocessing is completed first.")
    
    print("Preprocessed data loaded successfully.")

    feature_cols = [col for col in df.columns if re.match(r'^[a-z]+$', col) and df[col].dtype == 'float64']
    if not feature_cols:
        raise ValueError("No feature columns were found. Please verify the output of the previous checkpoint.")

    X = df[feature_cols].to_numpy()
    y = df['label']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    print("Dataset prepared for training.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)
    print("Dataset split into training+validation and test sets (90% train+val, 10% test).")

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=RANDOM_STATE)
    print("Dataset split into training and validation sets (80% train, 10% val).")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
