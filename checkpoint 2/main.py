# main.py
import os
from config import MODEL_DIR, OUTPUT_DIR
from data_loading import load_and_prepare_data
from train import train_and_evaluate_all_models
from evaluate import evaluate_trained_models
# Create model and output directories
os.makedirs(os.path.join(MODEL_DIR), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR), exist_ok=True)

if __name__ == "__main__":
    # Load and prepare data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_and_prepare_data()
    except FileNotFoundError as e:
        print(e)
        exit()
    except ValueError as e:
        print(e)
        exit()

    # Train all models
    all_results = train_and_evaluate_all_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)

    # Evaluate trained models
    evaluate_trained_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)

    print("Model training and testing process completed.")