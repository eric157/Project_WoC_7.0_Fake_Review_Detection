import os
import joblib
import torch
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_DIR, OUTPUT_DIR, METRICS_FILE
from model_definitions import BinaryClassifier
from config import RANDOM_STATE
def evaluate_trained_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    best_model_name = None
    best_val_f1 = 0
    model_names = [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "Neural Network"
    ]
    results = {}
    input_size = len(feature_cols)
    hidden_size = 256
    output_size = 1
    for model_name in model_names:
        model_file = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}_model.pkl")
        
        # Handle scikit-learn models
        if os.path.exists(model_file) and model_name != "Neural Network":
            # Load scikit-learn model
            model = joblib.load(model_file)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_metrics = {
                "accuracy": accuracy_score(y_train, y_train_pred),
                "precision": precision_score(y_train, y_train_pred, average='binary', zero_division=0),
                "recall": recall_score(y_train, y_train_pred, average='binary', zero_division=0),
                "f1": f1_score(y_train, y_train_pred, average='binary', zero_division=0),
            }
            val_metrics = {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, average='binary', zero_division=0),
                "recall": recall_score(y_val, y_val_pred, average='binary', zero_division=0),
                "f1": f1_score(y_val, y_val_pred, average='binary', zero_division=0),
            }
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, average='binary', zero_division=0),
                "recall": recall_score(y_test, y_test_pred, average='binary', zero_division=0),
                "f1": f1_score(y_test, y_test_pred, average='binary', zero_division=0),
            }
            
            # Store results
            results[model_name] = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics
            }

        # Handle PyTorch model
        elif model_name == "Neural Network":
            model_file = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}_model.h5")
            
            if os.path.exists(model_file):
                # Load the PyTorch model
                model = BinaryClassifier(input_size, hidden_size, output_size)
                model.load_state_dict(torch.load(model_file))
                model.eval()

                # Convert input data to PyTorch tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

                with torch.no_grad():
                    # Predictions
                    y_train_pred = (model(X_train_tensor) > 0.5).float().numpy().flatten()
                    y_val_pred = (model(X_val_tensor) > 0.5).float().numpy().flatten()
                    y_test_pred = (model(X_test_tensor) > 0.5).float().numpy().flatten()

                # Metrics
                train_metrics = {
                    "accuracy": accuracy_score(y_train, y_train_pred),
                    "precision": precision_score(y_train, y_train_pred, average='binary', zero_division=0),
                    "recall": recall_score(y_train, y_train_pred, average='binary', zero_division=0),
                    "f1": f1_score(y_train, y_train_pred, average='binary', zero_division=0),
                }
                val_metrics = {
                    "accuracy": accuracy_score(y_val, y_val_pred),
                    "precision": precision_score(y_val, y_val_pred, average='binary', zero_division=0),
                    "recall": recall_score(y_val, y_val_pred, average='binary', zero_division=0),
                    "f1": f1_score(y_val, y_val_pred, average='binary', zero_division=0),
                }
                test_metrics = {
                    "accuracy": accuracy_score(y_test, y_test_pred),
                    "precision": precision_score(y_test, y_test_pred, average='binary', zero_division=0),
                    "recall": recall_score(y_test, y_test_pred, average='binary', zero_division=0),
                    "f1": f1_score(y_test, y_test_pred, average='binary', zero_division=0),
                }

                # Store results
                results[model_name] = {
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }

    # Determine best model
    for model_name in results.keys():
        val_f1 = results[model_name]['val_metrics']['f1'] if 'val_metrics' in results[model_name] else 0
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = model_name

    # Print best model
    print(f"Best performing model based on validation F1 score: {best_model_name}")

    # Prepare table data
    table_data = []
    for model_name, metrics in results.items():
        table_data.append([
            model_name,
            metrics["train_metrics"]["f1"],
            metrics["val_metrics"]["f1"],
            metrics["test_metrics"]["f1"]
        ])

    # Format table
    table_headers = ["Model", "Train F1", "Validation F1", "Test F1"]
    table = tabulate(table_data, headers=table_headers, tablefmt="grid")
    print(table)

    # Save table to text file
    output_file_path = os.path.join(OUTPUT_DIR, METRICS_FILE)
    with open(output_file_path, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write(table)
        f.write("\n")
        f.write(f"Best performing model based on validation F1 score: {best_model_name}")
    return results
