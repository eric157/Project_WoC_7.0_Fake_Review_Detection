import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_DIR
from model_definitions import BinaryClassifier, ReviewDataset
from config import RANDOM_STATE

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression Model:")
    model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear')
    model.fit(X_train, y_train)

    model_filename = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    joblib.dump(model, model_filename)
    return model, model_filename

def train_random_forest(X_train, y_train):
    print("Training Random Forest Model:")
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    model_filename = os.path.join(MODEL_DIR, "random_forest_model.pkl")
    joblib.dump(model, model_filename)
    return model, model_filename

def train_neural_network(X_train, y_train, feature_cols):
    print("Training Feedforward Neural Network Model:")
    input_size = len(feature_cols)
    hidden_size = 256
    output_size = 1
    learning_rate = 3e-4
    num_epochs = 10
    batch_size = 32

    train_dataset = ReviewDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = BinaryClassifier(input_size, hidden_size, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_filename = os.path.join(MODEL_DIR, "neural_network_model.h5")
    torch.save(model.state_dict(), model_filename)
    return model, model_filename

def train_svm(X_train, y_train):
    print("Training SVM Model:")
    model = SGDClassifier(loss='hinge', random_state=RANDOM_STATE, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    
    model_filename = os.path.join(MODEL_DIR, "svm_model.pkl")
    joblib.dump(model, model_filename)
    return model, model_filename
def evaluate_model(model, X, y):
  y_pred = model.predict(X)
  metrics = {
      "accuracy": accuracy_score(y, y_pred),
      "precision": precision_score(y, y_pred, average='binary', zero_division=0),
      "recall": recall_score(y, y_pred, average='binary', zero_division=0),
      "f1": f1_score(y, y_pred, average='binary', zero_division=0),
  }
  return metrics

def evaluate_neural_network(model, X, y):
  with torch.no_grad():
    outputs = model(torch.tensor(X, dtype=torch.float32))
    y_pred = (outputs > 0.5).float().flatten().numpy()
  metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred, average='binary', zero_division=0),
    "recall": recall_score(y, y_pred, average='binary', zero_division=0),
    "f1": f1_score(y, y_pred, average='binary', zero_division=0),
  }
  return metrics

def train_and_evaluate_all_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    all_results = {}

    # Logistic Regression
    model, model_filename = train_logistic_regression(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"Model: Logistic Regression - Trained and Saved to: {model_filename}")
    print(f"  Train Metrics: {train_metrics}")
    print(f"  Validation Metrics: {val_metrics}")
    print(f"  Test Metrics: {test_metrics}")
    print("-" * 150)
    all_results["Logistic Regression"] = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_file": model_filename
    }
    
    # Random Forest
    model, model_filename = train_random_forest(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"Model: Random Forest - Trained and Saved to: {model_filename}")
    print(f"  Train Metrics: {train_metrics}")
    print(f"  Validation Metrics: {val_metrics}")
    print(f"  Test Metrics: {test_metrics}")
    print("-" * 150)
    all_results["Random Forest"] = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_file": model_filename
    }
    
    # Neural Network
    model, model_filename = train_neural_network(X_train, y_train, feature_cols)
    train_metrics = evaluate_neural_network(model, X_train, y_train)
    val_metrics = evaluate_neural_network(model, X_val, y_val)
    test_metrics = evaluate_neural_network(model, X_test, y_test)
    print(f"Model: Feedforward Neural Network - Trained and Saved to: {model_filename}")
    print(f"  Train Metrics: {train_metrics}")
    print(f"  Validation Metrics: {val_metrics}")
    print(f"  Test Metrics: {test_metrics}")
    print("-" * 150)
    all_results["Neural Network"] = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_file": model_filename
    }

    # SVM
    model, model_filename = train_svm(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"Model: SVM - Trained and Saved to: {model_filename}")
    print(f"  Train Metrics: {train_metrics}")
    print(f"  Validation Metrics: {val_metrics}")
    print(f"  Test Metrics: {test_metrics}")
    print("-" * 150)
    all_results["SVM"] = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_file": model_filename
    }

    return all_results
