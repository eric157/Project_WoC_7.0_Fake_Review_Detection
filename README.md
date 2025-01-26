# 🤖 Fake Review Detection Pipeline

A comprehensive pipeline for text data preprocessing, feature extraction, model training, and evaluation, culminating in the selection of the best-performing model for fake review detection. This project leverages advanced techniques from natural language processing (NLP) and machine learning (ML). Additionally, it features a web scraping component to collect review data directly from the web.

---

## 🗂 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [Usage](#-usage)
  - [Checkpoint 1](#checkpoint-1)
  - [Checkpoint 2](#checkpoint-2)
  - [Checkpoint 3](#checkpoint-3)
- [Modules](#-modules)
  - [Checkpoint 1 Modules](#checkpoint-1-modules)
  - [Checkpoint 2 Modules](#checkpoint-2-modules)
  - [Checkpoint 3 Modules](#checkpoint-3-modules)
- [Evaluation Metrics](#-evaluation-metrics)
- [Acknowledgements](#-acknowledgements)

---

## 📝 Overview

This pipeline is designed to process raw datasets of reviews through several stages: preprocessing, feature extraction, model training, and evaluation. The project is modularly divided into three checkpoints:

### Checkpoint 1: 
Focuses on text cleaning, preprocessing, and vectorization into numerical features using TF-IDF.

### Checkpoint 2: 
Trains a Logistic Regression machine learning model, evaluates it, and identifies the best-performing model for detecting fake reviews.

### Checkpoint 3: 
Scrapes review data directly from Amazon product pages to build datasets for further analysis.

---

## ✨ Features

- **Text Preprocessing**:
  - Expands contractions (e.g., "can't" → "cannot").
  - Removes punctuation, digits, and stopwords.
  - Converts text to lowercase and applies lemmatization.

- **TF-IDF Vectorization**:
  - Converts text into numerical features using Term Frequency-Inverse Document Frequency.
  - Configurable parameters for n-grams, feature limits, and frequency thresholds.

- **Model Training**:
  - Trains a Logistic Regression model with Bayesian optimization for hyperparameter tuning.

- **Evaluation Metrics**:
  - Reports metrics like accuracy, precision, recall, and F1-score.

- **Model Persistence**:
  - Saves trained models for future use.

- **Best Model Selection**:
  - Identifies the best model based on validation F1-score.

- **Web Scraping**:
  - Scrapes product reviews from Amazon, filters by star rating, and saves data to a CSV file.

- **Modular Design**:
  - Ensures maintainability, scalability, and customization.

---

## 📁 Folder Structure

```
Project_WoC_7.0_Fake_Review_Detection/
├── checkpoint 1/
│   ├── preprocessing.ipynb          # Jupyter notebook for preprocessing and feature extraction
│   ├── data/                        # Input datasets
│   └── output/                      # Preprocessed datasets
├── checkpoint 2/
│   ├── model_training.ipynb         # Jupyter notebook for model training
│   ├── models/                      # Trained models
│   └── output/                      # Output metrics and results
└── checkpoint 3/
│   ├── scraping.ipynb               # Jupyter notebook for web scraping
│   ├── scraped_reviews.csv          # Scraped reviews
└── README.md                        # Project documentation
```

---

## 🚀 Usage

### Checkpoint 1: Preprocessing and Feature Extraction

1. **Prepare the Input Dataset**:
   - Place your dataset in the `checkpoint 1/data/` folder.
   - Ensure the file is a CSV with a text column named `text`.

2. **Run the Preprocessing Pipeline**:
   - Execute the `preprocessing.ipynb` notebook.
   - Output: The processed dataset will be saved in `checkpoint 1/output/` as `FakeReviewDataPreprocessed.csv`.

---

### Checkpoint 2: Model Training and Evaluation

1. **Ensure Checkpoint 1 is Completed**:
   - Verify that `FakeReviewDataPreprocessed.csv` exists in the `checkpoint 1/output/` folder.

2. **Run the Training and Evaluation Pipeline**:
   - Execute the `model_training.ipynb` notebook.
   - Output:
     - Trained model saved in `checkpoint 2/models/` as `logistic_regression_model.pkl`.
     - Evaluation metrics saved in `checkpoint 2/output/` as `model_evaluation_metrics.txt`.

---

### Checkpoint 3: Web Scraping

1. **Run the Web Scraping Pipeline**:
   - Execute the `scraping.ipynb` notebook.
   - Enter the Amazon product URL when prompted.
   - Output:
     - Scraped reviews saved in the root directory as `scraped_reviews.csv`.

---

## 📦 Modules

### Checkpoint 1 Modules

1. **Preprocessing (`preprocessing.ipynb`)**:
   - Handles text cleaning and normalization:
     - Expands contractions (e.g., "can't" → "cannot").
     - Converts text to lowercase.
     - Removes punctuation, digits, and stopwords.
     - Applies tokenization and lemmatization.
   - Extracts numerical features from preprocessed text using TF-IDF with configurable parameters like `max_features`.

---

### Checkpoint 2 Modules

1. **Model Training (`model_training.ipynb`)**:
   - Loads and prepares the dataset from Checkpoint 1.
   - Trains and evaluates a Logistic Regression model:
     - Utilizes Bayesian optimization for hyperparameter tuning.
   - Saves the trained model and evaluation metrics to disk.

---

### Checkpoint 3 Modules

1. **Web Scraping (`scraping.ipynb`)**:
   - Scrapes product reviews from Amazon product pages.
   - Allows filtering reviews by star rating.
   - Saves scraped reviews to a CSV file.

---

## 📊 Evaluation Metrics

The following metrics were obtained after running Checkpoint 2:

- **Training Metrics**:
  - Accuracy: 0.8985
  - Precision: 0.8943
  - Recall: 0.9035
  - F1 Score: 0.8989

- **Validation Metrics**:
  - Accuracy: 0.8548
  - Precision: 0.8597
  - Recall: 0.8474
  - F1 Score: 0.8535

- **Test Metrics**:
  - Accuracy: 0.8642
  - Precision: 0.8683
  - Recall: 0.8637
  - F1 Score: 0.8660

---

## 🌟 Acknowledgements

This project is part of WoC 7.0, developed to demonstrate a comprehensive pipeline for detecting fake reviews using NLP and ML techniques. Special thanks to the mentors for their guidance and support.
