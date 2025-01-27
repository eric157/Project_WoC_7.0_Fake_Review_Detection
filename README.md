# 🤖 Fake Review Detection Pipeline

A comprehensive pipeline for text data preprocessing, feature extraction, model training, and evaluation, culminating in the selection of the best-performing model for fake review detection. This project leverages advanced techniques from natural language processing (NLP) and machine learning (ML). Additionally, it features a web scraping component to collect review data directly from the web and a prediction component which can be used to predict whether new reviews are fake or not.

---

## 🗂 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [Usage](#-usage)
  - [Checkpoint 1](#checkpoint-1)
  - [Checkpoint 2](#checkpoint-2)
  - [Checkpoint 3](#checkpoint-3)
  - [Checkpoint 4](#checkpoint-4)
- [Modules](#-modules)
  - [Checkpoint 1 Modules](#checkpoint-1-modules)
  - [Checkpoint 2 Modules](#checkpoint-2-modules)
  - [Checkpoint 3 Modules](#checkpoint-3-modules)
  - [Checkpoint 4 Modules](#checkpoint-4-modules)
- [Acknowledgements](#-acknowledgements)

---

## 📝 Overview

This pipeline is designed to process raw datasets of reviews through several stages: preprocessing, feature extraction, model training, evaluation and prediction. The project is modularly divided into four checkpoints:

### Checkpoint 1: 
Focuses on text cleaning, preprocessing, and vectorization into numerical features using TF-IDF.

### Checkpoint 2: 
Trains a Logistic Regression machine learning model, evaluates it, and identifies the best-performing model for detecting fake reviews.

### Checkpoint 3: 
Scrapes review data directly from Amazon product pages to build datasets for further analysis.

### Checkpoint 4:
Applies preprocessing techniques and the trained model to new scraped reviews to predict whether they are fake or not.

---

## ✨ Features

-   **Text Preprocessing**:
    -   Expands contractions (e.g., "can't" → "cannot").
    -   Removes punctuation, digits, and stopwords.
    -   Converts text to lowercase and applies lemmatization.
    -   Handles HTML and Javascript Removal.
    -   Handles URLs, Emails and Hashtags.
    -  Spelling correction.
    -  Handles Non-ASCII characters.
     -  Language Detection and Filtering.
      -  Handling Short Texts or Noise.

-   **TF-IDF Vectorization**:
    -   Converts text into numerical features using Term Frequency-Inverse Document Frequency.
    -   Configurable parameters for n-grams, feature limits, and frequency thresholds.

-   **Model Training**:
    -   Trains a Logistic Regression model with Bayesian optimization for hyperparameter tuning.

-   **Evaluation Metrics**:
    -   Reports metrics like accuracy, precision, recall, and F1-score.

-   **Model Persistence**:
    -   Saves trained models for future use.

-   **Best Model Selection**:
    -   Identifies the best model based on validation F1-score.

-   **Web Scraping**:
    -   Scrapes product reviews from Amazon, and saves data to a CSV file.

-   **Prediction**:
    -  Applies preprocessing to new scraped data.
    -   Predicts whether new reviews are fake or not using the trained model.

-   **Modular Design**:
    -   Ensures maintainability, scalability, and customization.

---

## 📁 Folder Structure

```
Project_WoC_7.0_Fake_Review_Detection/
├── checkpoint 1/
│   ├── preprocessing.ipynb          # Jupyter notebook for preprocessing and feature extraction
│   ├── data/                        # Input datasets
│   └── output/                      # Preprocessed datasets
│   └── models/
├── checkpoint 2/
│   ├── model_training.ipynb         # Jupyter notebook for model training
│   ├── models/                      # Trained models
│   └── output/                      # Output metrics and results
├── checkpoint 3/
│   ├── scraping.ipynb               # Jupyter notebook for web scraping
│   └── scraped_reviews.csv          # Scraped reviews
└── checkpoint 4/
    ├── prediction.ipynb          # Jupyter notebook for prediction
└── README.md                        # Project documentation
```

---

## 🚀 Usage

### Checkpoint 1: Preprocessing and Feature Extraction

1. **Prepare the Input Dataset**:
   - Place your dataset in the `checkpoint 1/data/` folder.
   - Ensure the file is a CSV with a text column named `text` and a rating column named `rating`.

2. **Run the Preprocessing Pipeline**:
   - Execute the `preprocessing.ipynb` notebook.
   - Output:
     - The processed dataset will be saved in `checkpoint 1/output/` as `FakeReviewDataPreprocessed.csv`.
    - The tfidf vectorizer will be saved in `checkpoint 1/models/` as `tfidf_vectorizer.pkl`.
     - The tfidf feature names will be saved in `checkpoint 1/models/` as `tfidf_feature_names.pkl`.
---

### Checkpoint 2: Model Training and Evaluation

1. **Ensure Checkpoint 1 is Completed**:
   - Verify that `FakeReviewDataPreprocessed.csv`, `tfidf_vectorizer.pkl`, and `tfidf_feature_names.pkl` exists in the `checkpoint 1/output/` and `checkpoint 1/models` folders.

2. **Run the Training and Evaluation Pipeline**:
   - Execute the `model_training.ipynb` notebook.
   - Output:
     - Trained model saved in `checkpoint 2/models/` as `logistic_regression_model.pkl`.
     - Evaluation metrics printed on the console.

---

### Checkpoint 3: Web Scraping

1.  **Run the Web Scraping Pipeline**:
    -   Execute the `scraping.ipynb` notebook.
    -   Enter the Amazon product URL, phone number and password when prompted.
    -   Output:
        -   Scraped reviews saved in the `checkpoint 3` directory as `scraped_reviews.csv`.

---
### Checkpoint 4: Prediction

1.  **Ensure Checkpoint 1, 2 and 3 are Completed**:
        - Verify that `FakeReviewDataPreprocessed.csv`, `tfidf_vectorizer.pkl`, and `tfidf_feature_names.pkl` exists in the `checkpoint 1/output/` and `checkpoint 1/models` folders.
        - Verify that `logistic_regression_model.pkl` exist in `checkpoint 2/models`.
        - Verify that `scraped_reviews.csv` is in `checkpoint 3`.

2.  **Run the Prediction Pipeline**:
    -   Execute the `prediction.ipynb` notebook.
    -   Output:
        -   Prints the dataframe with predictions on the console.

---

## 📦 Modules

### Checkpoint 1 Modules

1.  **Preprocessing (`preprocessing.ipynb`)**:
    -   Handles text cleaning and normalization:
        -   Expands contractions (e.g., "can't" → "cannot").
        -   Converts text to lowercase.
        -   Removes punctuation, digits, and stopwords.
        -   Applies tokenization and lemmatization.
    - Extracts numerical features from preprocessed text using TF-IDF with configurable parameters like `max_features`.
      - Saves the vectorizer and the feature names.

---

### Checkpoint 2 Modules

1.  **Model Training (`model_training.ipynb`)**:
    -   Loads and prepares the dataset from Checkpoint 1.
    -   Trains and evaluates a Logistic Regression model:
        -   Utilizes Bayesian optimization for hyperparameter tuning.
    -   Saves the trained model to disk.

---

### Checkpoint 3 Modules

1.  **Web Scraping (`scraping.ipynb`)**:
    -   Scrapes product reviews from Amazon product pages.
    -   Allows filtering reviews by star rating.
    - Applies preprocessing to the text data.
    -   Saves scraped reviews to a CSV file.

---

### Checkpoint 4 Modules

1.  **Prediction (`prediction.ipynb`)**:
    - Loads the trained logistic regression model and tfidf vectorizer.
    - Applies text preprocessing techniques to new text.
    - Predicts whether the reviews are fake or not using the trained model and saves the results to the csv.
---

## 🌟 Acknowledgements

This project is part of WoC 7.0, developed to demonstrate a comprehensive pipeline for detecting fake reviews using NLP and ML techniques. Special thanks to the mentors for their guidance and support.
