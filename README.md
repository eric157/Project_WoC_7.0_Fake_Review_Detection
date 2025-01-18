# 📚 Fake Review Detection Preprocessing Pipeline

A modular and scalable pipeline for text data preprocessing and feature extraction using TF-IDF. This project is designed to handle tasks essential in natural language processing (NLP) workflows, including text cleaning, lemmatization, stopword removal, and vectorization.

---

## 🗂 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [Usage](#-usage)
- [Modules](#-modules)
  - [Preprocessing](#1-preprocessing-preprocessingpy)
  - [Vectorization](#2-vectorization-vectorizationpy)
  - [Main Script](#3-main-script-mainpy)

---

## 📝 Overview

This pipeline preprocesses a dataset of reviews and prepares it for analysis or modeling. Key features include:

- Text preprocessing (e.g., contraction expansion, tokenization, lemmatization).
- Feature extraction via TF-IDF vectorization.
- Modularized components for maintainability and scalability.

---

## ✨ Features

- **Clean and Normalize Text**: Includes expansion of contractions, removal of punctuation and digits, tokenization, and lemmatization.
- **TF-IDF Feature Extraction**: Converts text into numerical features using unigrams and bigrams.
- **Scalable Design**: Modular architecture to extend and customize the pipeline for different use cases.

---

## 📁 Folder Structure

The project directory is organized as follows:

```plaintext
Project_WoC_7.0_Fake_Review_Detection/
└── checkpoint_1/
    ├── preprocessing.py       # Functions for text preprocessing (cleaning, lemmatization, etc.)
    ├── vectorization.py       # TF-IDF vectorization logic for feature extraction
    ├── main.py                # Main script to run the entire pipeline
    ├── data/                  # Folder to store input datasets (e.g., CSV files with reviews)
    ├── output/                # Folder to store the processed datasets and results
└── README.md              # Project documentation (this file)
```
---

## 🚀 Usage

1. **Prepare the Input Dataset**:
   - Place your dataset in the `data/` folder.
   - Ensure the file is a CSV with a text column named `text` (or update `TEXT_COLUMN` in `main.py`).

2. **Run the Pipeline**:
   Execute the pipeline by running:

   ```bash
   python checkpoint_1/main.py
   ```

3. **Output**:
   The processed dataset will be saved in the `output/` folder as `FakeReviewDataPreprocessed.csv`.

---

## 📦 Modules

### **1. Preprocessing (`preprocessing.py`)**

Handles text cleaning and normalization:

- Expands contractions (e.g., "can't" → "cannot").
- Converts text to lowercase.
- Removes punctuation and digits.
- Tokenizes, removes stopwords, and applies lemmatization.

### **2. Vectorization (`vectorization.py`)**

Extracts features from preprocessed text:

- Uses TF-IDF vectorization to convert text into numerical data.
- Configurable parameters include:
  - Maximum features (`max_features`).
  - N-gram range (`ngram_range`).
  - Frequency thresholds (`max_df`, `min_df`).

### **3. Main Script (`main.py`)**

Orchestrates the workflow:

- Loads the dataset.
- Applies preprocessing and vectorization.
- Saves the final dataset to the output folder.
