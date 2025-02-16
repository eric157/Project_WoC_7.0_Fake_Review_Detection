# 🛡️ Review Sentinel

A robust and user-friendly pipeline for identifying fake reviews using Natural Language Processing (NLP) and Machine Learning (ML).

---

## 🎬 Execution Video

[![Execution Video](assets/execution_thumbnail.png)](https://www.youtube.com/watch?v=Ozqh9fH66tE)

---

## ✨ Features

- **Advanced Text Preprocessing:**  
  Contraction expansion, noise removal, text normalization, HTML & JavaScript handling, spelling correction, language filtering, and more.
  
- **Sophisticated Feature Extraction:**  
  TF-IDF vectorization with customizable n-grams, feature limits, and frequency thresholds.
  
- **Robust Model Training:**  
  Logistic Regression model enhanced with Bayesian hyperparameter tuning and evaluated using metrics like accuracy, precision, recall, and F1-score.
  
- **Dynamic Web Scraping:**  
  Automated Amazon review scraping using Selenium and BeautifulSoup with headless browser support and cookie-based authentication.
  
- **Interactive Web Interface:**  
  A Flask-based backend paired with a modern dark-themed UI for real-time prediction display.

---

## 📦 Checkpoints Overview

### Checkpoint 1: Preprocessing & Feature Extraction
- Loads raw review data, cleans text through advanced preprocessing techniques, and converts it into numerical vectors using TF-IDF.
- Outputs are saved for subsequent model training.

### Checkpoint 2: Model Training & Evaluation
- Trains a Logistic Regression model with Bayesian hyperparameter tuning.
- Evaluates the model using accuracy, precision, recall, and F1-score, saving the best-performing model.

### Checkpoint 3: Web Scraping
- Automates the extraction of Amazon reviews with Selenium & BeautifulSoup.
- Handles dynamic content via headless browsing and cookie-based login.
- Stores scraped reviews in CSV format.

### Checkpoint 4: Web App & Prediction
- Provides a Flask API for real-time review analysis.
- Features an intuitive, dark-themed UI with interactive metric boxes to display predictions.

---

## 🌟 Acknowledgements

This project is developed as part of **WoC 7.0**, showcasing an end-to-end fake review detection pipeline. Special thanks to our mentors and contributors for their invaluable guidance and support throughout the project development.