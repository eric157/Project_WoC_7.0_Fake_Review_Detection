# 🛡️ Review Sentinel

A robust and user-friendly pipeline for identifying fake reviews using Natural Language Processing (NLP) and Machine Learning (ML). This project covers data preprocessing, advanced feature extraction, sophisticated model training, real-world prediction capabilities, and a web interface for analysis.

---

## ✨ Features

### 🔹 Advanced Text Preprocessing
- **Contraction Expansion**: Converts contractions (e.g., "can't" to "cannot") for text clarity.
- **Noise Removal**: Eliminates punctuation, digits, and stopwords for cleaner text.
- **Text Normalization**: Converts text to lowercase and applies lemmatization for consistency.
- **HTML & JavaScript Handling**: Strips out HTML tags and JavaScript code from reviews.
- **URL, Email, Hashtag Removal**: Cleans text from unnecessary web and social media elements.
- **Spelling Correction**: Corrects misspellings for improved text accuracy.
- **Non-ASCII Character Handling**: Standardizes text to ensure character encoding consistency.
- **Language Detection & Filtering**: Automatically removes reviews that are not in English.
- **Short Text & Noise Handling**: Filters out very short and non-informative text snippets.

### 🔹 Sophisticated Feature Extraction
- **TF-IDF Vectorization**: Converts processed text into numerical vectors using Term Frequency-Inverse Document Frequency.
- **Customizable N-grams**: Captures different text contexts by considering single words (unigrams), pairs (bigrams), and sequences of words.
- **Feature Limits**: Optimizes model performance by controlling the number of features extracted.
- **Frequency Thresholds**: Filters out extremely rare or overly common terms to focus on more informative vocabulary.

### 🔹 Robust Model Training
- **Logistic Regression Model**: Employs an efficient and effective model for binary classification (Real/Fake review detection).
- **Bayesian Hyperparameter Tuning**: Optimizes Logistic Regression model parameters using Bayesian optimization for peak performance.

### 🔹 Comprehensive Evaluation Metrics
- **Performance Metrics**: Provides detailed model evaluation using Accuracy, Precision, Recall, and F1-Score to assess effectiveness.

### 🔹 Efficient Model Persistence
- **Model Saving**: Stores trained Logistic Regression models and the TF-IDF vectorizer using `joblib` for easy loading and reuse.
- **Optimized Model Selection**: Selects and saves the best performing model based on the F1-Score metric.

### 🔹 Dynamic Web Scraping from Amazon
- **Amazon Review Scraping**: Automatically collects product reviews directly from Amazon product pages.
- **User-Defined Review Count**: Allows users to specify the number of reviews to scrape for analysis.
- **Data Storage**: Saves scraped reviews into structured CSV files for further processing and analysis.
- **Cookie-Based Login**: Implements cookie-based login to help bypass Amazon's login prompts and CAPTCHA challenges during scraping.
- **Headless Browser**: Utilizes Selenium with a headless browser for efficient and background web scraping operations.

### 🔹 Enhanced User Interface Design
- **Modern and Aesthetic Frontend**: Features a significantly improved, visually appealing, and modern web interface with a dark theme.
- **Tabbed Navigation**: Organizes features into intuitive tabs for easy access to Amazon Review Analysis and Single Review Prediction.
- **Interactive Metric Boxes**: Displays prediction results and summary metrics in stylish, interactive boxes for a more engaging user experience.

### 🔹 Inference Time Tracking
- **Real-Time Tracking**: Accurately calculates and displays the total time taken for model inference and prediction.

### 🔹 Modular and Scalable Design
- **Checkpoint-Based Structure**: Organizes the project into distinct, manageable modules (checkpoints).
- **Clear Module Separation**: Ensures easy project maintainability, scalability, and allows for focused development on individual components.

---

## 📁 Folder Structure
```
Project_WoC_7.0_Fake_Review_Detection/
├── checkpoint 1/
│ ├── preprocessing.ipynb # Preprocessing and feature extraction notebook
│ ├── data/ # Directory for raw review datasets (CSV files)
│ ├── output/ # Directory for processed datasets (CSV files)
│ ├── models/ # Directory to store TF-IDF vectorizer and feature names (pkl files)
├── checkpoint 2/
│ ├── model_training.ipynb # Model training and evaluation notebook
│ ├── models/ # Directory to store trained model files (pkl files)
│ ├── output/ # Directory for model evaluation results
├── checkpoint 3/
│ ├── scraping.ipynb # Amazon review scraping notebook
│ ├── amazon_cookies.pkl # File to store cookies for Amazon login (NOT tracked by Git - see Security Warning)
│ ├── scraped_reviews.csv # CSV file to store scraped reviews
├── checkpoint 4/
│ ├── app.py # Flask backend application (Python)
│ ├── frontend/ # Directory for web interface files
│ │ ├── index.html # Main webpage (HTML)
│ │ ├── script.js # Frontend logic (JavaScript)
│ │ ├── style.css # Frontend styling (CSS)
│ ├── prediction.ipynb # Model inference demo notebook
├── README.md # Project documentation (this file)
└── .gitignore # File specifying intentionally untracked files (like amazon_cookies.pkl)
└── requirements.txt
```
---

## 📦 Checkpoints Deep Dive

### 📌 Checkpoint 1: Preprocessing & Feature Extraction

- **Loads Raw Data**: Reads review datasets in CSV format.
- **Cleans Text**: Applies advanced preprocessing techniques such as contraction expansion, punctuation removal, stopword filtering, and lemmatization.
- **Feature Engineering**: Uses TF-IDF vectorization to convert text data into numerical vectors.
- **Saves Outputs**: Stores processed datasets and TF-IDF models for further training.

### 📌 Checkpoint 2: Model Training & Evaluation

- **Loads Processed Data**: Retrieves TF-IDF transformed text from Checkpoint 1.
- **Trains Logistic Regression Model**: Implements a machine learning model optimized with Bayesian hyperparameter tuning.
- **Evaluates Performance**: Calculates accuracy, precision, recall, and F1-score.
- **Stores Best Model**: Saves the highest-performing model for future use.

### 📌 Checkpoint 3: Web Scraping Module

- **Uses Selenium & BeautifulSoup**: Automates the extraction of Amazon product reviews.
- **Handles Dynamic Content**: Implements headless browsing and cookie-based authentication.
- **Stores Scraped Reviews**: Saves data in structured CSV format for analysis.

### 📌 Checkpoint 4: Prediction & Web App

- **Flask API**: Serves as the backend for model inference.
- **Interactive Web Interface**: Features an intuitive, dark-themed UI with tabbed navigation.
- **Displays Real/Fake Predictions**: Outputs review authenticity with interactive metric boxes.

---

## 🌟 Acknowledgements

This project is developed as part of **WoC 7.0**, showcasing an end-to-end fake review detection pipeline. Special thanks to mentors for their invaluable guidance, support, and feedback throughout the project development.