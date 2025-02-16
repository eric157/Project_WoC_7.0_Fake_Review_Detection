# 🛡️ Review Sentinel

A robust and user-friendly pipeline for identifying fake reviews using Natural Language Processing (NLP) and Machine Learning (ML).

---

## 🎬 Execution Video

[![Watch the Execution Video](https://img.shields.io/badge/Execution%20Video-Click%20Here-blue?style=for-the-badge)](https://youtu.be/Ozqh9fH66tE)


---

## ✨ Features

### Advanced Text Preprocessing
- Contraction Expansion: Converts contractions (e.g., "can't" to "cannot") for text clarity.
- Noise Removal: Eliminates punctuation, digits, and stopwords for cleaner text.
- Text Normalization: Converts text to lowercase and applies lemmatization for consistency.
- HTML & JavaScript Handling: Strips out HTML tags and JavaScript code from reviews.
- URL, Email, Hashtag Removal: Cleans text from unnecessary web and social media elements.
- Spelling Correction: Corrects misspellings for improved text accuracy.
- Non-ASCII Character Handling: Standardizes text to ensure character encoding consistency.
- Language Detection & Filtering: Automatically removes non-English reviews.
- Short Text & Noise Handling: Filters out very short and non-informative text snippets.

### Sophisticated Feature Extraction
- TF-IDF Vectorization: Converts processed text into numerical vectors.
- Customizable N-grams: Captures different text contexts using unigrams, bigrams, and more.
- Feature Limits: Optimizes model performance by controlling the number of features.
- Frequency Thresholds: Filters out extremely rare or overly common terms.

### Robust Model Training
- Logistic Regression Model: Employs an efficient model for binary classification (Real/Fake review detection).
- Bayesian Hyperparameter Tuning: Optimizes model parameters for peak performance.
- Comprehensive Evaluation Metrics: Uses Accuracy, Precision, Recall, and F1-Score to assess effectiveness.

### Dynamic Web Scraping
- Amazon Review Scraping: Automatically collects reviews from Amazon product pages.
- User-Defined Review Count: Allows users to specify the number of reviews to scrape.
- Data Storage: Saves scraped reviews in structured CSV files.
- Cookie-Based Login: Implements cookie-based authentication to bypass login prompts.
- Headless Browser: Utilizes Selenium for efficient, background scraping.

### Enhanced Web Interface & Real-Time Inference
- Flask API: Serves as the backend for model inference.
- Modern, Aesthetic UI: Features a dark-themed interface with tabbed navigation.
- Interactive Metric Boxes: Displays prediction results and summary metrics.
- Inference Time Tracking: Accurately calculates and displays the total inference time.

---

## 📁 Folder Structure

```
Project_WoC_7.0_Fake_Review_Detection/
├── checkpoint 1/
│   ├── preprocessing.ipynb   # Preprocessing and feature extraction notebook
│   ├── data/                 # Raw review datasets (CSV files)
│   ├── output/               # Processed datasets (CSV files)
│   ├── models/               # TF-IDF vectorizer and feature names (pkl files)
├── checkpoint 2/
│   ├── model_training.ipynb  # Model training and evaluation notebook
│   ├── models/               # Trained model files (pkl files)
│   ├── output/               # Model evaluation results
├── checkpoint 3/
│   ├── scraping.ipynb        # Amazon review scraping notebook
│   ├── amazon_cookies.pkl    # File to store cookies for Amazon login (not tracked by Git)
│   ├── scraped_reviews.csv   # CSV file to store scraped reviews
├── checkpoint 4/
│   ├── app.py                # Flask backend application
│   ├── frontend/             # Directory for web interface files
│   │   ├── index.html        # Main webpage (HTML)
│   │   ├── script.js         # Frontend logic (JavaScript)
│   │   ├── style.css         # Frontend styling (CSS)
│   ├── prediction.ipynb      # Model inference demo notebook
├── README.md                 # Project documentation
├── .gitignore                # Specifies intentionally untracked files (e.g., amazon_cookies.pkl)
└── requirements.txt          # Project dependencies
```

---

## 📦 Checkpoints Deep Dive

### Checkpoint 1: Preprocessing & Feature Extraction
- Loads raw review datasets.
- Applies advanced text preprocessing (contraction expansion, noise removal, normalization, etc.).
- Converts text data into numerical vectors using TF-IDF.
- Saves processed data and TF-IDF models for future training.

### Checkpoint 2: Model Training & Evaluation
- Loads TF-IDF transformed text from Checkpoint 1.
- Trains a Logistic Regression model with Bayesian hyperparameter tuning.
- Evaluates model performance using Accuracy, Precision, Recall, and F1-Score.
- Stores the best-performing model for future predictions.

### Checkpoint 3: Web Scraping Module
- Uses Selenium and BeautifulSoup to scrape Amazon product reviews.
- Handles dynamic content, cookie-based authentication, and headless browsing.
- Saves scraped reviews in structured CSV format for analysis.

### Checkpoint 4: Prediction & Web App
- Implements a Flask API for real-time model inference.
- Features an interactive, dark-themed web interface.
- Displays prediction results with engaging metric boxes.
- Tracks and displays the total time taken for inference.

---

## 🌟 Acknowledgements

Developed as part of **WoC 7.0**. Special thanks to the mentors for their guidance and support.