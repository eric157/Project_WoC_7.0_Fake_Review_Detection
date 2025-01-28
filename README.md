# 🛡️ Fake Review Detector

A robust and user-friendly pipeline for identifying fake reviews using Natural Language Processing (NLP) and Machine Learning (ML). This project encompasses data preprocessing, advanced feature extraction, sophisticated model training, and real-world prediction capabilities. It also features a web scraping module to gather review data directly from Amazon and a user-friendly website interface to analyze product reviews.

---

## ✨ Features

-   **Advanced Text Preprocessing**:
    -   **Contraction Expansion**: Converts contractions (e.g., "can't" to "cannot") for text clarity.
    -   **Noise Removal**: Eliminates punctuation, digits, and common stopwords to focus on meaningful content.
    -   **Text Normalization**: Transforms text to lowercase and applies lemmatization to standardize words.
    -   **HTML & JavaScript Handling**: Strips out HTML tags and JavaScript code from review text.
    -   **URL, Email, Hashtag Removal**: Cleans text by removing URLs, email addresses, and hashtags.
    -   **Spelling Correction**: Corrects spelling errors to improve text quality.
    -   **Non-ASCII Character Handling**: Manages and removes non-ASCII characters for text consistency.
    -   **Language Detection & Filtering**: Identifies and filters out reviews not written in English.
    -   **Short Text & Noise Handling**: Filters out very short texts or noisy data that may not be informative.

-   **Sophisticated Feature Extraction (TF-IDF Vectorization)**:
    -   **TF-IDF Conversion**: Transforms preprocessed text into numerical vectors using Term Frequency-Inverse Document Frequency.
    -   **Customizable N-grams**: Offers configurable n-gram ranges to capture varying levels of text context.
    -   **Feature Limits**: Allows setting limits on the number of features to optimize model performance and reduce dimensionality.
    -   **Frequency Thresholds**: Configurable frequency thresholds to filter out extremely rare or common terms.

-   **Robust Model Training (Logistic Regression)**:
    -   **Logistic Regression Model**: Employs a Logistic Regression model, known for its effectiveness in binary classification tasks.
    -   **Bayesian Hyperparameter Tuning**: Utilizes Bayesian optimization to fine-tune model hyperparameters for optimal performance.

-   **Comprehensive Evaluation Metrics**:
    -   **Performance Reporting**: Provides detailed evaluation metrics including Accuracy, Precision, Recall, and F1-Score to assess model effectiveness.

-   **Efficient Model Persistence**:
    -   **Model Saving**: Saves the trained Logistic Regression model and TF-IDF vectorizer using `joblib` for efficient future use and deployment.

-   **Optimized Model Selection**:
    -   **F1-Score Based Selection**: Selects the best-performing model based on the highest F1-Score achieved on the validation dataset, ensuring a balance between precision and recall.

-   **Dynamic Web Scraping from Amazon**:
    -   **Amazon Review Scraping**: Scrapes product reviews directly from Amazon product pages, handling dynamic content loading and pagination.
    -   **Data Storage**: Saves scraped review data into structured CSV files, ready for analysis and model input.
    -   **Cookie-Based Login**: Implements cookie handling to potentially bypass login and CAPTCHA challenges, enhancing scraping efficiency.
    -   **Headless Browser**: Utilizes a headless Chrome browser via Selenium for efficient and background scraping operations.

-   **User-Friendly Prediction Interface**:
    -   **Web-Based Prediction**: Features a Flask-based web application to provide an interactive interface for users to analyze reviews.
    -   **Real-Time Analysis**: Allows users to input Amazon product URLs and receive real-time analysis of reviews.
    -   **Review Summarization**: Displays key summary metrics such as Average Review Rating and Review Count.
    -   **Intuitive Review Display**: Presents individual reviews with clear predictions (Real or Fake) for easy interpretation.
    -   **Elegant Frontend Design**: Offers a professionally designed, responsive frontend for an optimal user experience.

-   **Modular and Scalable Design**:
    -   **Checkpoint-Based Structure**: Project is organized into modular checkpoints, promoting maintainability, scalability, and ease of customization.
    -   **Clear Module Separation**: Each checkpoint (Preprocessing, Model Training, Scraping, Prediction) is designed as a distinct module, facilitating independent updates and enhancements.

---

## 📁 Folder Structure
content_copy
download
Use code with caution.
Markdown
```
Project_WoC_7.0_Fake_Review_Detection/
├── checkpoint 1/
│ ├── preprocessing.ipynb # Jupyter notebook for preprocessing and feature extraction
│ ├── data/ # Input datasets (CSV files)
│ ├── output/ # Preprocessed datasets (CSV files)
│ └── models/ # Saved TF-IDF vectorizer and feature names (pkl files)
├── checkpoint 2/
│ ├── model_training.ipynb # Jupyter notebook for model training and evaluation
│ ├── models/ # Trained Logistic Regression model (pkl file)
│ └── output/ # Model evaluation metrics and results
├── checkpoint 3/
│ ├── scraping.ipynb # Jupyter notebook for web scraping from Amazon
│ └── amazon_cookies.pkl # File to store browser cookies for login bypass
│ └── scraped_reviews.csv # Scraped reviews dataset (CSV file)
└── checkpoint 4/
├── app.py # Flask backend application for website
├── frontend/ # Frontend files for the website
│ ├── index.html # Main HTML file
│ ├── script.js # JavaScript logic for frontend interactivity
│ └── style.css # CSS stylesheet for frontend design
├── prediction.ipynb # Jupyter notebook for prediction on scraped data (demonstration)
└── README.md # Project documentation (this file)
```
---

## 🚀 Usage

### Checkpoint 1: Data Preprocessing and Feature Extraction

1. **Prepare Input Data**:
   - Place your raw review dataset (CSV format) inside the `checkpoint 1/data/` directory.
   - Ensure your CSV file includes a column named `text` for review text and `rating` for review ratings.

2. **Execute Preprocessing**:
   - Open and run the `preprocessing.ipynb` Jupyter notebook located in `checkpoint 1/`.
   - Follow the notebook instructions to preprocess your data and extract TF-IDF features.

3. **Outputs**:
     - `FakeReviewDataPreprocessed.csv`: Preprocessed dataset saved to `checkpoint 1/output/`.
     - `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer model in `checkpoint 1/models/`.
     - `tfidf_feature_names.pkl`: Saved TF-IDF feature names in `checkpoint 1/models/`.

---

### Checkpoint 2: Model Training and Evaluation

1. **Verify Checkpoint 1 Completion**:
   - Confirm that `FakeReviewDataPreprocessed.csv`, `tfidf_vectorizer.pkl`, and `tfidf_feature_names.pkl` are present in `checkpoint 1/output/` and `checkpoint 1/models/` directories, respectively.

2. **Train and Evaluate Model**:
   - Open and run the `model_training.ipynb` Jupyter notebook from `checkpoint 2/`.
   - The notebook will train a Logistic Regression model using the preprocessed data and evaluate its performance.

3. **Outputs**:
     - `logistic_regression_model.pkl`: Trained Logistic Regression model saved to `checkpoint 2/models/`.
     - Model evaluation metrics (accuracy, precision, recall, F1-score) will be displayed in the notebook output.

---

### Checkpoint 3: Web Scraping from Amazon

1.  **Execute Web Scraping Script**:
    -   Open and run the `scraping.ipynb` Jupyter notebook located in `checkpoint 3/`.
    -   When prompted, enter the Amazon product URL you wish to scrape reviews from.
    -   Optionally, provide your Amazon account phone number and password if cookie-based login fails (or to save cookies for future use).

2.  **Outputs**:
        -   `amazon_cookies.pkl`: Saved cookies file in `checkpoint 3/` (if login is successful and cookies are saved).
        -   `scraped_reviews.csv`: CSV file containing scraped reviews, saved in the `checkpoint 3/` directory.

---

### Checkpoint 4: Prediction on Scraped Reviews & Website Interface

1.  **Verify Checkpoint 1, 2 & 3 Completion**:
        - Ensure outputs from Checkpoints 1, 2, and 3 are correctly generated and located in their respective directories.

2.  **Run Flask Backend**:
    -   Navigate to the `checkpoint 4/` directory in your terminal.
    -   Run the Flask application using the command: `python app.py`
    -   The Flask server will start, typically running at `http://127.0.0.1:5000/`.

3.  **Access Website Frontend**:
    -   Open your web browser and go to `http://127.0.0.1:5000/` or `http://127.0.0.1:5000/frontend/index.html`.
    -   Enter an Amazon Product URL in the input field and click "Analyze".
    -   The website will display the analysis summary (Average Rating, Review Count) and a list of scraped reviews with predictions (Real or Fake).

4.  **Run Prediction Notebook (Optional)**:
    -   Alternatively, to run prediction directly in a notebook (without the website), open and execute `prediction.ipynb` from `checkpoint 4/`.
    -   This notebook will load the trained model, preprocess the `scraped_reviews.csv` data, and print predictions.

---

## 📦 Modules

### Checkpoint 1 Modules

1.  **Preprocessing (`preprocessing.ipynb`)**:
    -   Implements comprehensive text preprocessing steps: contraction expansion, noise removal, text normalization, HTML/JavaScript handling, URL/email/hashtag removal, spelling correction, non-ASCII character handling, and language detection/filtering.
    -   Performs TF-IDF vectorization on preprocessed text, converting text data into numerical features suitable for machine learning models.
    -   Saves the trained TF-IDF vectorizer (`tfidf_vectorizer.pkl`) and extracted feature names (`tfidf_feature_names.pkl`) for consistent feature transformation in subsequent steps.

---

### Checkpoint 2 Modules

1.  **Model Training (`model_training.ipynb`)**:
    -   Loads the preprocessed dataset and TF-IDF features generated in Checkpoint 1.
    -   Trains a Logistic Regression model, optimizing hyperparameters using Bayesian optimization to maximize performance.
    -   Evaluates the trained model using key metrics: accuracy, precision, recall, and F1-score, providing a detailed performance analysis.
    -   Persists the best-trained Logistic Regression model to disk (`logistic_regression_model.pkl`) for deployment and future predictions.

---

### Checkpoint 3 Modules

1.  **Web Scraping (`scraping.ipynb`)**:
    -   Automates the process of scraping product reviews from Amazon product pages using Selenium and BeautifulSoup.
    -   Handles dynamic web content and pagination to ensure comprehensive review collection.
    -   Implements cookie management to maintain session persistence and potentially bypass login or CAPTCHA challenges.
    -   Saves scraped review data into a structured CSV format (`scraped_reviews.csv`), facilitating easy integration with the prediction pipeline.

---

### Checkpoint 4 Modules

1.  **Prediction (`prediction.ipynb` and `app.py`)**:
    -   **`prediction.ipynb`**: Demonstrates prediction on new, scraped data in a notebook environment, loading the trained model and vectorizer to classify reviews as fake or real.
    -   **`app.py` & Frontend**:
        -   Deploys a Flask web application (`app.py`) that integrates the scraping, preprocessing, and prediction modules, providing a user-friendly website interface.
        -   The frontend (HTML, CSS, JavaScript in `frontend/`) allows users to input Amazon product URLs and receive real-time analysis of reviews, including summary metrics and individual review predictions, creating an interactive and accessible tool.

---

## 🌟 Acknowledgements

This project is developed as part of the WoC 7.0 initiative, showcasing an end-to-end pipeline for fake review detection using state-of-the-art NLP and ML techniques. Special thanks to the mentors for their guidance and support.