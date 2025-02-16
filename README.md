# 🛡️ Review Sentinel  

A robust and user-friendly pipeline for identifying fake reviews using Natural Language Processing (NLP) and Machine Learning (ML).  

---

## 🎬 Execution Video  

Click the link below to watch the execution video directly from this repository:  

▶ **[Watch Execution Video](assets/execution.mp4)** 

## ✨ Features  
### 🔹 Advanced Text Preprocessing  
- **Contraction Expansion**  
- **Noise Removal**  
- **Text Normalization**  
- **HTML & JavaScript Handling**  
- **Spelling Correction**  
- **Language Detection & Filtering**  

### 🔹 Sophisticated Feature Extraction  
- **TF-IDF Vectorization**  
- **Customizable N-grams**  
- **Feature Limits & Frequency Thresholds**  

### 🔹 Robust Model Training  
- **Logistic Regression Model**  
- **Bayesian Hyperparameter Tuning**  
- **Performance Metrics (Accuracy, Precision, Recall, F1-Score)**  

### 🔹 Dynamic Web Scraping  
- **Amazon Review Scraping**  
- **Headless Browser Support**  
- **Cookie-Based Login**  

### 🔹 Web Interface & Real-Time Inference  
- **Flask Backend**  
- **Modern UI with Dark Theme**  
- **Interactive Prediction Results**  

---

## 📦 Checkpoints Deep Dive  

### 📌 **Checkpoint 1: Preprocessing & Feature Extraction**  
- Loads raw review datasets.  
- Cleans text and applies NLP preprocessing.  
- Converts text into numerical vectors using TF-IDF.  
- Saves processed data for training.  

### 📌 **Checkpoint 2: Model Training & Evaluation**  
- Loads processed text features.  
- Trains Logistic Regression with Bayesian tuning.  
- Evaluates performance using accuracy, precision, recall, and F1-score.  
- Stores the best-performing model.  

### 📌 **Checkpoint 3: Web Scraping**  
- Uses Selenium & BeautifulSoup for Amazon review scraping.  
- Handles dynamic content and login authentication.  
- Saves scraped reviews in a structured format.  

### 📌 **Checkpoint 4: Web App & Prediction**  
- Flask API for real-time review analysis.  
- Interactive frontend with a modern dark theme.  
- Displays predictions with user-friendly metric boxes.  

---

## 🌟 Acknowledgements  

This project is developed as part of **WoC 7.0**, showcasing an end-to-end fake review detection pipeline. Special thanks to mentors for their invaluable guidance, support, and feedback throughout the project development.  