import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
from autocorrect import Speller
from langdetect import detect, DetectorFactory
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pickle
import logging
from flask_cors import CORS
from flask import send_from_directory
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
DetectorFactory.seed = 0
spell = Speller(lang='en')

app = Flask(__name__)
CORS(app)

# --- Paths ---
MODEL_DIR = os.path.join("..", "checkpoint 2", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join("..", "checkpoint 1", "models", "tfidf_vectorizer.pkl")
FEATURE_NAMES_PATH = os.path.join("..", "checkpoint 1", "models", "tfidf_feature_names.pkl")
COOKIES_FILE_PATH = os.path.join("..", "checkpoint 3", "amazon_cookies.pkl") # Corrected path


# --- Load Model and Vectorizer ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train model first.")
    return joblib.load(MODEL_PATH)

def load_vectorizer():
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}. Please run preprocessing first.")
    return joblib.load(VECTORIZER_PATH)

model = load_model()
vectorizer = load_vectorizer()
feature_names = joblib.load(FEATURE_NAMES_PATH)


# --- Preprocessing Functions ---
def preprocess_text(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|\#\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [spell(word) for word in tokens]
    tokens = [word.encode('ascii', 'ignore').decode('ascii') for word in tokens]
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

def preprocess_new_text(text, rating):
    if not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(f"Feature Names not found: {FEATURE_NAMES_PATH}. Please train vectorizer first")
    if not os.path.exists(VECTORIZER_PATH):
       raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}. Please train vectorizer first")

    try:
        if detect(text) != 'en':
            return pd.DataFrame(columns=feature_names.tolist() + ['rating'])
    except:
         return pd.DataFrame(columns=feature_names.tolist() + ['rating'])

    preprocessed_text = preprocess_text(text)
    tfidf_matrix = vectorizer.transform([preprocessed_text])
    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_features['rating'] = float(rating)
    tfidf_features = tfidf_features.reindex(columns = feature_names.tolist() + ['rating'], fill_value=0)
    return tfidf_features


# --- Scraping Functions (From user provided code with fixes and comments) ---
def is_amazon_url(url):
    """Checks if a URL is a valid amazon.com product URL."""
    amazon_pattern = r"(https?://)?(www.)?amazon\.com/.*"
    return bool(re.match(amazon_pattern, url))

def scrape_amazon_reviews(product_url, max_reviews=50, phone_number="9727715703", password="kuku@1108"): # Placeholders, use env vars!
    """
    Scrapes reviews from an Amazon product page, handling dynamic loading, pagination, and CAPTCHAs.
    Attempts to load cookies first, if available, to bypass login.

    Phone number and password are required for login ONLY if cookies are not found or are expired.
    It's highly recommended to set AMAZON_PHONE_NUMBER and AMAZON_PASSWORD environment variables
    instead of hardcoding them for security.
    """
    logging.info(f"Starting scrape_amazon_reviews for URL: {product_url}, max_reviews: {max_reviews}") # Log max_reviews
    if not is_amazon_url(product_url):
        raise ValueError("Invalid amazon.com product URL.")

    scraped_reviews = []
    review_count = 0
    page_number = 1
    cookies_file_path = COOKIES_FILE_PATH

    # Chrome options for headless mode (to prevent window opening)
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # True headless mode for latest Chrome
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions") # Disable extensions
    chrome_options.add_argument("--window-size=1920,1080") # Set window size

    driver_headless = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        logging.info("Navigating to product page in headless mode.")
        driver_headless.get(product_url)

        try:
            logging.info("Loading cookies from: {}".format(cookies_file_path)) # Log cookie file path
            cookies = pickle.load(open(cookies_file_path, "rb"))
            for cookie in cookies:
                driver_headless.add_cookie(cookie)
            driver_headless.get(product_url) # Re-navigate after setting cookies
            logging.info("Cookies loaded successfully. Attempting to bypass login.")
        except FileNotFoundError:
            logging.warning(f"Cookies file not found at: {cookies_file_path}, proceeding with login.")
        except Exception as cookie_e:
            logging.warning(f"Error loading cookies, proceeding with login. {cookie_e}")


        # --- Login process (only if cookies fail or are not found) ---
        if not os.path.exists(cookies_file_path) or 'cookie_e' in locals(): # Check for cookie load failure
            logging.info("Performing login as cookies were not loaded or not found.")
            driver_login = webdriver.Chrome(service=Service(ChromeDriverManager().install())) # Non-headless for login
            try:
                logging.info("Attempting to log in using phone number and password.")
                driver_login.get(product_url)

                login_link = WebDriverWait(driver_login, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#nav-link-accountList"))
                )
                login_link.click()
                time.sleep(2)

                phone_input = WebDriverWait(driver_login, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#ap_email"))
                )
                phone_input.send_keys(phone_number)

                continue_button = WebDriverWait(driver_login, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#continue"))
                )
                continue_button.click()
                time.sleep(2)

                password_input = WebDriverWait(driver_login, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#ap_password"))
                )
                password_input.send_keys(password)

                sign_in_button = WebDriverWait(driver_login, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#signInSubmit"))
                )
                sign_in_button.click()
                time.sleep(5)

                if driver_login.find_elements(By.CSS_SELECTOR, '#auth-captcha-guess-text'):
                    logging.error("Captcha detected. Please solve it manually in the opened browser.")
                    logging.info("Sleeping for 2 minutes for captcha solving.")
                    time.sleep(120)
                    if driver_login.find_elements(By.CSS_SELECTOR, '#auth-captcha-guess-text'):
                        logging.error("Captcha was not solved. Login failed.")
                        driver_login.quit()
                        driver_headless.quit()
                        return []
                else:
                    logging.info("Captcha was not detected during login.")
                logging.info("Login successful.")

                logging.info("Saving cookies after successful login to: {}".format(cookies_file_path))
                pickle.dump(driver_login.get_cookies(), open(cookies_file_path, "wb"))

                driver_headless.quit() # Quit the initial headless driver
                driver_headless = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) # Re-create headless
                cookies = pickle.load(open(cookies_file_path, "rb")) # Load saved cookies
                for cookie in cookies:
                    driver_headless.add_cookie(cookie)
                driver_headless.get(product_url) # Navigate again in headless

            except Exception as login_e:
                logging.error(f"Login failed: {login_e}")
                driver_login.quit()
                driver_headless.quit()
                return []
            finally:
                driver_login.quit()
                logging.info("Login driver quitted after login process.")


        try:
            logging.info("Attempting to find 'see all reviews' link")
            see_all_link_element = WebDriverWait(driver_headless, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-hook='see-all-reviews-link-foot']"))
             )
            logging.info("Found 'see all reviews' link.")
            see_all_link = see_all_link_element.get_attribute('href')
            driver_headless.get(see_all_link) # Navigate directly to reviews URL
            time.sleep(5)

        except Exception as e:
             logging.error(f"Could not find or click 'see all reviews' link: {e}")
             driver_headless.quit()
             return scraped_reviews

        logging.info("Navigated to review page.")

        while review_count < max_reviews:
          logging.info(f"Fetching reviews from page: {page_number}, current review count: {review_count}") # Log page number and count

          try:
            WebDriverWait(driver_headless, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-hook="review"]'))
             )
          except:
             logging.info("Could not find any reviews in the page after waiting (page {}).".format(page_number))
             break

          soup = BeautifulSoup(driver_headless.page_source, 'html.parser')
          review_elements = soup.select('[data-hook="review"]')

          if not review_elements:
            logging.info(f"No review elements found on page {page_number} using selector '[data-hook=\"review\"]'.")
            break

          logging.info(f"Found {len(review_elements)} review elements on page {page_number}.")

          for review in review_elements:
            if review_count >= max_reviews: # Check limit at the beginning of inner loop
                break # Break inner loop if limit reached
            try:
              text_element = review.select_one('[data-hook="review-body"]')
              rating_element = review.select_one('[data-hook="review-star-rating"] > span.a-icon-alt')

              if text_element and rating_element:
                text = text_element.get_text(strip=True)
                rating_text = rating_element.get_text(strip=True)

                rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
                rating = rating_match.group(1) if rating_match else None

                scraped_reviews.append({"text": text, "rating": rating})
                review_count += 1
              else:
                logging.warning(f"Skipping review due to missing text or rating elements in a review item.")
            except Exception as inner_e:
               logging.error(f"Error parsing individual review: {inner_e}")
          if review_count >= max_reviews:
            break # Break outer loop if limit reached

          try:
              next_button = WebDriverWait(driver_headless, 20).until(
                  EC.presence_of_element_located((By.CSS_SELECTOR, "li.a-last > a"))
              )
              if next_button:
                  logging.info("Found 'next page' button. Clicking...")
                  next_button.click()
                  page_number+=1
                  time.sleep(5)
                  logging.info("Moving to next page (page {}).".format(page_number))
              else:
                  logging.info("Next page button not found. Stopping pagination.")
                  break
          except Exception as next_e:
                logging.info(f"Next page button not found or error clicking: {next_e}. Stopping pagination.")
                break

    except Exception as e:
        logging.error(f"An error occurred during scraping process: {e}")
    finally:
        driver_headless.quit()
        logging.info("Driver quitted after reviews scraping.")

    logging.info(f"scrape_amazon_reviews finished. Scraped {len(scraped_reviews)} reviews.")
    return scraped_reviews


# --- API Endpoint ---
@app.route('/scrape_and_predict', methods=['POST'])
def scrape_and_predict():
    product_url = request.json.get('product_url')
    max_reviews = request.json.get('max_reviews', 50) # Default to 50 if not provided
    if not product_url:
        return jsonify({"error": "Product URL is required"}), 400

    try:
        # --- IMPORTANT: Use environment variables for credentials in production ---
        phone_number = os.environ.get("AMAZON_PHONE_NUMBER") or "YOUR_PHONE_NUMBER" # Use env vars, fallback to placeholder
        password = os.environ.get("AMAZON_PASSWORD") or "YOUR_PASSWORD" # Use env vars, fallback to placeholder


        reviews = scrape_amazon_reviews(product_url, max_reviews=max_reviews, phone_number=phone_number, password=password)
        print("Scraped Reviews:", reviews)
        if not reviews:
            return jsonify({"error": "No reviews scraped", "reviews": []}), 200

        predictions = []
        fake_count = 0
        start_time = time.time()
        for review_data in reviews:
            try:
                processed_features = preprocess_new_text(review_data['text'], review_data['rating'])
                if processed_features.empty:
                    prediction = None
                    probability = None
                    prediction_text = "Language Error/Prediction Failed"
                else:
                    prediction = model.predict(processed_features)[0]
                    probability = model.predict_proba(processed_features)[0]
                    if prediction == 1.0: # Now correct
                      prediction_text = "Real"
                      probability = probability[1]  # Probability of being Real
                    elif prediction == 0.0: # Now correct
                        prediction_text = "Fake"
                        fake_count +=1
                        probability = probability[0] # Probability of being Fake
                    else:
                       prediction_text = "Language Error/Prediction Failed"
                       probability = None

                predictions.append({
                    "review": review_data['text'],
                    "rating": review_data['rating'],
                    "prediction": prediction_text,
                    "probability": str(probability) if probability is not None else None
                })
            except Exception as pred_err:
                logging.error(f"Prediction error for review: {pred_err}")
                predictions.append({"review": review_data['text'], "rating": review_data['rating'], "prediction": "Prediction Error", "probability":None})

        end_time = time.time()
        inference_time = end_time - start_time
        total_reviews = len(predictions)
        fake_percentage = (fake_count/total_reviews) * 100 if total_reviews > 0 else 0
        return jsonify({"reviews_data": predictions, "fake_percentage": fake_percentage, "inference_time": inference_time}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Scraping and prediction error: {e}")
        return jsonify({"error": "Internal server error during scraping and prediction", "details": str(e)}), 500

@app.route('/frontend/<path:filename>')
def frontend(filename):
    static_folder = os.path.join(os.path.dirname(__file__), 'frontend')
    return send_from_directory(static_folder, filename)

@app.route('/')
def home():
    return frontend('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)