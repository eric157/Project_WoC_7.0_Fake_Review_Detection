import os
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# --- Path for saving cookies ---
COOKIES_FILE_PATH = os.path.join("..", "checkpoint 3", "amazon_cookies.pkl") # Ensure consistent path with app.py

def generate_amazon_cookies():
    """
    Generates and saves Amazon cookies to bypass login in scraping scripts.

    Instructions:
    1. Run this script. It will open a Chrome browser window.
    2. Manually log in to your Amazon account in the opened browser.
    3. **IMPORTANT:** After successfully logging in on the webpage, come back to your terminal/command prompt
       where you ran this script and press the Enter key.
    4. The script will save your cookies to 'amazon_cookies.pkl'.
    5. You can then use these cookies in your scraping scripts (like app.py) to attempt
       bypassing login.
    """

    # Chrome options (non-headless for manual login)
    chrome_options = Options()
    # chrome_options.add_argument("--headless=new") # REMOVE headless for manual login!
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--window-size=1920,1080")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    chrome_options.add_argument(f"user-agent={user_agent}")

    driver_login_for_cookies = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    product_url = "https://www.amazon.com/Oxford-Notebook-Subject-Assorted-65205/dp/B0B9W6QDPY"  # Replace with ANY Amazon product URL
    print(f"Navigating to: {product_url}")
    print("Please log in to Amazon manually in the browser window that will open.")
    print("After logging in successfully on the webpage, return to this terminal and press Enter.")


    try:
        driver_login_for_cookies.get(product_url)
        input("Press Enter here after you have logged in to Amazon in the browser...") # Pause for manual login

        cookies = driver_login_for_cookies.get_cookies()
        os.makedirs(os.path.dirname(COOKIES_FILE_PATH), exist_ok=True) # Ensure directory exists
        pickle.dump(cookies, open(COOKIES_FILE_PATH, "wb"))
        print(f"Cookies saved to: {COOKIES_FILE_PATH}")

    finally:
        driver_login_for_cookies.quit()


if __name__ == "__main__":
    generate_amazon_cookies()