import os
import time
import requests
from PIL import Image
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import io
import base64

def fetch_images_from_bing(keyword_hdr, chrome_driver_path_hdr, output_dir_hdr, image_limit_hdr=10):
    # Format the search term to make it suitable for a URL
    formatted_keyword_hdr = quote(keyword_hdr)
    destination_hdr = os.path.join(output_dir_hdr, keyword_hdr)
    os.makedirs(destination_hdr, exist_ok=True)
    chrome_options_hdr = Options()
    chrome_options_hdr.add_argument("--headless")
    driver_service_hdr = Service(chrome_driver_path_hdr)
    browser_hdr = webdriver.Chrome(service=driver_service_hdr, options=chrome_options_hdr)

    try:
        # Open the Bing Images page for the search term
        search_url_hdr = f"https://www.bing.com/images/search?q={formatted_keyword_hdr}"
        browser_hdr.get(search_url_hdr)
        for _ in range(10):
            browser_hdr.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        # Identify all image elements on the page
        image_elements_hdr = browser_hdr.find_elements(By.CSS_SELECTOR, "img.mimg")
        saved_count_hdr = 0
        for idx_hdr, img_element_hdr in enumerate(image_elements_hdr):
            if saved_count_hdr >= image_limit_hdr:
                break
            image_source_hdr = img_element_hdr.get_attribute("src")
            if image_source_hdr:
                file_path_hdr = os.path.join(destination_hdr, f"{keyword_hdr}_{idx_hdr + 1}.jpg")
                # Handle HTTP and Base64 images separately
                if image_source_hdr.startswith('http'):
                    response_hdr = requests.get(image_source_hdr)
                    with open(file_path_hdr, "wb") as file_hdr:
                        file_hdr.write(response_hdr.content)
                        saved_count_hdr += 1
                elif image_source_hdr.startswith('data:image'):
                    base64_data_hdr = image_source_hdr.split('base64,')[1]
                    decoded_image_hdr = Image.open(io.BytesIO(base64.b64decode(base64_data_hdr)))
                    decoded_image_hdr.save(file_path_hdr)
                    saved_count_hdr += 1
                print(f"[INFO] Downloaded {saved_count_hdr}/{image_limit_hdr} for '{keyword_hdr}'")
    except Exception as error_hdr:
        # Print any errors encountered during execution
        print(f"[ERROR] {str(error_hdr)}")
    finally:
        # Close the browser after the operation completes
        browser_hdr.quit()
        print(f"[INFO] Image download for '{keyword_hdr}' completed. Images saved at: {destination_hdr}")

if __name__ == "__main__":
    # Define paths and keywords for scraping
    chrome_driver_path_hdr = "/usr/local/bin/chromedriver"
    output_directory_hdr = "./downloads"
    search_keywords_hdr = ["armchair", "bedframe", "storage unit", "wall decor", "area rug", "window drapes", "wax candles", "pendant lights"]
    max_images_hdr = 200
    # Start the image download process for each keyword
    for word_hdr in search_keywords_hdr:
        fetch_images_from_bing(word_hdr, chrome_driver_path_hdr, output_directory_hdr, image_limit_hdr=max_images_hdr)
