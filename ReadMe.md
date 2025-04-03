
# 🏠 Home Decor Recommendation System

This project is a Home Decor Recommendation System built with Python, Streamlit, and custom image scraping + feature extraction. Users can select a home decor item and receive visually similar recommendations based on deep learning features.

---

## 🚀 Features

- Upload or select a decor image
- Get visually similar recommendations using precomputed features
- Web interface built with Streamlit
- Scrapes decor images using BingImageScraper
- Stores and handles user feedback
- Uses cosine similarity on extracted image features (ResNet-based)

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit app
├── BingImageScraper.py    # Script to scrape decor images from Bing
├── feature_extraction.py  # Extract features using pretrained model
├── features.pkl            # Precomputed features of decor images
├── img_names.pkl           # Corresponding image filenames
├── feedback.txt            # Collected user feedback
```

---

## 🛠️ Requirements

- Python 3.8+
- Streamlit
- TensorFlow / Keras
- OpenCV
- Requests
- BeautifulSoup4
- PIL
- Scikit-learn

Install using:

```bash
pip install -r requirements.txt
```

---

## 🖼️ How It Works

1. Images are scraped using `BingImageScraper.py`.
2. Features are extracted using a CNN (e.g., ResNet) via `feature_extraction.py`.
3. Features are stored in `features.pkl` and matched using cosine similarity.
4. The user selects an image in `app.py`, and similar images are recommended.
5. User feedback is logged to `feedback.txt`.

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🧠 Future Improvements

- Add user login and personalization
- Integrate ML model to learn from feedback
- Deploy on cloud (e.g., Streamlit Cloud or AWS)

---

## 📝 Feedback

User feedback is stored in `feedback.txt`. Feel free to test and contribute!

---

## 👤 Author

**Lokesh Harinath**  
Salesforce Certified | ML Enthusiast  
[LinkedIn](https://linkedin.com/in/lokesh-harinath-a8b21b195) | [GitHub](https://github.com/Lokeshharinath)

---
