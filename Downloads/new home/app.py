import os
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D
import random

# Streamlit layout setup
st.set_page_config(page_title="Home Decor Recommendation System", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True
)

# Load pre-trained model for feature extraction
base_model_hdr = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_hdr = Model(inputs=base_model_hdr.input, outputs=GlobalMaxPooling2D()(base_model_hdr.output))

# Load features and image names
with open("features.pkl", "rb") as f_hdr:
    features_list_hdr = np.array(pickle.load(f_hdr))
with open("img_names.pkl", "rb") as f_hdr:
    img_paths_hdr = pickle.load(f_hdr)

# Navigation Tabs
st.sidebar.title("Navigation")
page_hdr = st.sidebar.radio("Go to", ["Home", "Upload & Recommend", "Shop Links", "Custom Recommendations", "Feedback", "About"])

# Home Page
if page_hdr == "Home":
    st.title("Welcome to the Home Decor Recommendation System")
    st.write(
        "This application helps you find similar home decor items based on an uploaded image. "
        "Upload an image to get recommendations for similar items from our curated database."
    )
    st.image("static/home_banner.jpg", use_container_width=True)

# Upload and Recommend Page
elif page_hdr == "Upload & Recommend":
    st.title("Find Similar Home Decor Items")
    st.write("Upload an image to find similar decor items!")

    uploaded_file_hdr = st.file_uploader("Please upload an image for recommendation", type=["jpg", "jpeg", "png"])

    if uploaded_file_hdr is not None:
        with st.spinner("Processing..."):
            img_path_hdr = os.path.join("uploaded_images", uploaded_file_hdr.name)
            with open(img_path_hdr, "wb") as f_hdr:
                f_hdr.write(uploaded_file_hdr.getbuffer())
            display_image_hdr = Image.open(uploaded_file_hdr)
            st.image(display_image_hdr, caption="Uploaded Image", use_container_width=True)

            def extract_features_hdr(img_path_hdr, model_hdr):
                img_hdr = image.load_img(img_path_hdr, target_size=(224, 224))
                img_array_hdr = image.img_to_array(img_hdr)
                img_array_hdr = np.expand_dims(img_array_hdr, axis=0)
                img_array_hdr = preprocess_input(img_array_hdr)
                features_hdr = model_hdr.predict(img_array_hdr).flatten()
                normalized_features_hdr = features_hdr / np.linalg.norm(features_hdr)
                return normalized_features_hdr

            features_hdr = extract_features_hdr(img_path_hdr, model_hdr)
            neighbors_hdr = NearestNeighbors(n_neighbors=12, algorithm='brute', metric='euclidean')
            neighbors_hdr.fit(features_list_hdr)
            distances_hdr, indices_hdr = neighbors_hdr.kneighbors([features_hdr])

            st.subheader("Here are some recommended items:")
            cols_hdr = st.columns(4)
            feedback_scores_hdr = []
            for i_hdr, idx_hdr in enumerate(indices_hdr[0]):
                recommended_img_path_hdr = img_paths_hdr[idx_hdr]
                try:
                    recommended_img_hdr = Image.open(recommended_img_path_hdr)
                    cols_hdr[i_hdr % 4].image(recommended_img_hdr, caption=f"Recommendation {i_hdr+1}", use_container_width=True)
                    is_relevant_hdr = cols_hdr[i_hdr % 4].radio(
                        f"Is Recommendation {i_hdr+1} relevant?", ("Yes", "No"), key=f"relevance_{i_hdr}"
                    )
                    feedback_scores_hdr.append(1 if is_relevant_hdr == "Yes" else 0)
                except Exception as e_hdr:
                    st.warning(f"Could not load image {recommended_img_path_hdr}.")

            if st.button("Calculate Evaluation Metrics"):
                if feedback_scores_hdr:
                    true_positives_hdr = sum(feedback_scores_hdr)
                    predicted_positives_hdr = sum(feedback_scores_hdr)
                    total_relevant_items_hdr = len(feedback_scores_hdr)

                    precision_hdr = true_positives_hdr / predicted_positives_hdr if predicted_positives_hdr > 0 else 0
                    recall_hdr = true_positives_hdr / total_relevant_items_hdr if total_relevant_items_hdr > 0 else 0
                    f1_score_hdr = (2 * precision_hdr * recall_hdr) / (precision_hdr + recall_hdr) if (precision_hdr + recall_hdr) > 0 else 0
                    accuracy_hdr = true_positives_hdr / total_relevant_items_hdr if total_relevant_items_hdr > 0 else 0

                    st.write(f"**Evaluation Metrics**:")
                    st.write(f"- Precision: {precision_hdr:.2f}")
                    st.write(f"- Recall: {recall_hdr:.2f}")
                    st.write(f"- F1-Score: {f1_score_hdr:.2f}")
                    st.write(f"- Accuracy: {accuracy_hdr:.2f}")
                else:
                    st.warning("Please provide feedback for all recommendations before calculating metrics.")
    else:
        st.info("Upload an image to start finding similar items.")

# Shop Links Page
elif page_hdr == "Shop Links":
    st.title("Shop for Home Decor Items")
    st.write("Find decor items on popular shopping sites like Amazon, Walmart, and Costco.")

    st.subheader("Browse Popular Home Decor Categories:")
    col1_hdr, col2_hdr, col3_hdr = st.columns(3)
    with col1_hdr:
        st.markdown("[Beds on Amazon](https://www.amazon.com/s?k=bed)")
        st.markdown("[Sofas on Amazon](https://www.amazon.com/s?k=sofa)")
        st.markdown("[Dining Tables on Amazon](https://www.amazon.com/s?k=dining+table)")
        st.markdown("[Chairs on Amazon](https://www.amazon.com/s?k=chair)")
        st.markdown("[Cabinets on Amazon](https://www.amazon.com/s?k=cabinet)")
    with col2_hdr:
        st.markdown("[Beds on Walmart](https://www.walmart.com/search/?query=bed)")
        st.markdown("[Sofas on Walmart](https://www.walmart.com/search/?query=sofa)")
        st.markdown("[Dining Tables on Walmart](https://www.walmart.com/search/?query=dining+table)")
        st.markdown("[Chairs on Walmart](https://www.walmart.com/search/?query=chair)")
        st.markdown("[Cabinets on Walmart](https://www.walmart.com/search/?query=cabinet)")
    with col3_hdr:
        st.markdown("[Beds on Costco](https://www.costco.com/beds.html)")
        st.markdown("[Sofas on Costco](https://www.costco.com/sofas.html)")
        st.markdown("[Dining Tables on Costco](https://www.costco.com/dining-tables.html)")
        st.markdown("[Chairs on Costco](https://www.costco.com/chairs.html)")
        st.markdown("[Cabinets on Costco](https://www.costco.com/storage-cabinets.html)")

# Custom Recommendations Page
elif page_hdr == "Custom Recommendations":
    st.title("Personalized Recommendations for Your New Home")
    st.write("Provide details about your new home to get customized decor recommendations.")

    home_type_hdr = st.selectbox("Type of Home", ["Apartment", "Villa", "Townhouse", "Studio"])
    location_hdr = st.text_input("Location (City or Area)")
    home_size_hdr = st.slider("Size of Home (in sq ft)", 500, 5000, step=100)
    num_rooms_hdr = st.number_input("Number of Rooms", min_value=1, max_value=10, step=1)
    occupancy_hdr = st.radio("Occupancy Type", ["Family", "Single", "Couple"])
    preferences_hdr = st.multiselect(
        "Preferences",
        ["Modern", "Traditional", "Minimalist", "Industrial", "Luxury", "Eco-friendly"]
    )

    if st.button("Get Recommendations"):
        with st.spinner("Fetching personalized recommendations..."):
            st.subheader("Based on Your Inputs, Here are 15 Recommendations:")

            recommendations_hdr = []

            if "Modern" in preferences_hdr:
                recommendations_hdr += [
                    "[Modern Sofa - Amazon](https://www.amazon.com/s?k=modern+sofa)",
                    "[Modern Lamp - Walmart](https://www.walmart.com/search/?query=modern+lamp)",
                    "[Modern Bed - Costco](https://www.costco.com/beds.html)"
                ]
            if "Traditional" in preferences_hdr:
                recommendations_hdr += [
                    "[Traditional Rug - Amazon](https://www.amazon.com/s?k=traditional+rug)",
                    "[Traditional Lamp - Walmart](https://www.walmart.com/search/?query=traditional+lamp)",
                    "[Traditional Wall Art - Costco](https://www.costco.com/wall-art.html)"
                ]
            if home_size_hdr <= 1000:
                recommendations_hdr += [
                    "[Space-saving Furniture - Amazon](https://www.amazon.com/s?k=space+saving+furniture)",
                    "[Compact Shelves - Walmart](https://www.walmart.com/search/?query=compact+shelves)",
                    "[Foldable Beds - Costco](https://www.costco.com/foldable-beds.html)"
                ]
            elif 1001 <= home_size_hdr <= 2500:
                recommendations_hdr += [
                    "[Medium-sized Sectional Sofa - Amazon](https://www.amazon.com/s?k=sectional+sofa)",
                    "[6-Seater Dining Table - Walmart](https://www.walmart.com/search/?query=6-seater+dining+table)",
                    "[Elegant Chandeliers - Costco](https://www.costco.com/chandeliers.html)"
                ]
            else:
                recommendations_hdr += [
                    "[Luxury Sofa Set - Amazon](https://www.amazon.com/s?k=luxury+sofa+set)",
                    "[Grand Dining Tables - Walmart](https://www.walmart.com/search/?query=grand+dining+tables)",
                    "[Large Decorative Mirrors - Costco](https://www.costco.com/decorative-mirrors.html)"
                ]
            if "Eco-friendly" in preferences_hdr:
                recommendations_hdr += [
                    "[Eco-friendly Bamboo Chairs - Amazon](https://www.amazon.com/s?k=eco-friendly+bamboo+chairs)",
                    "[Sustainable Rugs - Walmart](https://www.walmart.com/search/?query=sustainable+rugs)",
                    "[Recycled Glass Decor - Costco](https://www.costco.com/recycled-glass-decor.html)"
                ]

            if len(recommendations_hdr) >= 15:
                recommendations_hdr = random.sample(recommendations_hdr, 15)
            else:
                while len(recommendations_hdr) < 15:
                    recommendations_hdr += random.choices(recommendations_hdr, k=15 - len(recommendations_hdr))

            for rec_hdr in recommendations_hdr[:15]:
                st.markdown(f"- {rec_hdr}")

# Feedback Page
elif page_hdr == "Feedback":
    st.title("We Value Your Feedback!")
    feedback_hdr = st.text_area("Enter your feedback here")
    user_name_hdr = st.text_input("Name (optional)")
    email_hdr = st.text_input("Email (optional)")
    if st.button("Submit Feedback"):
        if feedback_hdr:
            st.success("Thank you for your feedback!")
            with open("feedback.txt", "a") as f_hdr:
                f_hdr.write(f"Name: {user_name_hdr}\nEmail: {email_hdr}\nFeedback: {feedback_hdr}\n{'-'*40}\n")
        else:
            st.warning("Please enter your feedback before submitting.")

# About Page
elif page_hdr == "About":
    st.title("About This Project")
    st.write("This project uses a ResNet50 model to help users find visually similar home decor items.")
