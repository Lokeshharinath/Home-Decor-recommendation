import os
import numpy as np
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D

# Load ResNet50 model with GlobalMaxPooling2D
base_model_hdr = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_hdr = Model(inputs=base_model_hdr.input, outputs=GlobalMaxPooling2D()(base_model_hdr.output))

# Paths to IKEA and Google images directories
ikea_images_path_hdr = "ikea/images"
bing_images_path_hdr = "images"

# Dictionary to store image paths and features
image_paths_hdr = []
features_list_hdr = []

# Function to extract features from an image
def extract_features_hdr(img_path_hdr, model_hdr):
    img_hdr = image.load_img(img_path_hdr, target_size=(224, 224))
    img_array_hdr = image.img_to_array(img_hdr)
    img_array_hdr = np.expand_dims(img_array_hdr, axis=0)
    img_array_hdr = preprocess_input(img_array_hdr)
    features_hdr = model_hdr.predict(img_array_hdr).flatten()
    normalized_features_hdr = features_hdr / np.linalg.norm(features_hdr)
    return normalized_features_hdr

# Extract features from IKEA images
for category_hdr in os.listdir(ikea_images_path_hdr):
    category_path_hdr = os.path.join(ikea_images_path_hdr, category_hdr)
    if os.path.isdir(category_path_hdr):
        for img_name_hdr in os.listdir(category_path_hdr):
            img_path_hdr = os.path.join(category_path_hdr, img_name_hdr)
            try:
                features_hdr = extract_features_hdr(img_path_hdr, model_hdr)
                image_paths_hdr.append(img_path_hdr)
                features_list_hdr.append(features_hdr)
                print(f"Extracted features for IKEA image: {img_name_hdr}")
            except Exception as e_hdr:
                print(f"Error processing image {img_name_hdr} in IKEA: {e_hdr}")

# Extract features from Google images
for category_hdr in os.listdir(google_images_path_hdr):
    category_path_hdr = os.path.join(google_images_path_hdr, category_hdr)
    if os.path.isdir(category_path_hdr):
        for img_name_hdr in os.listdir(category_path_hdr):
            img_path_hdr = os.path.join(category_path_hdr, img_name_hdr)
            try:
                features_hdr = extract_features_hdr(img_path_hdr, model_hdr)
                image_paths_hdr.append(img_path_hdr)
                features_list_hdr.append(features_hdr)
                print(f"Extracted features for Google image: {img_name_hdr}")
            except Exception as e_hdr:
                print(f"Error processing image {img_name_hdr} in Google: {e_hdr}")

# Save the extracted features and image paths
with open("features.pkl", "wb") as f_hdr:
    pickle.dump(features_list_hdr, f_hdr)
with open("img_names.pkl", "wb") as f_hdr:
    pickle.dump(image_paths_hdr, f_hdr)

print("Feature extraction completed and saved.")
