import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load MobileNetV2 as feature extractor (pretrained on ImageNet)
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Path to folders
RAMY_FOLDER = "Sorted_Images2"  # The folder containing multiple subfolders (categories)
DETECTED_PRODUCTS_FOLDER = "extracted_products"  # Folder with detected objects from shelf images

# Define a function to extract features from an image
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Step 1: Extract features for Ramy products
ramy_features = []
ramy_labels = []

print("Extracting features from Ramy products...")
for category in os.listdir(RAMY_FOLDER):
    category_path = os.path.join(RAMY_FOLDER, category)
    if os.path.isdir(category_path):
        for img_name in tqdm(os.listdir(category_path)):
            img_path = os.path.join(category_path, img_name)
            features = extract_features(img_path)
            if features is not None:
                ramy_features.append(features)
                ramy_labels.append(category)  # Store category label (optional)

ramy_features = np.array(ramy_features)

# Step 2: Classify detected products in shelf images
results = []
ramy_count = 0
total_detected = 0
THRESHOLD = 0.85  # Adjust based on performance

print("Classifying detected products...")
for img_name in tqdm(os.listdir(DETECTED_PRODUCTS_FOLDER)):
    img_path = os.path.join(DETECTED_PRODUCTS_FOLDER, img_name)
    detected_features = extract_features(img_path)
    
    if detected_features is not None:
        total_detected += 1
        similarities = cosine_similarity([detected_features], ramy_features)  # Compare with Ramy dataset
        max_similarity = np.max(similarities)  # Get highest similarity score
        
        if max_similarity >= THRESHOLD:
            classification = "Ramy"
            ramy_count += 1
        else:
            classification = "Other brand"
        
        results.append((img_name, classification, max_similarity))

# Step 3: Calculate percentage of Ramy products
ramy_percentage = (ramy_count / total_detected) * 100 if total_detected > 0 else 0
print(f"Percentage of Ramy products: {ramy_percentage:.2f}%")

# Step 4: Save results
with open("classification_results.txt", "w") as f:
    for res in results:
        f.write(f"{res[0]} - {res[1]} (Similarity: {res[2]:.2f})\n")

print("Classification completed! Results saved in 'classification_results.txt'.")
