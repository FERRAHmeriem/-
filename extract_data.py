import cv2
import os
from ultralytics import YOLO

# Charger le modèle YOLO
detector = YOLO("yolov8m.pt")  # Remplace avec ton modèle

# Dossier pour sauvegarder les produits extraits
output_dir = "extracted_products"
os.makedirs(output_dir, exist_ok=True)

# Charger l’image du rayonnage
image_path = "rayonnage.jpg"
image = cv2.imread(image_path)

# Détection des produits
results = detector(image , conf = 0.1)

# Sauvegarde des produits détectés
for i, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_product = image[y1:y2, x1:x2]
        cv2.imwrite(f"{output_dir}/product_{i}_{j}.jpg", cropped_product)

print("✅ Extraction terminée ! Vérifie le dossier 'extracted_products'.")
