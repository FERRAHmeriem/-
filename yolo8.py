import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Charger un modèle YOLOv8 pré-entraîné
model = YOLO("yolov8m.pt"  )  # Modèle moyen (moyen entre vitesse et précision)

# Charger une image de rayonnage
image_path = "rayonnage.jpg"  # Remplace par ton image
image = cv2.imread(image_path)

# Détection des objets
results = model(image ,conf=0.1)

# Dessiner les boîtes de détection sur l'image
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
        label = model.names[int(box.cls[0])]  # Nom de la classe détectée
        confidence = float(box.conf[0])  # Confiance

        # Dessiner la boîte et le label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")  # Hide axes
plt.show()

