import cv2
import os
import albumentations as A

# Input folder containing original images
input_folder = "./initial_data/Photos produits Ramy"
output_root = "./augmented_images"

# Ensure output root exists
os.makedirs(output_root, exist_ok=True)

# Define augmentation pipeline
augmentations = A.Compose([
    A.Affine(scale=(0.8, 1.2), translate_percent=(0.2, 0.2), rotate=(0, 20), shear=(-10, 10), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)
])

# Loop through each image in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    # Ensure it's an image file
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
        continue

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping {img_name}, unable to read image.")
        continue

    # Extract image name without extension
    img_name_no_ext = os.path.splitext(img_name)[0]

    # Create a new folder for this image
    output_folder = os.path.join(output_root)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(20):
        augmented = augmentations(image=image)
        augmented_image = augmented['image']
        output_path = os.path.join(output_folder, f"{img_name_no_ext}_{i}.jpg")
        cv2.imwrite(output_path, augmented_image)

    print(f"Augmented images saved in: {output_folder}")
