import os
import shutil

# Set the path to your images folder
image_folder = "./augmented_images"

# Create a destination folder
output_folder = os.path.join("Sorted_Images2")
os.makedirs(output_folder, exist_ok=True)

# Loop through images in the folder
for image in os.listdir(image_folder):
    if image.endswith((".png", ".jpg", ".jpeg", ".gif")):
        # Extract category (assuming it's before the first underscore 

        if image.startswith('Cannete'):
            category_folder = os.path.join(output_folder, "Cannete")
        elif image.startswith('ENERGY') :
            category_folder = os.path.join(output_folder, "ENERGY")
        elif image.startswith('extra') :
            category_folder = os.path.join(output_folder, "extra")
        elif image.startswith('Pack_Kids') :
            category_folder = os.path.join(output_folder, "Pack_Kids")
        elif image.startswith('Pack_Ramy') :
            category_folder = os.path.join(output_folder, "Pack_Ramy")
        elif image.startswith('PET_Energie') :
            category_folder = os.path.join(output_folder, "ENERGY")
        elif image.startswith('PET_Extra') :
            category_folder = os.path.join(output_folder, "extra")
        elif image.startswith('PET_Malt') :
            category_folder = os.path.join(output_folder, "Malt")
        elif image.startswith('PET_Ramy') :
            category_folder = os.path.join(output_folder, "bottle")
        elif image.startswith('PET_WaterFruits') :
            category_folder = os.path.join(output_folder, "WaterFruits")
        elif image.startswith('UP') :
            category_folder = os.path.join(output_folder, "UP")
        elif image.startswith('ramy canette') :
            category_folder = os.path.join(output_folder, "Cannete")
        elif image.startswith('ramy LBEN') :
            category_folder = os.path.join(output_folder, "LBEN")  # Default category for unknown types

        # Create the folder if it doesn't exist
        os.makedirs(category_folder, exist_ok=True)

        # Move the image
        shutil.move(os.path.join(image_folder, image), os.path.join(category_folder, image))

print("Images sorted successfully!")
