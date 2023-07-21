import cv2
import os

directory_path = 'C:/Users/SowaibaArshad/PycharmProjects/PlantVillage/PlantVillage-Dataset-master/raw/color'
images = []
labels = []

# Walking into our dataset of PLANT VILLAGE
# First image path is made with subdirectory path and joined with file name
for sub, _, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.JPG'):
            img_path = os.path.join(sub, file)
            image = cv2.imread(img_path)
            label = os.path.basename(sub)
            images.append(image)
            labels.append(label)

output_directory = 'C:/Users/SowaibaArshad/PycharmProjects/PlantVillage/PlantVillage-Dataset-master/raw/preprocessed_images_final'

for i, img in enumerate(images):
    # Convert image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper color thresholds for the plant (in HSV space)
    lower_green = (40, 40, 40)
    upper_green = (70, 255, 255)

    # Create a binary mask of the plant using the color thresholds
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # Apply morphological operations to remove noise and fill in gaps in the plant
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Apply the binary mask to the original image to extract the plant
    img_segmented = cv2.bitwise_and(img, img, mask=mask_cleaned)

    # Resize the segmented image
    img_resized = cv2.resize(img_segmented, (256, 256))

    # Increase brightness
    brightness = 50
    img_brightened = cv2.add(img_resized, brightness)

    # Apply denoising, grayscale conversion, and histogram equalization
    dst = cv2.fastNlMeansDenoisingColored(img_brightened, None, 10, 10, 7, 21)
    img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.equalizeHist(img_gray)

    label = labels[i]
    filename = f'{label}_{i}.jpg'
    output_path = os.path.join(output_directory, label, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_filtered)
