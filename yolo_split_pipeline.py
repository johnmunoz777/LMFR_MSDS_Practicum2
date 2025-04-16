
import os
import cv2
import glob
import random
import shutil
import matplotlib.pyplot as plt
def detect_face_and_generate_yolo_format(image_path, output_txt_path, class_id=0):
    """
    Detects the first face in the image and writes a YOLO-format annotation to output_txt_path.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    (all coordinates normalized to [0,1])
    """
    # Load Haar Cascade for face detection.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the image.
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print(f"No face detected in: {image_path}")
        return False
    # Use the first detected face.
    x, y, w, h = faces[0]
    # Get image dimensions.
    image_height, image_width, _ = img.shape
    # Calculate normalized center coordinates and box dimensions.
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    norm_width = w / image_width
    norm_height = h / image_height
    # Create YOLO-format string.
    yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
    # Write the annotation to the .txt file.
    with open(output_txt_path, 'w') as file:
        file.write(yolo_format)
    print(f"Annotation saved to {output_txt_path}")
    return True

def process_single_category_folder(main_folder, output_base, split_ratios={"train": 0.7, "valid": 0.2, "test": 0.1}, class_id=0):
    """
    Processes a single category folder (e.g., "john") containing images.

    Splits the images into train/valid/test sets according to split_ratios,
    generates YOLO-format .txt annotation files for each image by detecting a face,
    and copies the images and annotation files into the corresponding output folders.

    The output folder structure will be:
      output_base/
         train/
            images/
            labels/
         valid/
            images/
            labels/
         test/
            images/
            labels/
    The annotation file for each image will have the same base filename as the image,
    with a .txt extension.
    """
    # List all .jpg images in the main folder.
    image_paths = glob.glob(os.path.join(main_folder, "*.jpg"))
    random.shuffle(image_paths)
    n = len(image_paths)
    n_train = int(n * split_ratios["train"])
    n_valid = int(n * split_ratios["valid"])
    n_test = n - n_train - n_valid  # Remaining images go to test.
    print(f"Processing folder '{main_folder}': Total images: {n} | Train: {n_train}, Valid: {n_valid}, Test: {n_test}")
    # Create the output folder structure.
    for split in split_ratios.keys():
        os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)

    # Split the images.
    train_images = image_paths[:n_train]
    valid_images = image_paths[n_train:n_train + n_valid]
    test_images = image_paths[n_train + n_valid:]

    # Function to process each list of images.
    def process_image_list(image_list, split):
        for image_path in image_list:
            filename = os.path.basename(image_path)
            # Destination path for the image.
            out_image_path = os.path.join(output_base, split, "images", filename)
            # The .txt file will have the same base name as the image.
            name_without_ext = os.path.splitext(filename)[0]
            out_txt_path = os.path.join(output_base, split, "labels", name_without_ext + ".txt")
            # Generate YOLO annotation.
            success = detect_face_and_generate_yolo_format(image_path, out_txt_path, class_id=class_id)
            if success:
                # Copy the image to the destination folder.
                shutil.copy(image_path, out_image_path)
            else:
                print(f"Skipping {image_path} due to face detection failure.")

    # Process each split.
    process_image_list(train_images, "train")
    process_image_list(valid_images, "valid")
    process_image_list(test_images, "test")

# Example usage:
# Pass in a single main folder (e.g., "john") and the desired output folder.
#process_single_category_folder("john", "my_data_split", {"train": 0.7, "valid": 0.2, "test": 0.1}, class_id=0)