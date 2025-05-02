# import csv
# import os
# import random
# import boto3
# import botocore
# import tqdm
# from concurrent import futures
# import pandas as pd
# import re
# import sys

# #Rec: Take 5000 from the beginning 

# # Constants
# BUCKET_NAME = 'open-images-dataset'
# REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

# ###################################
# # Step 1: #From a large dataset, find all images with at least 2 cars and write their IDs to a file -> car_images.txt
# ###################################
# def find_images_with_multiple_cars(output_file="car_images.txt",subset_size=5000, seed=1610):
#     """
#     Find all images in the Open Images training set with at least 2 cars and
#     write their IDs to a file in the format 'train/$id'.
    
#     Args:
#         output_file: Path to the output file where image IDs will be saved
#     """
#     print("Downloading Open Images annotations if needed...")
    
#     # Paths to annotation files
#     class_labels_file = "class-descriptions-boxable.csv"
#     train_annotations_file = "train-annotations-bbox.csv"
    
#     # Download the class descriptions if needed
#     if not os.path.exists(class_labels_file):
#         print(f"Downloading {class_labels_file}...")
#         os.system(f"wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv")
        
        
#     # Download the training annotations if needed
#     if not os.path.exists(train_annotations_file):
#         print(f"Downloading {train_annotations_file}...")
#         os.system(f"wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv")
    
#     # Get the class ID for 'Car'
#     car_class_id = None
#     print("Finding car class ID...")
#     with open(class_labels_file, 'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             if row[1].lower() == 'car':
#                 car_class_id = row[0]
#                 break
    
#     if not car_class_id:
#         print("Error: Could not find class ID for 'Car'")
#         return
    
#     print(f"Car class ID: {car_class_id}")
    
#     # Count car occurrences per image
#     print("Reading annotations and counting cars per image...")
#     car_counts = {}
    
#     # Read the annotations file in chunks to handle its large size
#     chunk_size = 1000000
#     for chunk in tqdm.tqdm(pd.read_csv(train_annotations_file, chunksize=chunk_size)):
#         # Filter to only car annotations
#         car_annotations = chunk[chunk['LabelName'] == car_class_id]
        
#         # Count cars per image
#         for image_id in car_annotations['ImageID']:
#             car_counts[image_id] = car_counts.get(image_id, 0) + 1
    
#     # Filter images with at least 2 cars
#     images_with_multiple_cars = [img_id for img_id, count in car_counts.items() if count >= 2]
    
#     print(f"Found {len(images_with_multiple_cars)} images with at least 2 cars")
    
#     # Randomly sample a subset with seed
#     if len(images_with_multiple_cars) > subset_size:
#         random.seed(seed)
#         images_with_multiple_cars = random.sample(images_with_multiple_cars, subset_size)
#         print(f"Randomly selected {subset_size} images with seed={seed}")

#     with open(output_file, 'w') as f:
#         for img_id in images_with_multiple_cars:
#             f.write(f"train/{img_id}\n")
    
#     print(f"Image IDs saved to {output_file}")
#     print(f"You can now use the downloader script with this file: python downloader.py {output_file}")
    
    

# ###################################
# # Step 2: From the car_images.txt file, download images to the data folder 
# ###################################
# def check_and_homogenize_one_image(image):
#   split, image_id = re.match(REGEX, image).groups()
#   yield split, image_id

# def check_and_homogenize_image_list(image_list):
#   for line_number, image in enumerate(image_list):
#     try:
#       yield from check_and_homogenize_one_image(image)
#     except (ValueError, AttributeError):
#       raise ValueError(
#           f'ERROR in line {line_number} of the image list. The following image '
#           f'string is not recognized: "{image}".')

# def read_image_list_file(image_list_file):
#   with open(image_list_file, 'r') as f:
#     for line in f:
#       yield line.strip().replace('.jpg', '')


# def download_one_image(bucket, split, image_id, download_folder):
#   try:
#     bucket.download_file(f'{split}/{image_id}.jpg',
#                          os.path.join(download_folder, f'{image_id}.jpg'))
#   except botocore.exceptions.ClientError as exception:
#     sys.exit(
#         f'ERROR when downloading image `{split}/{image_id}`: {str(exception)}')

# def download_all_images(image_list_file, download_folder="data", num_processes=14):
#     bucket = boto3.resource('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

#     if not os.path.exists(download_folder):
#         os.makedirs(download_folder)
#     try:
#         image_list = list(check_and_homogenize_image_list(read_image_list_file(image_list_file)))
#     except ValueError as exception:
#         sys.exit(exception)

#     progress_bar = tqdm.tqdm(total=len(image_list), desc='Downloading images', leave=True)
#     with futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
#         all_futures = [
#             executor.submit(download_one_image, bucket, split, image_id, download_folder)
#             for (split, image_id) in image_list
#         ]
#         for future in futures.as_completed(all_futures):
#             future.result()
#             progress_bar.update(1)
#     progress_bar.close()
    
# ###################################
# # Step 3: Given data folder, extract car labels from Open Images dataset -> export to car_labels.csv
# ###################################
# def extract_car_labels_for_images(image_list_file, output_file="car_labels.csv"):
#     """
#     Extract only car labels for images specified in image_list_file from Open Images dataset.
    
#     Args:
#         image_list_file: File containing image IDs in format 'train/[IMAGE_ID]'
#         output_file: Path to save extracted labels """
#     print("Reading image IDs...")
#     # Read image IDs from the file
#     image_ids = []
#     with open(image_list_file, 'r') as f:
#         for line in f:
#             # Extract just the ID part from "train/ID" format
#             parts = line.strip().split('/')
#             if len(parts) == 2:
#                 image_ids.append(parts[1])

#     image_ids_set = set(image_ids)
#     print(f"Found {len(image_ids_set)} unique image IDs")

#     # Check if annotation files exist
#     class_desc_file = "class-descriptions-boxable.csv"  
#     bbox_file = "train-annotations-bbox.csv" #Tao sua o day 

#     if not os.path.exists(class_desc_file):
#         print(f"Please download the class descriptions file first:")
#         print("wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv")
#         return
    
#     if not os.path.exists(bbox_file):
#         print(f"Please download the bounding box annotations file first:")
#         print("wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv")
#         return
    
#     # Load class descriptions
#     print("Loading class descriptions...")
#     class_descriptions = {}
#     car_class_ids = []
    
#     with open(class_desc_file, 'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             class_descriptions[row[0]] = row[1]
#             # Identify car-related classes
#             if any(car_term in row[1].lower() for car_term in ['car', 'truck', 'taxi', 'bus']):
#                 car_class_ids.append(row[0])

#     print(f"Found {len(car_class_ids)} car-related classes")

#     # Process annotations in chunks to handle large file size
#     print("Extracting car labels for your images...")
#     extracted_labels = []
    
#     # Process the file in chunks to save memory
#     chunk_size = 1000000
#     total_labels_found = 0

#     for chunk in tqdm.tqdm(pd.read_csv(bbox_file, chunksize=chunk_size)):
#         # Filter to only include our image IDs AND car-related classes
#         # matching_annotations = chunk[(chunk['ImageID'].isin(image_ids_set)) & 
#         #                             (chunk['LabelName'].isin(car_class_ids))]
#         matching_annotations = chunk[(chunk['ImageID'].isin(image_ids_set)) & 
#                               (chunk['LabelName'].isin(car_class_ids))].copy()

        
#         if not matching_annotations.empty:
#             total_labels_found += len(matching_annotations)
#             # Add class name based on LabelName ID
#             matching_annotations['LabelName_Text'] = matching_annotations['LabelName'].apply(
#                 lambda x: class_descriptions.get(x, 'Unknown'))
#             extracted_labels.append(matching_annotations)

#     if extracted_labels:
#         # Combine all chunks
#         all_labels = pd.concat(extracted_labels)
        
#         # Save to CSV
#         all_labels.to_csv(output_file, index=False)
        
#         print(f"Found {total_labels_found} car annotations for {len(all_labels['ImageID'].unique())} images")
#         print(f"Labels saved to {output_file}")
        
#         # Show counts by class
#         class_counts = all_labels['LabelName_Text'].value_counts()
#         print("\nTop 10 car class counts:")
#         print(class_counts.head(10))
#     else:
#         print("No matching car annotations found for your images")
    

# ###################################
# # Run all steps in order
# ###################################
# #Set num_processes according to the number of cores in your machine: Download takes 23 mins
# if __name__ == "__main__":
#     find_images_with_multiple_cars(output_file="car_images.txt",subset_size=5000, seed=1610)
#     download_all_images("car_images.txt", download_folder="data", num_processes=14)
#     extract_car_labels_for_images("car_images.txt")
    
    
#     # Load data 
#     labels = pd.read_csv("car_labels.csv") 
#     labels = labels[['ImageID','XMin', 'XMax', 'YMin','YMax', 'LabelName_Text']] 
#     labels.to_csv("car_labels.csv", index=False)
    
    
    
    

#####################################

import csv
import os
import random
import boto3
import botocore
import tqdm
from concurrent import futures
import pandas as pd
import re
import sys
from collections import defaultdict

# Constants
BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'
TARGET_CLASSES = ['car', 'truck', 'bus', 'taxi']

###################################
# Step 1: Find images with target vehicle classes and write their IDs to a file
###################################
def find_balanced_vehicle_images(output_file="vehicle_images.txt", images_per_class=1000, seed=1610):
    """
    Find images in the Open Images training set containing the target vehicle classes
    and create a balanced dataset with equal representation of each class.
    
    Args:
        output_file: Path to the output file where image IDs will be saved
        images_per_class: Number of images to collect per target class
        seed: Random seed for reproducibility
    """
    print("Downloading Open Images annotations if needed...")
    
    # Paths to annotation files
    class_labels_file = "class-descriptions-boxable.csv"
    train_annotations_file = "train-annotations-bbox.csv"
    
    # Download the class descriptions if needed
    if not os.path.exists(class_labels_file):
        print(f"Downloading {class_labels_file}...")
        os.system(f"wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv")
    
    # Download the training annotations if needed
    if not os.path.exists(train_annotations_file):
        print(f"Downloading {train_annotations_file}...")
        os.system(f"wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv")
    
    # Get class IDs for target classes
    print("Finding class IDs for target vehicle classes...")
    class_id_map = {}
    with open(class_labels_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            class_name = row[1].lower()
            for target_class in TARGET_CLASSES:
                if target_class.lower() == class_name:
                    class_id_map[row[0]] = target_class
                    print(f"Found {target_class} class ID: {row[0]}")
    
    if not class_id_map:
        print("Error: Could not find class IDs for target vehicle classes")
        return
    
    # Create a reverse mapping for easy lookup
    target_class_ids = list(class_id_map.keys())
    
    # Track images by class
    class_to_images = defaultdict(set)
    images_with_multiple_classes = defaultdict(set)
    
    # Read the annotations file in chunks to handle its large size
    print("Reading annotations and categorizing images by vehicle class...")
    chunk_size = 1000000
    for chunk in tqdm.tqdm(pd.read_csv(train_annotations_file, chunksize=chunk_size)):
        # Filter to only target class annotations
        vehicle_annotations = chunk[chunk['LabelName'].isin(target_class_ids)]
        
        # Group by image and class
        for _, row in vehicle_annotations.iterrows():
            image_id = row['ImageID']
            class_id = row['LabelName']
            class_name = class_id_map[class_id]
            class_to_images[class_name].add(image_id)
            
            # Also keep track of which classes each image has
            images_with_multiple_classes[image_id].add(class_name)
    
    # Print statistics
    for class_name, images in class_to_images.items():
        print(f"Found {len(images)} images with {class_name}")
    
    # Select a balanced set of images for each class
    selected_images = set()
    random.seed(seed)
    
    for class_name in TARGET_CLASSES:
        available_images = list(class_to_images.get(class_name, set()))
        
        if not available_images:
            print(f"Warning: No images found for class '{class_name}'")
            continue
            
        # Prioritize images that have only this class first
        exclusive_images = [img_id for img_id in available_images 
                           if len(images_with_multiple_classes[img_id]) == 1]
        
        # If we don't have enough exclusive images, use images with multiple classes
        if len(exclusive_images) >= images_per_class:
            class_selection = random.sample(exclusive_images, images_per_class)
        else:
            # Take all exclusive images first
            class_selection = exclusive_images
            
            # Then sample from images with multiple classes
            multi_class_images = [img_id for img_id in available_images 
                                 if len(images_with_multiple_classes[img_id]) > 1]
            
            remaining_needed = images_per_class - len(class_selection)
            if multi_class_images and remaining_needed > 0:
                additional = random.sample(multi_class_images, 
                                          min(remaining_needed, len(multi_class_images)))
                class_selection.extend(additional)
        
        selected_images.update(class_selection)
        print(f"Selected {len(class_selection)} images for {class_name}")
    
    print(f"Total selected images: {len(selected_images)}")
    
    # Write selected image IDs to file
    with open(output_file, 'w') as f:
        for img_id in selected_images:
            f.write(f"train/{img_id}\n")
    
    print(f"Image IDs saved to {output_file}")
    return selected_images

###################################
# Step 2: From the vehicle_images.txt file, download images to the data folder 
###################################
def check_and_homogenize_one_image(image):
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id

def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f'ERROR in line {line_number} of the image list. The following image '
                f'string is not recognized: "{image}".')

def read_image_list_file(image_list_file):
    with open(image_list_file, 'r') as f:
        for line in f:
            yield line.strip().replace('.jpg', '')

def download_one_image(bucket, split, image_id, download_folder):
    try:
        bucket.download_file(f'{split}/{image_id}.jpg',
                            os.path.join(download_folder, f'{image_id}.jpg'))
    except botocore.exceptions.ClientError as exception:
        print(f'ERROR when downloading image `{split}/{image_id}`: {str(exception)}')
        return False
    return True

def download_all_images(image_list_file, download_folder="data", num_processes=10):
    bucket = boto3.resource('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    try:
        image_list = list(check_and_homogenize_image_list(read_image_list_file(image_list_file)))
    except ValueError as exception:
        sys.exit(exception)

    progress_bar = tqdm.tqdm(total=len(image_list), desc='Downloading images', leave=True)
    successful_downloads = []
    
    with futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        all_futures = {
            executor.submit(download_one_image, bucket, split, image_id, download_folder): (split, image_id)
            for (split, image_id) in image_list
        }
        for future in futures.as_completed(all_futures):
            result = future.result()
            split, image_id = all_futures[future]
            if result:
                successful_downloads.append(f"{split}/{image_id}")
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Write successful downloads to a new file
    success_file = image_list_file.replace(".txt", "_successful.txt")
    with open(success_file, 'w') as f:
        for image_path in successful_downloads:
            f.write(f"{image_path}\n")
    
    print(f"Successfully downloaded {len(successful_downloads)} of {len(image_list)} images")
    print(f"Successful downloads list saved to {success_file}")
    return success_file
    
###################################
# Step 3: Extract vehicle labels from Open Images dataset -> export to vehicle_labels.csv
###################################
def extract_vehicle_labels_for_images(image_list_file, output_file="vehicle_labels.csv"):
    """
    Extract labels for target vehicle classes from images specified in image_list_file.
    
    Args:
        image_list_file: File containing image IDs in format 'train/[IMAGE_ID]'
        output_file: Path to save extracted labels 
    """
    print("Reading image IDs...")
    # Read image IDs from the file
    image_ids = []
    with open(image_list_file, 'r') as f:
        for line in f:
            # Extract just the ID part from "train/ID" format
            parts = line.strip().split('/')
            if len(parts) == 2:
                image_ids.append(parts[1])

    image_ids_set = set(image_ids)
    print(f"Found {len(image_ids_set)} unique image IDs")

    # Check if annotation files exist
    class_desc_file = "class-descriptions-boxable.csv"  
    bbox_file = "train-annotations-bbox.csv"

    if not os.path.exists(class_desc_file):
        print(f"Please download the class descriptions file first:")
        print("wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv")
        return
    
    if not os.path.exists(bbox_file):
        print(f"Please download the bounding box annotations file first:")
        print("wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv")
        return
    
    # Load class descriptions
    print("Loading class descriptions...")
    class_descriptions = {}
    vehicle_class_ids = []
    
    with open(class_desc_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            class_descriptions[row[0]] = row[1]
            # Filter for target vehicle classes
            if row[1] in TARGET_CLASSES or row[1].lower() in [c.lower() for c in TARGET_CLASSES]:
                vehicle_class_ids.append(row[0])

    print(f"Found {len(vehicle_class_ids)} target vehicle class IDs")

    # Process annotations in chunks to handle large file size
    print("Extracting vehicle labels for your images...")
    extracted_labels = []
    
    # Process the file in chunks to save memory
    chunk_size = 1000000
    total_labels_found = 0

    for chunk in tqdm.tqdm(pd.read_csv(bbox_file, chunksize=chunk_size)):
        # Filter to only include our image IDs AND target vehicle classes
        matching_annotations = chunk[(chunk['ImageID'].isin(image_ids_set)) & 
                                     (chunk['LabelName'].isin(vehicle_class_ids))].copy()
        
        if not matching_annotations.empty:
            total_labels_found += len(matching_annotations)
            # Add class name based on LabelName ID
            matching_annotations['LabelName_Text'] = matching_annotations['LabelName'].apply(
                lambda x: class_descriptions.get(x, 'Unknown'))
            extracted_labels.append(matching_annotations)

    if extracted_labels:
        # Combine all chunks
        all_labels = pd.concat(extracted_labels)
        
        # Check class balance
        class_counts = all_labels['LabelName_Text'].value_counts()
        print("\nClass distribution in the dataset:")
        print(class_counts)
        
        # Calculate class weights for potential use in model training
        total_samples = len(all_labels)
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
        
        print("\nClass weights (for balanced training):")
        for class_name, weight in class_weights.items():
            print(f"{class_name}: {weight:.4f}")
        
        # Save to CSV
        all_labels.to_csv(output_file, index=False)
        
        print(f"\nFound {total_labels_found} vehicle annotations for {len(all_labels['ImageID'].unique())} images")
        print(f"Labels saved to {output_file}")
        
        # Return class counts for potential further processing
        return class_counts
    else:
        print("No matching vehicle annotations found for your images")
        return None

###################################
# Run all steps in order
###################################
if __name__ == "__main__":
    # Step 1: Find balanced set of images containing target classes
    selected_images = find_balanced_vehicle_images(output_file="vehicle_images.txt", images_per_class=1000, seed=1610)
    
    # Step 2: Download the selected images
    successful_downloads_file = download_all_images("vehicle_images.txt", download_folder="data", num_processes=10)
    
    # Step 3: Extract labels for the downloaded images
    class_counts = extract_vehicle_labels_for_images(successful_downloads_file, output_file="vehicle_labels.csv")
    
    # Load and prepare final data
    if os.path.exists("vehicle_labels.csv"):
        labels = pd.read_csv("vehicle_labels.csv") 
        labels = labels[['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'LabelName_Text']] 
        labels.to_csv("vehicle_labels.csv", index=False)
        
        # Save basic dataset statistics
        stats = {
            "total_images": len(labels['ImageID'].unique()),
            "total_annotations": len(labels),
            "class_distribution": labels['LabelName_Text'].value_counts().to_dict()
        }
        
        pd.DataFrame([stats]).to_json("dataset_stats.json", orient="records")
        print("Final dataset statistics saved to dataset_stats.json")

#Problem: The whole process takes a lot of time, especially the downloading part.