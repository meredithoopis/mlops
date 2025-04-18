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



# Constants
BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

###################################
# Step 1: #From a large dataset, find all images with at least 2 cars and write their IDs to a file -> car_images.txt
###################################
def find_images_with_multiple_cars(output_file="car_images.txt",subset_size=5000, seed=1610):
    """
    Find all images in the Open Images training set with at least 2 cars and
    write their IDs to a file in the format 'train/$id'.
    
    Args:
        output_file: Path to the output file where image IDs will be saved
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
    
    # Get the class ID for 'Car'
    car_class_id = None
    print("Finding car class ID...")
    with open(class_labels_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1].lower() == 'car':
                car_class_id = row[0]
                break
    
    if not car_class_id:
        print("Error: Could not find class ID for 'Car'")
        return
    
    print(f"Car class ID: {car_class_id}")
    
    # Count car occurrences per image
    print("Reading annotations and counting cars per image...")
    car_counts = {}
    
    # Read the annotations file in chunks to handle its large size
    chunk_size = 1000000
    for chunk in tqdm.tqdm(pd.read_csv(train_annotations_file, chunksize=chunk_size)):
        # Filter to only car annotations
        car_annotations = chunk[chunk['LabelName'] == car_class_id]
        
        # Count cars per image
        for image_id in car_annotations['ImageID']:
            car_counts[image_id] = car_counts.get(image_id, 0) + 1
    
    # Filter images with at least 2 cars
    images_with_multiple_cars = [img_id for img_id, count in car_counts.items() if count >= 2]
    
    print(f"Found {len(images_with_multiple_cars)} images with at least 2 cars")
    
    # Randomly sample a subset with seed
    if len(images_with_multiple_cars) > subset_size:
        random.seed(seed)
        images_with_multiple_cars = random.sample(images_with_multiple_cars, subset_size)
        print(f"Randomly selected {subset_size} images with seed={seed}")

    with open(output_file, 'w') as f:
        for img_id in images_with_multiple_cars:
            f.write(f"train/{img_id}\n")
    
    print(f"Image IDs saved to {output_file}")
    print(f"You can now use the downloader script with this file: python downloader.py {output_file}")
    
    

###################################
# Step 2: From the car_images.txt file, download images to the data folder 
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
    sys.exit(
        f'ERROR when downloading image `{split}/{image_id}`: {str(exception)}')

def download_all_images(image_list_file, download_folder="data", num_processes=14):
    bucket = boto3.resource('s3', config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    try:
        image_list = list(check_and_homogenize_image_list(read_image_list_file(image_list_file)))
    except ValueError as exception:
        sys.exit(exception)

    progress_bar = tqdm.tqdm(total=len(image_list), desc='Downloading images', leave=True)
    with futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        all_futures = [
            executor.submit(download_one_image, bucket, split, image_id, download_folder)
            for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()
    
###################################
# Step 3: Given data folder, extract car labels from Open Images dataset -> export to car_labels.csv
###################################
def extract_car_labels_for_images(image_list_file, output_file="car_labels.csv"):
    """
    Extract only car labels for images specified in image_list_file from Open Images dataset.
    
    Args:
        image_list_file: File containing image IDs in format 'train/[IMAGE_ID]'
        output_file: Path to save extracted labels """
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
    bbox_file = "train-annotations-bbox.csv" #Tao sua o day 

    if not os.path.exists(class_desc_file):
        print(f"Please download the class descriptions file first:")
        print("wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv")
        return
    
    if not os.path.exists(bbox_file):
        print(f"Please download the bounding box annotations file first:")
        print("wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv")
        return
    
    # Load class descriptions
    print("Loading class descriptions...")
    class_descriptions = {}
    car_class_ids = []
    
    with open(class_desc_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            class_descriptions[row[0]] = row[1]
            # Identify car-related classes
            if any(car_term in row[1].lower() for car_term in ['car', 'truck', 'taxi', 'bus']):
                car_class_ids.append(row[0])

    print(f"Found {len(car_class_ids)} car-related classes")

    # Process annotations in chunks to handle large file size
    print("Extracting car labels for your images...")
    extracted_labels = []
    
    # Process the file in chunks to save memory
    chunk_size = 1000000
    total_labels_found = 0

    for chunk in tqdm.tqdm(pd.read_csv(bbox_file, chunksize=chunk_size)):
        # Filter to only include our image IDs AND car-related classes
        # matching_annotations = chunk[(chunk['ImageID'].isin(image_ids_set)) & 
        #                             (chunk['LabelName'].isin(car_class_ids))]
        matching_annotations = chunk[(chunk['ImageID'].isin(image_ids_set)) & 
                              (chunk['LabelName'].isin(car_class_ids))].copy()

        
        if not matching_annotations.empty:
            total_labels_found += len(matching_annotations)
            # Add class name based on LabelName ID
            matching_annotations['LabelName_Text'] = matching_annotations['LabelName'].apply(
                lambda x: class_descriptions.get(x, 'Unknown'))
            extracted_labels.append(matching_annotations)

    if extracted_labels:
        # Combine all chunks
        all_labels = pd.concat(extracted_labels)
        
        # Save to CSV
        all_labels.to_csv(output_file, index=False)
        
        print(f"Found {total_labels_found} car annotations for {len(all_labels['ImageID'].unique())} images")
        print(f"Labels saved to {output_file}")
        
        # Show counts by class
        class_counts = all_labels['LabelName_Text'].value_counts()
        print("\nTop 10 car class counts:")
        print(class_counts.head(10))
    else:
        print("No matching car annotations found for your images")
    

###################################
# Run all steps in order
###################################
#Set num_processes according to the number of cores in your machine: Download takes 23 mins
if __name__ == "__main__":
    #find_images_with_multiple_cars(output_file="car_images.txt",subset_size=5000, seed=1610)
    #download_all_images("car_images.txt", download_folder="data", num_processes=14)
    extract_car_labels_for_images("car_images.txt")
    
    
    # Load data 
    labels = pd.read_csv("car_labels.csv") 
    labels = labels[['ImageID','XMin', 'XMax', 'YMin','YMax', 'LabelName_Text']] 
    labels.to_csv("car_labels.csv", index=False)