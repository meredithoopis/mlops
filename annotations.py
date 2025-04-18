# import pandas as pd
# import csv
# import os
# from tqdm import tqdm

# def extract_labels_for_images(image_list_file, output_file="car_labels.csv"):
#     """
#     Extract labels for images specified in image_list_file from Open Images dataset.
    
#     Args:
#         image_list_file: File containing image IDs in format 'train/[IMAGE_ID]'
#         output_file: Path to save extracted labels
#     """
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
#     bbox_file = "oidv6-train-annotations-bbox.csv"
    
#     if not os.path.exists(class_desc_file):
#         print(f"Please download the class descriptions file first:")
#         print("wget https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions-boxable.csv")
#         return
    
#     if not os.path.exists(bbox_file):
#         print(f"Please download the bounding box annotations file first:")
#         print("wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv")
#         return
    
#     # Load class descriptions
#     print("Loading class descriptions...")
#     class_descriptions = {}
#     with open(class_desc_file, 'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             class_descriptions[row[0]] = row[1]
    
#     # Process annotations in chunks to handle large file size
#     print("Extracting labels for your images...")
#     extracted_labels = []
    
#     # Process the file in chunks to save memory
#     chunk_size = 1000000
#     total_labels_found = 0
    
#     for chunk in tqdm(pd.read_csv(bbox_file, chunksize=chunk_size)):
#         # Filter to only include our image IDs
#         matching_annotations = chunk[chunk['ImageID'].isin(image_ids_set)]
        
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
#         print(f"Found {total_labels_found} annotations for {len(all_labels['ImageID'].unique())} images")
#         print(f"Labels saved to {output_file}")
        
#         # Show counts by class
#         class_counts = all_labels['LabelName_Text'].value_counts()
#         print("\nTop 10 class counts:")
#         print(class_counts.head(10))
#     else:
#         print("No matching annotations found for your images")

# if __name__ == "__main__":
#     extract_labels_for_images("car_images.txt")



import pandas as pd
import csv
import os
from tqdm import tqdm

def extract_car_labels_for_images(image_list_file, output_file="car_labels.csv"):
    """
    Extract only car labels for images specified in image_list_file from Open Images dataset.
    
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
    bbox_file = "oidv6-train-annotations-bbox.csv"
    
    if not os.path.exists(class_desc_file):
        print(f"Please download the class descriptions file first:")
        print("wget https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions-boxable.csv")
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
    
    for chunk in tqdm(pd.read_csv(bbox_file, chunksize=chunk_size)):
        # Filter to only include our image IDs AND car-related classes
        matching_annotations = chunk[(chunk['ImageID'].isin(image_ids_set)) & 
                                    (chunk['LabelName'].isin(car_class_ids))]
        
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

if __name__ == "__main__":
    extract_car_labels_for_images("car_images.txt")