## Workflow

### Step 1: Image ID collection

- filter_car.py: Filters the Open Images dataset to obtain image IDs containing at least 2 cars
- Output: car_images.txt - a list of filtered image IDs

### Step 2: Image download

- downloader.py: Downloads images using IDs from car_images.txt
- Output: Images saved to the data folder (large dataset available on OneDrive)

### Step 3: Annotation collection

- annotations.py: Retrieves image annotations for IDs in car_images.txt
- Output: car_labels.csv - contains bounding box annotations for car objects

## Additional files

- car_images_1.txt: Extended dataset containing >40k image IDs
- class-descriptions-boxable.csv: Maps object IDs to object names (format: $object-id,$object-name)
