import os
import pandas as pd
import shutil

# Define paths
image_folder = 'Raw Images/All Images Unclassified'
excel_file = 'Raw Images/Image Classification via MTurk.xlsx'

# Read Excel file using pandas
df = pd.read_excel(excel_file)

# Create folders for each label
label_0_folder = 'Raw Images/1.5/new_static'
label_1_folder = 'Raw Images/1.5/new_dynamic'

os.makedirs(label_0_folder, exist_ok=True)
os.makedirs(label_1_folder, exist_ok=True)

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    image_name = row['imagename']  # Replace 'ImageNameColumn' with the actual column name in your Excel file
    label = row['dyno_1_avgeabove1.5']  # Replace 'LabelColumn' with the actual column name in your Excel file

    # Construct the source and destination paths
    source_path = os.path.join(image_folder, image_name)
    
    if label == 0:
        destination_folder = label_0_folder
    elif label == 1:
        destination_folder = label_1_folder
    else:
        print(f"Invalid label {label} for image {image_name}")
        continue

    destination_path = os.path.join(destination_folder, image_name)

    # Copy the image to the respective folder
    try:
        shutil.copy(source_path, destination_path)
        print(f"Copied {image_name} to {destination_folder}")
    except FileNotFoundError:
        print(f"File {image_name} not found in {image_folder}")
    except Exception as e:
        print(f"Error copying {image_name}: {e}")
