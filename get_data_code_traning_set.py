import os
import csv
import glob
from pattern import generate_2d_array

from IPython import embed 

def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['filename']
            label = row['ZLX']
            data.append({'name': name, 'label': label})
    return data

def search_files(folder, name):
    files = glob.glob(os.path.join(folder, f'*{name}*fits'))
    return files

# Path to the CSV file
csv_file_path = '/Users/xie/WORKSPACE/jet_pattern/roma_test_sample/sample_1000_1 - sample_1000_1.csv.csv'

# Folder path to search for files
folder_path = '/Users/xie/WORKSPACE/jet_pattern/sample_1000_1'

# Read the 'name' and 'label_name' columns from the CSV file
data = read_csv_file(csv_file_path)

# Dictionary to store the data
data_dict = {}

# Iterate over the data
for entry in data:
    name = entry['name']
    label = entry['label']

    # Search for files with the given name in the folder
    files = search_files(folder_path, name)

    # Iterate over the found files
    for file_path in files:
        # Call the 'func' function and get the return value 'signal'
        
        try:
            signal = generate_2d_array(file_path)
            if signal.shape==(120,120): # make sure it is 120,120 size
                signal=signal.tolist() # in this way, the result can be saved as json file
            else:
                continue
        except:
            continue

        # Extract the filename from the file path
        filename = os.path.basename(file_path)

        # Create a dictionary entry with 'name', 'signal', 'label', and 'filename'
        data_dict[filename] = {
            'name': name,
            'signal': signal,
            'label': label,
            'filename': filename
        }

# Save the dictionary to a file (e.g., JSON)
# You can choose the desired file format for saving the dictionary
# For example, using JSON:
import json

output_file_path = 'sample_1000_1.json'

with open(output_file_path, 'w') as file:
    json.dump(data_dict, file)

print("Data saved successfully!")
