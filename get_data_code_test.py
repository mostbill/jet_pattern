import os
import csv
import glob
from pattern import generate_2d_array

from tqdm import tqdm

from IPython import embed 

# def read_csv_file(filename):
#     data = []
#     with open(filename, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             name = row['filename']
#             label = row['ZLX']
#             data.append({'name': name, 'label': label})
#     return data

def search_files(folder, name):
    files = glob.glob(os.path.join(folder, f'*{name}*fits'))
    return files

def get_file_names(folder_path):
    file_names = []
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.fits'):
            name = file.split('_')[0]
            if name!='.': # weird problem encountered, no idea why some file will start like '._XXX.fits'
                file_names.append(file)
                names.append(name)
    return names, file_names

# Folder path to search for files
folder_path = '/Users/xie/WORKSPACE/jet_pattern/smss_sample/'

names, filenames = get_file_names(folder_path)

# Dictionary to store the data
data_dict = {}

# Iterate over the data
for i in tqdm(range(len(names))):
    name = names[i]
    filename = filenames[i]

    # Search for files with the given name in the folder
    files = search_files(folder_path, name)
    
    # if name == 'LAMOSTJ033243.77+385859.5':
    #     embed()

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
        
        # Create a dictionary entry with 'name', 'signal', 'label', and 'filename'
        data_dict[filename] = {
            'name': name,
            'signal': signal,
            'filename': filename
        }

# Save the dictionary to a file (e.g., JSON)
# You can choose the desired file format for saving the dictionary
# For example, using JSON:
import json

output_file_path = 'smss_sample.json'

with open(output_file_path, 'w') as file:
    json.dump(data_dict, file)

print("Data saved successfully!")
