#!/bin/bash

# Function to find and copy files based on the provided string
find_and_copy_files() {
    local source_folder="$1"
    local dest_folder="$2"
    local string="$3"

    # Use find to locate files in the source folder matching the string and ending with .fits
    find "$source_folder" -type f -name "*$string*fits" -exec cp {} "$dest_folder/" \;
}

# Input CSV file
csv_file="./csv/mojave_single_epoch_sources.csv"

# Source folder containing the .fits files
source_folder="./mojave/"

# Destination folder to copy the files
dest_folder="./mojave_all_single_epoch_sources/"

# Ensure the destination folder exists
mkdir -p "$dest_folder"

# Find the column index of 'filename' in the CSV header
filename_column=$(awk -F ',' 'NR==1 {for(i=1;i<=NF;i++) if($i=="filename") print i}' "$csv_file")

# Check if 'filename' column was found
if [ -z "$filename_column" ]; then
    echo "Error: 'filename' column not found in the CSV file."
    exit 1
fi

# Loop through each line in the CSV file and extract the string from the 'filename' column
awk -F ',' -v col="$filename_column" 'NR>1 {print $col}' "$csv_file" | while IFS= read -r string; do
    echo "Processing string: $string"
    
    # Call the function to find and copy files based on the string
    find_and_copy_files "$source_folder" "$dest_folder" "$string"
done

echo "Files copied successfully."
