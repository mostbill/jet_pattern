#!/bin/bash

# Specify the folder path
folder="/Volumes/Elements/vlass/"

# Loop through numbers 1, 2, and 3
for i in 1 2 3; do
    # Get the unique source names for each number
    sources=$(find "$folder" -name "*EP0$i*" | awk -F'_' '{print $1}' | sort -u)

    # Count the unique source names
    count=$(echo "$sources" | wc -l)

    # Display the count of unique source names for each number
    echo "Number $i: Count = $count"
done
