#!/bin/bash

echo "Please enter the directory name:"
read dir_name

classes=("VISUAL NEEDED")

for class in "${classes[@]}"
do
    # Python script to get the names from the csv file and write them to a file called names.txt
    python3 << END
import pandas as pd

df = pd.read_csv('/Users/xie/WORKSPACE/jet_pattern/output/${dir_name}.csv')  # Replace 'file.csv' with your csv file name
selected_rows = df[df['final_class'] == '$class']
selected_names = selected_rows['filename']

with open('names.txt', 'w') as f:
    for name in selected_names:
        f.write(name + '\n')
END

    # Bash script to copy the files
    mkdir -p "$class"

    while IFS= read -r name; do
        find "$dir_name" -name "$name*.pdf" -print0 | xargs -0 -I{} cp {} "$class"/
    done < names.txt

    # Cleaning up
    rm names.txt
done
