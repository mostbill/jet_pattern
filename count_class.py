import csv
from IPython import embed

def calculate_percentage(csv_file, column_name):
    # Create a dictionary to store the counts of each string
    string_counts = {}
    total_count = 0

    # Open the CSV file in read mode
    with open(csv_file, 'r') as file:
        # Create a CSV reader object
        reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Get the value of the 'final_class' column
            value = row[column_name]
            
            # Update the count for the current string
            string_counts[value] = string_counts.get(value, 0) + 1
            total_count += 1

    # Calculate the percentage for each string
    percentages = {}
    counts = {}
    for string, count in string_counts.items():
        percentage = (count / total_count) * 100
        percentages[string] = percentage
        counts[string]= count

    return counts, percentages

# Prompt the user to enter the CSV file path
csv_file = input("Enter the path to the CSV file: ")

# Prompt the user to enter the column name
column_name = input("Enter the column name: ")

counts, percentages = calculate_percentage(csv_file, column_name)

# Print the counts and percentages
for (string, count), (string, percentage) in zip(counts.items(), percentages.items()):
    print(f"{string}: {count:.0f} ({percentage:.2f}%)")
