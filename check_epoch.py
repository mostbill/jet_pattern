import os
from astropy.io import fits
import csv

def process_fits_files(folder_path):
    result_data = []

    # Iterate through all FITS files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".fits"):
            fits_path = os.path.join(folder_path, filename)

            try:
                # Read the FITS header and extract the 'DATE-OBS' entry
                with fits.open(fits_path) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS', None)

                    # Extract the year from the 'DATE-OBS' string
                    if date_obs and len(date_obs) >= 4:
                        year = float(date_obs[:4])

                        # Determine the value based on the year
                        value = "EP01" if year <= 2019 else "EP02"

                        # Append the result to the list
                        result_data.append((filename, value))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Save the result to a CSV file
    csv_file = "./single_epoch_statistics/roma_result.csv"
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Value'])

        for filename, value in result_data:
            csv_writer.writerow([filename[:20], value])

    print(f"Results saved to {csv_file}")

process_fits_files('/Users/xie/WORKSPACE/jet_pattern/single_epoch_statistics/roma_bzcat_all_single_epoch_sources/')
