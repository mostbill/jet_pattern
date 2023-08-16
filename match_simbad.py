import csv
from astroquery.simbad import Simbad
from astropy import units as u
from astropy.coordinates import SkyCoord

from IPython import embed

def search_simbad(ra_list, dec_list):
    customSimbad = Simbad()
    customSimbad.add_votable_fields('ra', 'dec', 'otype')

    matches = []

    for ra, dec in zip(ra_list, dec_list):
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
        result_table = customSimbad.query_region(coord, radius=10 * u.arcsec)
        
        embed()

        if result_table is not None:
            # Filter the matches based on the size criteria
            filtered_matches = result_table[result_table['Radius'] > 5 * u.arcsec]
            matches.append(filtered_matches)

    return matches

def extract_coordinates(csv_file):
    ras=[]
    decs=[]
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['final_class'] == 'COMPACT_offset':
                ra = row['ra']
                dec = row['dec']
                ras.append(ra)
                decs.append(dec)
    return ras, decs

ra_list, dec_list=extract_coordinates('/Users/xie/WORKSPACE/jet_pattern/output/roma_bzcat.csv')
results = search_simbad(ra_list, dec_list)

for result in results:
    print(result)
