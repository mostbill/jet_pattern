from astropy.io import ascii
from astropy.table import Table
from IPython import embed

def read_csv_to_astropy_table(csv_file):
    # Read the CSV file into an Astropy table
    data_table = Table.read(csv_file, format='csv')

    data_table['ra'] = ['%.5f' % ra for ra in data_table['ra'].astype(float)]
    data_table['dec'] = ['%.5f' % dec for dec in data_table['dec'].astype(float)]

    return data_table

def write_into_latex(data):
    #data = {'name': ['bike', 'car'], 'mass': [75,1200], 'speed': [10, 130]}
    ascii.write(data, Writer=ascii.Latex,
                    latexdict = {'units': {'mass': 'kg', 'speed': 'km/h'}})

if __name__ == "__main__":
    # Replace 'input.csv' with the path to your CSV file
    data_table = read_csv_to_astropy_table('/Users/xie/WORKSPACE/jet_pattern/output/kdebllacs_sample.csv')
    write_into_latex(data_table)                   
    # print(data_table)
    
    
    
# data = {'name': ['bike', 'car'], 'mass': [75,1200], 'speed': [10, 130]}
# ascii.write(data, Writer=ascii.Latex,
#                  latexdict = {'units': {'mass': 'kg', 'speed': 'km/h'}})