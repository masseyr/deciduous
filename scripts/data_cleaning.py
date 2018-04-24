from modules import *
import seaborn as sns

if __name__ == '__main__':

    infile = "C:\\Users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\rf_info_all.csv"

    names, data = Handler(infile).read_csv_as_array()
    print(names)
    print(data)