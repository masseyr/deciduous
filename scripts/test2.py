from modules import Samples, Sublist, Handler
import numpy as np


if __name__ == '__main__':

    samp = Samples(csv_file="D:\\Shared\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\ABoVE_all_2010_sampV1.csv",
                   label_colname='Decid_AVG')

    print(samp)

    kk = samp.extract_column(column_name='system:index')
    print(kk['value'])

    samp.delete_column(column_name='system:index')
    samp.delete_column(column_name='.geo')

    print(samp)

    data_indx = list(int(val.split('_')[0]) for val in kk['value'])
    site_indx = list(int(val.split('_')[1]) for val in kk['value'])
    px_indx = list(int(val.split('_')[2]) for val in kk['value'])

    samp.add_column(column_data=site_indx, column_name='Site_index')
    samp.add_column(column_data=px_indx, column_name='Pixel_index')

    samp.save_to_file('D:\\Shared\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\test_sampV1.csv')




