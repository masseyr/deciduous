import os
import fnmatch
import pandas as pd

"""
This module houses functions specific to the data used
"""



# dictionary for a use case
bname_dict = {
    'NIR': 'NIR_Period1',
    'SWIR1': 'SWIR1_Period1',
    'NDVI': 'NDVI_Period1',
    'NDWI': 'NDWI_Period1',
    'NBR': 'NBR_Period1',
    'RATIO57': 'RATIO57_Period1',
    'ND57': 'ND57_Period1',
    'NMDI': 'NMDI_Period1',
    'NIR_1': 'NIR_Period2',
    'SWIR1_1': 'SWIR1_Period2',
    'NDVI_1': 'NDVI_Period2',
    'NDWI_1': 'NDWI_Period2',
    'NBR_1': 'NBR_Period2',
    'RATIO57_1': 'RATIO57_Period2',
    'ND57_1': 'ND57_Period2',
    'NMDI_1': 'NMDI_Period2',
    'NIR_2': 'NIR_Period3',
    'SWIR1_2': 'SWIR1_Period3',
    'NDVI_2': 'NDVI_Period3',
    'NDWI_2': 'NDWI_Period3',
    'NBR_2': 'NBR_Period3',
    'RATIO57_2': 'RATIO57_Period3',
    'ND57_2': 'ND57_Period3',
    'NMDI_2': 'NMDI_Period3',
}


# Tree cover files ftp server name
TCserver = 'ftp.glcf.umd.edu'


# Tree cover files ftp file path
def get_TCdata_filepath(path, row, year):
    return {
               'filestr': "/glcf/LandsatTreecover/WRS2/p{p}/r{r}/p{p}r{r}_TC_2010/p{p}r{r}_TC_{y}.tif.gz".format(
                p=str(path).zfill(3),
                r=str(row).zfill(3),
                y=str(year).zfill(3)),
               'errfilestr': "/glcf/LandsatTreecover/WRS2/p{p}/r{r}/p{p}r{r}_TC_2010/p{p}r{r}_TC_{y}_err.tif.gz".format(
                p=str(path).zfill(3),
                r=str(row).zfill(3),
                y=str(year).zfill(3))
            }


def get_rfpickle_info(infile):
    """
    Read statistics of random forest performance from csv file
    :param infile:
    :return:
    """
    jumble = pd.read_csv(infile)
    return {'name': jumble.iloc[4, 0], 'rsq': jumble.iloc[3,0], 'rmse': jumble.iloc[2,0]}


def read_y_param_from_summary(csv_file):
    """
    read y parameter output from csv file
    :param csv_file:
    :return:
    """
    # read lines
    with open(csv_file, 'r') as fileptr:
            lines = fileptr.readlines()

    # y observed
    y = [float(elem.strip()) for elem in lines[0].split(',')[1:]]

    # y mean
    mean_y = [float(elem.strip()) for elem in lines[1].split(',')[1:]]

    # variance in y
    var_y = [float(elem.strip()) for elem in lines[2].split(',')[1:]]

    # rms error
    rmse = float(lines[3].split(',')[1].strip())

    # r squared
    r_sq = float(lines[4].split(',')[1].strip())

    # random forest file
    rf_file = lines[5].split(',')[1].strip()

    return {
            'obs_y': y,
            'mean_y': mean_y,
            'var_y': var_y,
            'rmse': rmse,
            'r_sq': r_sq,
            'rf_file': rf_file
            }
