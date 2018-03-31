import os
import fnmatch
import pandas as pd


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


def get_rfpickle_info(infile):
    jumble = pd.read_csv(infile)
    return {'name': jumble.iloc[4,0], 'rsq': jumble.iloc[3,0], 'rmse': jumble.iloc[2,0]}














