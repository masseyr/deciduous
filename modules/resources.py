import os
import fnmatch
import json
import numpy as np
from shapely.geometry import Polygon, mapping
from osgeo import ogr, gdal
import sys

"""
This module houses functions specific to the data used
"""

__all__ = ['bname_dict',
           'get_TCdata_filepath',
           'TCserver',
           'read_y_param_from_summary',
           'find_path_row']

# dictionary for a use case
bname_dict = {

    'ls57': {
        'BLUE': 'BLUE_Period1',
        'GREEN': 'GREEN_Period1',
        'RED': 'RED_Period1',
        'NIR': 'NIR_Period1',
        'SWIR1': 'SWIR1_Period1',
        'SWIR2': 'SWIR2_Period1',
        'NDVI': 'NDVI_Period1',
        'NDWI': 'NDWI_Period1',
        'NBR': 'NBR_Period1',
        'RATIO57': 'RATIO57_Period1',
        'ND57': 'ND57_Period1',
        'NMDI': 'NMDI_Period1',
        'EVI': 'EVI_Period1',
        'BLUE_1': 'BLUE_Period2',
        'GREEN_1': 'GREEN_Period2',
        'RED_1': 'RED_Period2',
        'NIR_1': 'NIR_Period2',
        'SWIR1_1': 'SWIR1_Period2',
        'SWIR2_1': 'SWIR2_Period2',
        'NDVI_1': 'NDVI_Period2',
        'NDWI_1': 'NDWI_Period2',
        'NBR_1': 'NBR_Period2',
        'RATIO57_1': 'RATIO57_Period2',
        'ND57_1': 'ND57_Period2',
        'NMDI_1': 'NMDI_Period2',
        'EVI_1': 'EVI_Period2',
        'BLUE_2': 'BLUE_Period3',
        'GREEN_2': 'GREEN_Period3',
        'RED_2': 'RED_Period3',
        'NIR_2': 'NIR_Period3',
        'SWIR1_2': 'SWIR1_Period3',
        'SWIR2_2': 'SWIR2_Period3',
        'NDVI_2': 'NDVI_Period3',
        'NDWI_2': 'NDWI_Period3',
        'NBR_2': 'NBR_Period3',
        'RATIO57_2': 'RATIO57_Period3',
        'ND57_2': 'ND57_Period3',
        'NMDI_2': 'NMDI_Period3',
        'EVI_2': 'EVI_Period3',
        'SLOPE': 'SLOPE',
        'ELEVATION': 'ELEVATION'},

    'ls8': {
        'BLUE': 'BLUE_Period1',
        'GREEN': 'GREEN_Period1',
        'RED': 'RED_Period1',
        'NIR': 'NIR_Period1',
        'SWIR1': 'SWIR1_Period1',
        'SWIR2': 'SWIR2_Period1',
        'NDVI': 'NDVI_Period1',
        'NDWI': 'NDWI_Period1',
        'NBR': 'NBR_Period1',
        'RATIO57': 'RATIO57_Period1',
        'ND57': 'ND57_Period1',
        'NMDI': 'NMDI_Period1',
        'EVI': 'EVI_Period1',
        'BLUE_1': 'BLUE_Period2',
        'GREEN_1': 'GREEN_Period2',
        'RED_1': 'RED_Period2',
        'NIR_1': 'NIR_Period2',
        'SWIR1_1': 'SWIR1_Period2',
        'SWIR2_1': 'SWIR2_Period2',
        'NDVI_1': 'NDVI_Period2',
        'NDWI_1': 'NDWI_Period2',
        'NBR_1': 'NBR_Period2',
        'RATIO57_1': 'RATIO57_Period2',
        'ND57_1': 'ND57_Period2',
        'NMDI_1': 'NMDI_Period2',
        'EVI_1': 'EVI_Period2',
        'BLUE_2': 'BLUE_Period3',
        'GREEN_2': 'GREEN_Period3',
        'RED_2': 'RED_Period3',
        'NIR_2': 'NIR_Period3',
        'SWIR1_2': 'SWIR1_Period3',
        'SWIR2_2': 'SWIR2_Period3',
        'NDVI_2': 'NDVI_Period3',
        'NDWI_2': 'NDWI_Period3',
        'NBR_2': 'NBR_Period3',
        'RATIO57_2': 'RATIO57_Period3',
        'ND57_2': 'ND57_Period3',
        'NMDI_2': 'NMDI_Period3',
        'EVI_2': 'EVI_Period3',
        'SLOPE': 'SLOPE',
        'ELEVATION': 'ELEVATION'},

}

# Tree cover files ftp server name
TCserver = 'ftp.glcf.umd.edu'


# Tree cover files ftp file path
def get_TCdata_filepath(path, row, year):
    """
    Tree cover files ftp file path dictionary
    :param path: WRS2 path
    :param row: WRS2 row
    :param year: Year
    :return: Dictionary
    """
    return {
        'filestr': "/glcf/LandsatTreecover/WRS2/p{p}/r{r}/p{p}r{r}_TC_{y}/p{p}r{r}_TC_{y}.tif.gz".format(
            p=str(path).zfill(3),
            r=str(row).zfill(3),
            y=str(year).zfill(4)),
        'errfilestr': "/glcf/LandsatTreecover/WRS2/p{p}/r{r}/p{p}r{r}_TC_{y}/p{p}r{r}_TC_{y}_err.tif.gz".format(
            p=str(path).zfill(3),
            r=str(row).zfill(3),
            y=str(year).zfill(4))
    }


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


def find_path_row(polygonfile, wrsfile):
    """
    module to identify path row of intersecting landsat footprints
    :param polygonfile: Shapefile with only one polygon
    :param wrsfile: WRS2 - descending shapefile
    :return: list of path and row tuples
    """

    fieldshpfile = ogr.Open(polygonfile)
    fieldshape = fieldshpfile.GetLayer(0)
    fieldfeat = fieldshape.GetFeature(0)
    fieldgeom = fieldfeat.GetGeometryRef()

    wrsshpfile = ogr.Open(wrsfile)
    wrsshape = wrsshpfile.GetLayer(0)

    pathrow = list()

    feature = wrsshape.GetNextFeature()

    while feature:

        wrsgeom = feature.GetGeometryRef()

        if wrsgeom.Intersects(fieldgeom):
            path = json.loads(feature.ExportToJson())['properties']['PATH']
            row = json.loads(feature.ExportToJson())['properties']['ROW']
            pathrow.append((path, row))

        feature = wrsshape.GetNextFeature()

    fieldshpfile = None
    wrsshpfile = None

    return pathrow

