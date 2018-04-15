import os
import fnmatch
import json
import numpy as np
from shapely.geometry import Polygon, mapping
import pandas as pd
from osgeo import ogr

"""
This module houses functions specific to the data used
"""

__all__ = ['bname_dict',
           'get_rfpickle_info',
           'get_TCdata_filepath',
           'TCserver',
           'read_y_param_from_summary',
           'find_intersecting_tiles']


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
    """
    Tree cover files ftp file path dictionary
    :param path: WRS2 path
    :param row: WRS2 row
    :param year: Year
    :return: Dictionary
    """
    return {
               'filestr': "/glcf/LandsatTreecover/WRS2/p{p}/r{r}/p{p}r{r}_TC_2010/p{p}r{r}_TC_{y}.tif.gz".format(
                p=str(path).zfill(3),
                r=str(row).zfill(3),
                y=str(year).zfill(4)),
               'errfilestr': "/glcf/LandsatTreecover/WRS2/p{p}/r{r}/p{p}r{r}_TC_2010/p{p}r{r}_TC_{y}_err.tif.gz".format(
                p=str(path).zfill(3),
                r=str(row).zfill(3),
                y=str(year).zfill(4))
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


def _eucl_dist(x1, x2, y1, y2):
    """Euclidean distance between (x1,y1) and (x2,y2)"""
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


def find_intersecting_tiles(infile, wrs2file):
    """
    Find intersecting Landsat tiles using wrs2 system
    :param infile: input polygon(s) file
    :param wrs2file: wrs2 file
    :return: list of path and row
    """
    # open input shapefile
    inshpfile = ogr.Open(infile)
    inshape = inshpfile.GetLayer(0)
    ptlist = [[]]

    # get list of all points in the input shapefile
    for i in range(0, inshape.GetFeatureCount()):
        feature = inshape.GetFeature(i)
        ptlist.extend(json.loads(feature.ExportToJson())['geometry']['coordinates'][0][0:-1])

    # find limits of the points in the input shapefile
    leftlim = min([pt[0] for pt in ptlist[1:]])
    rightlim = max([pt[0] for pt in ptlist[1:]])
    lowlim = min([pt[1] for pt in ptlist[1:]])
    uplim = max([pt[1] for pt in ptlist[1:]])

    # find the bounding box of the shapefile
    boundbox = Polygon([[leftlim, lowlim],
                        [leftlim, uplim],
                        [rightlim, uplim],
                        [rightlim, lowlim]])

    # get all the corners of the bounding box
    boundboxptlist = list(list(elem) for elem in list(mapping(boundbox)['coordinates']))

    # controid of bounding box
    centroid = [boundbox.centroid.x, boundbox.centroid.y]

    # centroid to vertex distance
    vertxdistlist = list(_eucl_dist(boundboxpt[0], centroid[0],
                         boundboxpt[1], centroid[1]) for boundboxpt in boundboxptlist)

    # add buffer
    buf = 0.05 * max(vertxdistlist)

    # buffered bounding box; this reduces the number of tiles
    #  the input polygon(s) will be compared to
    boundbox = Polygon([[leftlim - buf, lowlim - buf],
                        [leftlim - buf, uplim + buf],
                        [rightlim + buf, uplim + buf],
                        [rightlim + buf, lowlim - buf]])

    # open wrs2 shapefile
    wrsshpfile = ogr.Open(wrs2file)
    wrsshape = wrsshpfile.GetLayer(0)
    featlist = [[]]

    # get all features
    for i in range(wrsshape.GetFeatureCount()):
        feature = wrsshape.GetFeature(i)
        featlist.append(feature)

    # list of Landsat footprints from wrs2file
    LSpolylist = list(Polygon(json.loads(feature.ExportToJson())['geometry']['coordinates'][0][0:-1]) for feature in
                      featlist[1:])

    # list of landsat path and row
    LSpathrow = list([json.loads(feature.ExportToJson())['properties']['PATH'],
                  json.loads(feature.ExportToJson())['properties']['ROW']] for feature in featlist[1:])

    # find the footprints that intersect with budffered bounding box
    LSftprnt_indx = list(indx for indx, LSftprnt in enumerate(LSpolylist) if LSftprnt.intersects(boundbox))

    # inside the bounding box, find the foot prints that intersect with polygon
    indx = list()  # initialize indx
    for i in range(0, inshape.GetFeatureCount()):  # iterate over features
        feature = inshape.GetFeature(i)
        for j, LSscene in enumerate([LSpolylist[i] for i in LSftprnt_indx]):  # iterate over wrs2 tiles
            if Polygon(json.loads(feature.ExportToJson())['geometry']['coordinates'][0][0:-1]).intersects(LSscene):
                indx.extend(j)  # add to index if intersects

    # find unique indices
    indx = [LSftprnt_indx[i] for i in np.unique(indx)]

    # find LS path row
    path_row = [LSpathrow[i] for i in indx]

    return path_row
