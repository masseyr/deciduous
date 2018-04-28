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
        'EVI_2': 'EVI_Period3'},

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
        'EVI_2': 'EVI_Period3'},

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


# module to identify, path row, bounding box, and intersecting landsat footprints
def find_intersecting_tiles(fieldfile, wrsfile):
    fieldshpfile = ogr.Open(fieldfile)
    fieldshape = fieldshpfile.GetLayer(0)
    fieldgeom = list()
    for i in range(0, fieldshape.GetFeatureCount()):
        feat = fieldshape.GetFeature(i)
        coordinates = json.loads(feat.ExportToJson())['geometry']['coordinates'][0][0:-1]
        fieldgeom.append(Polygon(coordinates))
    if fieldshape.GetFeatureCount() == 1:
        fieldgeom = fieldgeom[0]

    ptlist = [[]]

    for i in range(0, fieldshape.GetFeatureCount()):
        feature = fieldshape.GetFeature(i)
        ptlist.extend(json.loads(feature.ExportToJson())['geometry']['coordinates'][0][0:-1])

    # bounding box and coordinates
    leftlim = min([pt[0] for pt in ptlist[1:]])
    rightlim = max([pt[0] for pt in ptlist[1:]])
    lowlim = min([pt[1] for pt in ptlist[1:]])
    uplim = max([pt[1] for pt in ptlist[1:]])

    boundbox = Polygon([[leftlim, lowlim],
                        [leftlim, uplim],
                        [rightlim, uplim],
                        [rightlim, lowlim]])

    boundboxptlist = [list(elem) for elem in list(mapping(boundbox)['coordinates'])][0]
    centroid = [boundbox.centroid.x, boundbox.centroid.y]

    vertxdistlist = list(_eucl_dist(boundboxpt[0], centroid[0], boundboxpt[1], centroid[1])
                         for boundboxpt in boundboxptlist)

    buf = 0.05 * max(vertxdistlist)

    # buffered bounding box
    boundbox = Polygon([[leftlim - buf, lowlim - buf],
                        [leftlim - buf, uplim + buf],
                        [rightlim + buf, uplim + buf],
                        [rightlim + buf, lowlim - buf]])

    wrsshpfile = ogr.Open(wrsfile)
    wrsshape = wrsshpfile.GetLayer(0)
    featlist = [[]]

    for i in range(wrsshape.GetFeatureCount()):
        feature = wrsshape.GetFeature(i)
        featlist.append(feature)

    # list of Landsat footprints and centers
    polylist = list(Polygon(json.loads(feature.ExportToJson())['geometry']['coordinates'][0][0:-1])
                    for feature in featlist[1:])

    pathrow = list([json.loads(feature.ExportToJson())['properties']['PATH'],
                    json.loads(feature.ExportToJson())['properties']['ROW']] for feature in featlist[1:])

    ftprnt_indx = list(indx for indx, ftprnt in enumerate(polylist) if ftprnt.intersects(boundbox))

    final_indx = list(indx for indx, ftprnt in enumerate([polylist[i] for i in ftprnt_indx])
                      if ftprnt.intersects(fieldgeom))

    scene_centers = list(list(mapping(poly.centroid)['coordinates'])
                         for poly in [polylist[i] for i in [ftprnt_indx[k] for k in final_indx]])

    return [pathrow[i] for i in [ftprnt_indx[k] for k in final_indx]], scene_centers
