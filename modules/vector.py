from osgeo import ogr, osr
import csv
from common import *


__all__ = ['OGR_TYPE_DEF',
           'OGR_FIELD_DEF',
           'Vector']


OGR_FIELD_DEF = {
    'int': ogr.OFTInteger,
    'float': ogr.OFTReal,
    'str': ogr.OFTString,
    'bool': ogr.OFTInteger,
    'NoneType': ogr.OFSTNone
}


OGR_TYPE_DEF = {
            'point': 1,
            'line': 2,
            'polygon': 3,
            'multipoint': 4,
            'multilinestring': 5,
            'multipolygon': 6,
            'geometry': 0,
            'no geometry': 100
}


class Vector(object):

    def __init__(self,
                 vectorfile=None,
                 datasource=None,
                 name=None,
                 source_type=None):
        """
        Constructor
        :param vectorfile: Name of file to write to disk
        :param datasource: GDAL data source object
        :param name: Name of the vector
        :param source_type: Driver that should be used to initialize the layer in gdal
                     (e.g. 'ESRI Shapefile', or 'GeoJSON')
        """

        self.vectorfile = vectorfile
        self.datasource = datasource
        self.name = name
        self.source_type = source_type

        self.geometry_type = None  # geometry type (e.g.: 'point','multipoint','polygon')
        self.spref = None  # spatial reference WKT string
        self.attribute_def = dict()  # dictionary of attribute definitions
        # {'name':'str', 'count':'int','rate':'float', 'auth':'bool'}

        self.featlist = list()  # list of dictionaries with WKT geometry and
        # attributes as in attributes_def property
        # e.g. [{'id':1,'name':'r2d2','geometry':'POINT((12.11 11.12))'}, ...]

        self.precision = 8  # Precision is set only for float attributes
        self.width = 50  # Width is set for string characters
        self.epsg = 4326  # EPSG SRID, default: 4326 (geographic lat/lon, WGS84)

    def __repr__(self):
        """
        object representation
        :return: String
        """
        return "<Vector object at {}".format(hex(id(self)))

    def point_features_from_csv(self,
                                csvfile=None,
                                header=True,
                                column_names=None,
                                geometry_columns=None):
        """
        Method to extract point features with attributes from a csv file
        :param csvfile: Name of the csv file
        :param header: If the csv file has a header
        :param column_names: Names of the columns for the csv file; ignored if header=True
        :param geometry_columns: Column names containing point geometry (e.g. ['lat','lon']
                                 Should be in order of lat,lon or x,y
        :return: Nonetype, stores list of dictionaries in self.featlist
        """

        def _get_rows(filename):
            with open(filename, "r") as f:
                reader = csv.reader(f)
                for data_row in reader:
                    yield data_row

        xloc = None
        yloc = None

        for i, row in enumerate(_get_rows(csvfile)):

            if i == 0:
                if header:
                    column_names = row

                xloc = Sublist(column_names) == geometry_columns[0]
                yloc = Sublist(column_names) == geometry_columns[1]

            else:
                if xloc is None or yloc is None or xloc == [] or yloc == []:
                    raise ValueError('Column name mismatch or no header in file')

                else:
                    feat = dict()
                    point = (row[xloc], row[yloc])
                    geom = self.wkt_from_coords(point,
                                                geom_type='point')
                    feat['geometry'] = geom

                    for j, name in enumerate(column_names):
                        if j != xloc and j != yloc:
                            feat[name] = row[j]

                    self.featlist.append(feat)

    @staticmethod
    def wkt_from_coords(coords,
                        geom_type='point'):

        """
        Method to return WKT string representation from a list
        :param coords: List of tuples [(x1,y1),(x2,y2),...] for multipoint
                       or a single tuple (x, y) in case of 'point'
        :param geom_type: multipoint or point
        :return: WKT string representation
        """

        if geom_type.upper() == 'POINT':
            tempstring = ' '.join(coords)
            wktstring = 'POINT({})'.format(tempstring)

        elif geom_type.upper() == 'MULTIPOINT':
            tempstring = '), ('.join(list(' '.join([str(x), str(y)]) for (x, y) in coords))
            wktstring = 'MULTIPOINT(({}))'.format(tempstring)

        elif geom_type.upper() == 'POLYGON':

            tempstring = ', '.join(list(' '.join([str(x), str(y)]) for (x, y) in coords))
            wktstring = 'POLYGON(({}))'.format(tempstring)

        elif geom_type.upper() == 'LINESTRING' or geom_type.upper() == 'LINE':

            tempstring = ', '.join(list(' '.join([str(x), str(y)]) for (x, y) in coords))
            wktstring = 'LINESTRING({})'.format(tempstring)

        else:
            raise ValueError("Unknown geometry type")

        return wktstring

    def construct_point_vector(self):
        """
        Method to return a point vector in memory or written to file
        :return: Vector in memory or in file
        """

        # datasource
        out_driver = ogr.GetDriverByName(self.source_type)
        out_datasource = out_driver.CreateDataSource(self.vectorfile)

        # spatial reference
        spref = osr.SpatialReference()
        res = spref.ImportFromEPSG(self.epsg)

        # layer
        point_layer = out_datasource.CreateLayer(self.name,
                                                 srs=spref,
                                                 geom_type=OGR_TYPE_DEF[self.geometry_type])

        # create fields
        for attribute_name, attribute_type in self.attribute_def.items():

            temp_field = ogr.FieldDefn()
            temp_field.SetName(attribute_name)
            temp_field.SetType(OGR_FIELD_DEF[attribute_type])

            if attribute_type == 'float':
                temp_field.SetPrecision(self.precision)
            if attribute_type == 'str':
                temp_field.SetWidth(self.width)

            res = point_layer.CreateField(temp_field)

        # get layer definition
        layer_def = point_layer.GetLayerDefn()

        # add features
        for feat_dict in self.featlist:

            # create feature
            feature = ogr.Feature(layer_def)

            # add attributes and geometry
            for key, value in feat_dict.items():
                if key == 'geometry':
                    geom = ogr.CreateGeometryFromWkt(value)
                    feature.SetGeometry(geom)
                else:
                    feature.SetField(key,
                                     value)

            # create feature in layer
            point_layer.CreateFeature(feature)

        if self.vectorfile:

            out_driver = None
            out_datasource = None

        else:
            self.datasource = out_datasource

