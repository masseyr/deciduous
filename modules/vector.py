from osgeo import ogr, osr
from common import *
import csv

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


OGR_GEOM_DEF = {
                1: 'point',
                2: 'line',
                3: 'polygon',
                4: 'multipoint',
                5: 'multilinestring',
                6: 'multipolygon',
                0: 'geometry',
                100: 'no geometry',
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
        self.spref = None  # spatial reference
        self.spref_str = None  # spatial reference string
        self.attribute_def = dict()  # dictionary of attribute definitions
        # {'name':'str', 'count':'int','rate':'float', 'auth':'bool'}
        self.attributes = list()

        self.features = list()  # list of dictionaries with WKT geometry and
        # attributes as in attributes_def property
        # e.g. [{'id':1,'name':'r2d2','geometry':'POINT((12.11 11.12))'}, ...]
        self.fields = list()

        self.wktlist = None
        self.nfeat = None
        self.layer = None

        self.precision = 8  # Precision is set only for float attributes
        self.width = 50  # Width is set for string characters
        self.epsg = 4326  # EPSG SRID, default: 4326 (geographic lat/lon, WGS84)

    def __repr__(self):
        """
        object representation
        :return: String
        """
        return "<Vector object at {} of type {} ".format(hex(id(self)),
                                                         OGR_GEOM_DEF[self.geometry_type].upper()) + \
               "with {} feature(s) and {} attribute(s) >".format(str(len(self.features)),
                                                                 str(len(self.fields)))

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
        :return: Nonetype, stores list of dictionaries in self.features
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

                    self.features.append(feat)

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
        for feat_dict in self.features:

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

    @classmethod
    def vector_from_string(cls,
                           geom_strings,
                           geom_string_type='wkt',
                           spref=None,
                           spref_string=None,
                           spref_string_type='wkt',
                           vector_type=None,
                           out_epsg=4326,
                           attributes=None,
                           attribute_types=None):
        """
        Make a vector object from a list of geometries in string (json, wkt, or wkb) format.
        :param geom_strings: Single or a list of geometries in WKT format
        :param geom_string_type: Geometry string type (e.g. 'wkt', 'json', 'wkb'; default: 'wkt)
        :param spref: OSR Spatial reference object
        :param spref_string: WKT representation of the spatial reference for the Vector object
        :param spref_string_type: Spatial reference string type (e.g. 'wkt', 'proj4', 'epsg'; default: 'wkt)
        :param vector_type: Type of vector geometry (e.g. 'point','polygon','multipolygon','line'; default: 'polygon')
        :param out_epsg: EPSG SRID for the geometry object
        :param attributes: Dictionary or list of dictionaries of feature attributes.
                           The 'key' names in this list of dicts should match exactly with attribute_types
        :param attribute_types: Dictionary of feature attribute names with their OGR datatypes.
                                This is the attribute definition dictionary.
                                This dictionary must match the 'attributes'.
        :return: Vector object
        """

        vector = cls()
        vector.nfeat = len(geom_strings)

        if type(geom_strings).__name__ != 'list':
            geom_strings = [geom_strings]
        if type(attributes).__name__ != 'list':
            attributes = [attributes]

        if geom_string_type == 'wkt':
            geoms = list(ogr.CreateGeometryFromWkt(geom_string) for geom_string in geom_strings)
        elif geom_string_type == 'json':
            geoms = list(ogr.CreateGeometryFromJson(geom_string) for geom_string in geom_strings)
        elif geom_string_type == 'wkb':
            geoms = list(ogr.CreateGeometryFromWkb(geom_string) for geom_string in geom_strings)
        else:
            raise TypeError("Unsupported geometry type")

        if spref is None:
            spref = osr.SpatialReference()

            if spref_string is not None:
                if spref_string_type == 'wkt':
                    res = spref.ImportFromWkt(spref_string)
                elif spref_string_type == 'proj4':
                    res = spref.ImportFromProj4(spref_string)
                elif spref_string_type == 'epsg':
                    res = spref.ImportFromEPSG(spref_string)
                else:
                    raise RuntimeError("No spatial reference")
            else:
                res = spref.ImportFromEPSG(out_epsg)

        vector.spref = spref
        vector.spref_str = spref.ExportToWkt()

        # get driver to write to memory
        memory_driver = ogr.GetDriverByName('Memory')
        temp_datasource = memory_driver.CreateDataSource('out')
        vector.data_source = temp_datasource

        if vector_type is None:
            geom_type = geoms[0].GetGeometryType()
        elif type(vector_type).__name__ == 'str':
            geom_type = OGR_TYPE_DEF[vector_type]
        elif type(vector_type).__name__ == 'int' or \
                type(vector_type).__name__ == 'long':
            geom_type = vector_type
        else:
            raise ValueError("Invalid geometry type")

        vector.geometry_type = geom_type

        # create layer in memory
        temp_layer = temp_datasource.CreateLayer('temp_layer',
                                                 srs=spref,
                                                 geom_type=geom_type)
        vector.layer = temp_layer
        vector.fields = list()
        vector.attribute_def = attribute_types

        if (attributes is not None) != (attribute_types is not None):
            raise RuntimeError('One of attribute values or attribute definitions is missing')
        elif attributes is not None and attribute_types is not None:
            for attr_name, attr_val in attributes[0].items():
                if attr_name not in attribute_types:
                    raise RuntimeError('Attribute values supplied for undefined attributes')
        else:
            attribute_types = {'GeomID': ogr.OFTInteger}
            attributes = list({'GeomID': i} for i in range(0, len(geom_strings)))

        # create the attribute fields in the layer
        for attr_name, attr_type in attribute_types.items():
            fielddefn = ogr.FieldDefn(attr_name, OGR_FIELD_DEF[attr_type])
            vector.fields.append(fielddefn)
            res = temp_layer.CreateField(fielddefn)

        # layer definition with new fields
        temp_layer_definition = temp_layer.GetLayerDefn()
        vector.wktlist = list()
        vector.attributes = attributes

        for geom in geoms:
            # create new feature using geometry
            temp_feature = ogr.Feature(temp_layer_definition)
            temp_feature.SetGeometry(geom)

            # copy attributes to each feature, the order is the order of features
            for attribute in attributes:
                for attrname, attrval in attribute.items():
                    temp_feature.SetField(attrname, attrval)

            # create feature in layer
            temp_layer.CreateFeature(temp_feature)

            vector.features.append(temp_feature)
            vector.wktlist.append(geom.ExportToWkt())

        return vector

    def split(self):
        """
        Method to split (or flatten) multi-geometry vector to multiple single geometries vector.
        The vector can have single or multiple multi-geometry features
        :return: Vector object with all single type geometries
        """

        if self.geometry_type < 4:
            return self
        else:

            # get layer information
            layr = self.layer

            # get field (attribute) information
            feat_defns = layr.GetLayerDefn()
            nfields = feat_defns.GetFieldCount()
            field_defs = list(feat_defns.GetFieldDefn(i) for i in range(0, nfields))

            # create list of features with geometries and attributes
            out_feat_list = list()

            out_type = None

            # loop thru all the feature and all the multi-geometries in each feature
            for feat in self.features:

                geom_ref = feat.GetGeometryRef()
                n_geom = geom_ref.GetGeometryCount()

                feat_attr = dict()
                for field in field_defs:
                    feat_attr[field.GetName()] = feat.GetField(field.GetName())

                # create list of features from multi-geometries
                for j in range(0, n_geom):
                    temp_feat_dict = dict()
                    temp_feat_dict['geom'] = geom_ref.GetGeometryRef(j)
                    temp_feat_dict['attr'] = feat_attr

                    # find output geometry type
                    if out_type is None:
                        out_type = temp_feat_dict['geom'].GetGeometryType()

                    # append to output list
                    out_feat_list.append(temp_feat_dict)

            # get driver to write to memory
            memory_driver = ogr.GetDriverByName('Memory')
            temp_datasource = memory_driver.CreateDataSource('out')
            temp_layer = temp_datasource.CreateLayer('temp_layer',
                                                     srs=self.spref,
                                                     geom_type=out_type)

            # initialize vector
            temp_vector = Vector()

            # update features and crs
            temp_vector.nfeat = len(out_feat_list)
            temp_vector.type = out_type
            temp_vector.crs = self.spref
            temp_vector.layer = temp_layer
            temp_vector.data_source = temp_datasource
            temp_vector.wkt_list = list()

            # create field in layer
            for field in field_defs:
                res = temp_layer.CreateField(field)
                temp_vector.fields.append(field)

            temp_layer_definition = temp_layer.GetLayerDefn()

            # create new features using geometry
            for out_feat in out_feat_list:

                # add geometry and attributes
                temp_feature = ogr.Feature(temp_layer_definition)
                temp_feature.SetGeometry(out_feat['geom'])

                for field_name, field_val in out_feat['attr'].items():
                    temp_feature.SetField(field_name,
                                          field_val)

                # create feature in layer
                temp_layer.CreateFeature(temp_feature)

                temp_vector.features.append(temp_feature)
                temp_vector.wkt_list.append(out_feat['geom'].ExportToWkt())

            return temp_vector


