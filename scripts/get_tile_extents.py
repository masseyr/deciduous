import glob
import json
from sys import argv
from osgeo import ogr, osr
from shutil import copyfile
from modules import *


# main program
if __name__ == '__main__':
    # import folder name containing all the rasters, and the shapefile to output
    script, rasterfiledir, boundaryfile, tiledir, outshpfile = argv

    print('------------------------------------------')
    print('Running ' + script)
    print('------------------------------------------')

    # get shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # open boundary file and get geometry
    boundptr = driver.Open(boundaryfile)
    boundlyr = boundptr.GetLayerByIndex(0)
    boundfeat = boundlyr.GetFeature(0)
    boundgeom = boundfeat.GetGeometryRef()

    print('Working in directory: ' + rasterfiledir)
    print('------------------------------------------')
    print('Files found: ')

    # initialize filelist
    filelist = list()

    # open out file pointer
    outfileptr = driver.CreateDataSource(outshpfile)

    # loop thru all files in the folder
    for filename in glob.glob(rasterfiledir + os.path.sep + "*.tif"):
        filelist.append(filename)
        print(os.path.basename(filename))
        
    print('------------------------------------------')
    print('Total files: ' + str(len(filelist)))
    print('------------------------------------------')
    print('Creating shapefile: ' + outshpfile)

    # get meta data from the first raster and make feature geojson
    raster_obj = Raster(filelist[0])
    raster_metaDict = raster_obj.get_raster_metadict()

    # get projection information
    spref = osr.SpatialReference(raster_metaDict['projection'])
    raster_metaDict = None

    # initialize layer for the features
    layer = outfileptr.CreateLayer("Tiles", spref, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("Name", ogr.OFTString))
    defn = layer.GetLayerDefn()

    # loop thru all files in the folder
    for i, filename in enumerate(filelist):

        # get meta data from the first raster and make feature geojson
        raster_obj2 = Raster(filename)
        raster_metaDict = raster_obj2.get_raster_metadict()
        raster_feat_geojson = raster_obj2.make_polygon_geojson_feature()

        print('Adding feature ' + str(i + 1) + ' : ' + raster_metaDict["name"])

        # output geometry
        json_string = json.dumps(raster_feat_geojson['geometry'])
        outgeom = ogr.CreateGeometryFromJson(json_string)
        
        if boundgeom.Intersects(outgeom):

            # copy tif file to desired folder
            copyfile(filename, tiledir + os.path.sep + os.path.basename(filename))

            # create output feature
            outFeat = ogr.Feature(defn)

            # create/set geometry of the feature
            outFeat.SetGeometry(outgeom)

            # set properties of output features
            outFeat.SetField("Name", raster_metaDict["name"])

            # create the feature in the layer
            layer.CreateFeature(outFeat)
            
            print('2')
            
            # destroy the objects
            outFeat = None
            raster_metaDict = None
            raster_feat_geojson = None

    # destroy file pointer and layer objects
    outfileptr = layer = None
    bound_ptr = None





