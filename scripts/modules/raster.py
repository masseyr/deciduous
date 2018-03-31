import os
import numpy as np
from common import *
from osgeo import gdal, gdal_array, ogr, osr
np.set_printoptions(suppress=True)

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


class Raster:
    """
    Raster class to read and write raster from/to files and numpy arrays
    """

    def __init__(self, name, array=None, bnames=None, metadict=None,
                 dtype=None, shape=None, transform=None, crs_string=None):
        self.array = array
        self.bnames = bnames
        self.shape = shape
        self.transform = transform
        self.crs_string = crs_string
        self.name = name
        self.dtype = dtype
        self.metadict = metadict
        self.nodatavalue = None

    def __repr__(self):
        if self.shape is not None:
            return "<raster {ras} of size {bands}x{rows}x{cols}>".format(ras=os.path.basename(self.name),
                                                                         bands=self.shape[0],
                                                                         rows=self.shape[1], cols=self.shape[2])
        else:
            return "<raster with path {ras}>".format(ras=self.name,)

    def write_to_file(self, outfile=None):
        """
        Write raster to file, given all the properties
        :param self - Raster object
        :param outfile - Name of output file
        """
        if outfile is None:
            outfile = self.name
            outfile = file_rename_check(outfile)

        print('')
        print('Writing ' + outfile)
        print('')
        gtiffdriver = gdal.GetDriverByName('GTiff')
        fileptr = gtiffdriver.Create(outfile, self.shape[2], self.shape[1],
                                     self.shape[0], self.dtype)
        nbands = self.shape[0]
        fileptr.SetGeoTransform(self.transform)
        fileptr.SetProjection(self.crs_string)

        if nbands == 1:
            fileptr.GetRasterBand(1).WriteArray(self.array, 0, 0)
            fileptr.GetRasterBand(1).SetDescription(self.bnames[0])
            print('Writing band: ' + self.bnames[0])
        else:
            for i in range(0, nbands):
                fileptr.GetRasterBand(i + 1).WriteArray(self.array[i, :, :], 0, 0)
                fileptr.GetRasterBand(i + 1).SetDescription(self.bnames[i])
                # fileptr.GetRasterBand(i + 1).SetNoDataValue(-32759)
                print('Writing band: ' + self.bnames[i])

        fileptr.FlushCache()
        print('File written to disk!')
        fileptr = None

    @classmethod
    def initialize(cls, raster, get_array=False, band_order=None,
                   finite_only=True, nan_replacement=0.0, use_dict=None):
        """
        Initialize a raster object from a file
        :param cls - Raster object class <empty>
        :param raster - raster filename
        :param get_array - flag to include raster as 3 dimensional array (bool)
        :param band_order - band location array (int starting at 0; ignored if get_array is False)
        :param finite_only - flag to remove non-finite values from array (ignored if get_array is False)
        :param nan_replacement - replacement for all non-finite replacements
        :param use_dict
        (ignored if finite_only, get_array is false)
        :return raster object
        """

        if os.path.isfile(raster):

            # open file
            fileptr = gdal.Open(raster)

            # get shape metadata
            bands = fileptr.RasterCount
            rows = fileptr.RasterYSize
            cols = fileptr.RasterXSize

            # if get_array flag is true
            if get_array:

                # get band names
                names = list()

                # band order
                if band_order is None:
                    array3d = fileptr.ReadAsArray()

                    # if flag for finite values is present
                    if finite_only:
                        if np.isnan(array3d).any() or np.isinf(array3d).any():
                            array3d[np.isnan(array3d)] = nan_replacement
                            array3d[np.isinf(array3d)] = nan_replacement
                            print("Non-finite values replaced with " + str(nan_replacement))
                        else:
                            print("Non-finite values absent in file")

                    # get band names
                    for i in range(0, bands):
                        names.append(fileptr.GetRasterBand(i + 1).GetDescription())

                # band order present
                else:

                    # bands in array
                    n_array_bands = len(band_order)

                    # initialize array
                    array3d = np.zeros((n_array_bands, rows, cols),
                                       gdal_array.GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).DataType))

                    # read array and store the band values and name in array
                    for i, b in enumerate(band_order):
                        array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray()
                        names.append(fileptr.GetRasterBand(b + 1).GetDescription())

                    # if flag for finite values is present
                    if finite_only:
                        if np.isnan(array3d).any() or np.isinf(array3d).any():
                            array3d[np.isnan(array3d)] = nan_replacement
                            array3d[np.isinf(array3d)] = nan_replacement
                            print("Non-finite values replaced with " + str(nan_replacement))
                        else:
                            print("Non-finite values absent in file")

                # assign to empty class object
                raster_obj = cls(name=raster,
                                 array=array3d,
                                 bnames=names,
                                 shape=[bands, rows, cols],
                                 transform=fileptr.GetGeoTransform(),
                                 crs_string=fileptr.GetProjection(),
                                 dtype=fileptr.GetRasterBand(1).DataType,
                                 metadict=Raster.get_raster_metadict(raster))

            # if get_array is false
            else:
                # get band names
                names = list()
                for i in range(0, bands):
                    names.append(fileptr.GetRasterBand(i + 1).GetDescription())

                # assign to empty class object without the array
                raster_obj = cls(name=raster,
                                 bnames=names,
                                 shape=[bands, rows, cols],
                                 transform=fileptr.GetGeoTransform(),
                                 crs_string=fileptr.GetProjection(),
                                 dtype=fileptr.GetRasterBand(1).DataType,
                                 metadict=Raster.get_raster_metadict(raster))
            fileptr = None

            # remap band names
            if use_dict is not None:
                raster_obj.bnames = [use_dict[b] for b in raster_obj.bnames]

            return raster_obj
        else:
            raise ValueError('No matching file found on disk')

    @staticmethod
    def get_raster_metadict(filename):
        """
        Function to get all the spatial metadata associated with a geotiff raster
        """

        # open raster
        img_pointer = gdal.Open(filename)

        # get tiepoint, pixel size, pixel rotation
        geoMetadata = img_pointer.GetGeoTransform()

        # make dictionary of all the metadata
        metaDict = {'ulx': geoMetadata[0],
                    'uly': geoMetadata[3],
                    'xpixel': abs(geoMetadata[1]),
                    'ypixel': abs(geoMetadata[5]),
                    'rotationx': geoMetadata[2],
                    'rotationy': geoMetadata[4],
                    'columns': img_pointer.RasterXSize,  # columns from raster pointer
                    'rows': img_pointer.RasterYSize,  # rows from raster pointer
                    'bands': img_pointer.RasterCount,  # bands from raster pointer
                    'projection': img_pointer.GetProjection(),  # projection information from pointer
                    'name': os.path.basename(filename)}  # file basename

        # remove pointer
        img_pointer = None
        return metaDict

    def make_polygon_geojson_feature(self):
        """
        Make a feature geojson for the raster using its metaDict data
        """
        metaDict = self.metadict
        return {"type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                         [metaDict['ulx'], metaDict['uly']],
                         [metaDict['ulx'], metaDict['uly'] - (metaDict['ypixel'] * (metaDict['rows'] + 1))],
                         [metaDict['ulx'] + (metaDict['xpixel'] * (metaDict['columns'] + 1)),
                          metaDict['uly'] - (metaDict['ypixel'] * (metaDict['rows'] + 1))],
                         [metaDict['ulx'] + (metaDict['xpixel'] * (metaDict['columns'] + 1)), metaDict['uly']],
                         [metaDict['ulx'], metaDict['uly']]
                         ]]
                    },
                "properties": {
                    "name": metaDict['name'].split('.')[0]
                    },
                }

    def chk_for_empty_tiles(self):
        """
        check the tile for empty bands, return true if one exists
        :return: bool
        """
        fileptr = gdal.Open(self.name)

        filearr = fileptr.ReadAsArray()
        nb, ns, nl = filearr.shape

        truth_about_empty_bands = [np.isfinite(filearr[i, :, :]).any() for i in range(0, nb)]

        fileptr = None

        return any([not x for x in truth_about_empty_bands])

    def make_tile(self, tile_size_x, tile_size_y, out_path):
        """
        Make tiles from the tif file
        :param tile_size_y:
        :param tile_size_x:
        :param out_path:
        :return:
        """
        in_file = self.name
        bands, rows, cols = self.shape

        out_file_basename = os.path.basename(in_file).split('.')[0]

        for i in range(0, cols, tile_size_x):
            for j in range(0, rows, tile_size_y):

                if (cols - i) != 0 and (rows - j) != 0:

                    if (cols - i) < tile_size_x:
                        tile_size_x = cols - i + 1

                    if (rows - j) < tile_size_y:
                        tile_size_y = rows - j + 1

                    out_file = str(out_path) + os.path.sep + str(out_file_basename) + "_" + str(i) + "_" + str(
                        j) + ".tif"

                    try:
                        os.remove(out_file)
                    except OSError:
                        pass

                    print(out_file_basename)

                    run_string = "gdal_translate -of GTiff -ot Float32 -co TILED=YES -srcwin " + str(i) + ", " + \
                                 str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + \
                                 str(in_file) + " " + str(out_file)
                    os.system(run_string)

                    out_raster = Raster(out_file)
                    if out_raster.chk_for_empty_tiles():
                        print('Removing empty raster file: ' + os.path.basename(out_file))
                        os.remove(out_file)
                        print('')
                    out_raster = None
