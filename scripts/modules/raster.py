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

    def __init__(self,
                 name,
                 array=None,
                 bnames=None,
                 metadict=None,
                 dtype=None,
                 shape=None,
                 transform=None,
                 crs_string=None):

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
    def initialize(cls,
                   raster_name,
                   get_array=False,
                   band_order=None,
                   finite_only=True,
                   nan_replacement=0.0,
                   use_dict=None):

        """
        Initialize a raster object from a file
        :param cls - Raster object class <empty>
        :param raster_name - raster filename
        :param get_array - flag to include raster as 3 dimensional array (bool)
        :param band_order - band location array (int starting at 0; ignored if get_array is False)
        :param finite_only - flag to remove non-finite values from array (ignored if get_array is False)
        :param nan_replacement - replacement for all non-finite replacements
        :param use_dict
        (ignored if finite_only, get_array is false)
        :return raster object
        """

        if os.path.isfile(raster_name):

            # open file
            fileptr = gdal.Open(raster_name)

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
                raster_obj = cls(name=raster_name,
                                 array=array3d,
                                 bnames=names,
                                 shape=[bands, rows, cols],
                                 transform=fileptr.GetGeoTransform(),
                                 crs_string=fileptr.GetProjection(),
                                 dtype=fileptr.GetRasterBand(1).DataType,
                                 metadict=Raster.get_raster_metadict(raster_name))

            # if get_array is false
            else:
                # get band names
                names = list()
                for i in range(0, bands):
                    names.append(fileptr.GetRasterBand(i + 1).GetDescription())

                # assign to empty class object without the array
                raster_obj = cls(name=raster_name,
                                 bnames=names,
                                 shape=[bands, rows, cols],
                                 transform=fileptr.GetGeoTransform(),
                                 crs_string=fileptr.GetProjection(),
                                 dtype=fileptr.GetRasterBand(1).DataType,
                                 metadict=Raster.get_raster_metadict(raster_name))
            fileptr = None

            # remap band names
            if use_dict is not None:
                raster_obj.bnames = [use_dict[b] for b in raster_obj.bnames]

            return raster_obj
        else:
            raise ValueError('No matching file found on disk')

    @property
    def chk_for_empty_tiles(self):
        """
        check the tile for empty bands, return true if one exists
        :return: bool
        """
        if os.path.isfile(self.name):
            fileptr = gdal.Open(self.name)

            filearr = fileptr.ReadAsArray()
            nb, ns, nl = filearr.shape

            truth_about_empty_bands = [np.isfinite(filearr[i, :, :]).any() for i in range(0, nb)]

            fileptr = None

            return any([not x for x in truth_about_empty_bands])
        else:
            raise ValueError("File does not exist.")

    def make_tiles(self,
                   tile_size_x,
                   tile_size_y,
                   out_path):

        """
        Make tiles from the tif file
        :param tile_size_y: Tile size along x
        :param tile_size_x: tile size along y
        :param out_path: Output folder
        :return:
        """

        # get all the file parameters and metadata
        in_file = self.name
        bands, rows, cols = self.shape

        if 0 < tile_size_x <= cols and 0 < tile_size_y <= rows:

            if self.metadict is not None:

                # assign variables
                metadict = self.metadict
                dtype = metadict['datatype']
                ulx, uly = [metadict['ulx'], metadict['uly']]
                px, py = [metadict['xpixel'], metadict['ypixel']]
                rotx, roty = [metadict['rotationx'], metadict['rotationy']]
                crs_string = metadict['projection']

                # file name without extension (e.g. .tif)
                out_file_basename = os.path.basename(in_file).split('.')[0]

                # open file
                in_file_ptr = gdal.Open(in_file)

                # loop through the tiles
                for i in range(0, cols, tile_size_x):
                    for j in range(0, rows, tile_size_y):

                        if (cols - i) != 0 and (rows - j) != 0:

                            # check size of tiles for edge tiles
                            if (cols - i) < tile_size_x:
                                tile_size_x = cols - i + 1

                            if (rows - j) < tile_size_y:
                                tile_size_y = rows - j + 1

                            # name of the output tile
                            out_file_name = str(out_path) + os.path.sep + str(out_file_basename) + \
                                            "_" + str(i + 1) + "_" + str(j + 1) + ".tif"

                            # check if file already exists
                            out_file_name = file_remove_check(out_file_name)

                            # get/calculate spatial parameters
                            new_ul = [ulx + i * px, uly + j * py]
                            new_lr = [new_ul[0] + px * tile_size_x, new_ul[1] + py * tile_size_y]
                            new_transform = (new_ul[0], px, rotx, new_ul[1], py, roty)

                            # initiate output file
                            driver = gdal.GetDriverByName("GTiff")
                            out_file_ptr = driver.Create(out_file_name, tile_size_x, tile_size_y, bands, dtype)

                            for k in range(0, bands):
                                # get data
                                band_name = in_file_ptr.GetRasterBand(k + 1).GetDescription()
                                band = in_file_ptr.GetRasterBand(k + 1)
                                band_data = band.ReadAsArray(i, j, tile_size_x, tile_size_y)

                                # put data
                                out_file_ptr.GetRasterBand(k + 1).WriteArray(band_data, 0, 0)
                                out_file_ptr.GetRasterBand(k + 1).SetDescription(band_name)

                            # set spatial reference and projection parameters
                            out_file_ptr.SetGeoTransform(new_transform)
                            srs = osr.SpatialReference()
                            srs.ImportFromWkt(crs_string)
                            out_file_ptr.SetProjection(crs_string)

                            # delete pointers
                            out_file_ptr.FlushCache()  # save to disk
                            out_file_ptr = None
                            driver = None

                            # check for empty tiles
                            out_raster = Raster(out_file_name)
                            if out_raster.chk_for_empty_tiles:
                                print('Removing empty raster file: ' + os.path.basename(out_file_name))
                                os.remove(out_file_name)
                                print('')

                            # unassign
                            out_raster = None
            else:
                raise AttributeError("Metadata dictionary does not exist.")
        else:
            raise ValueError("Tile size {}x{} is larger than original raster {}x{}.".format(tile_size_y,
                                                                                            tile_size_x,
                                                                                            self.shape[1],
                                                                                            self.shape[2]))

    @staticmethod
    def get_raster_metadict(file_name):
        """
        Function to get all the spatial metadata associated with a geotiff raster
        """

        if os.path.isfile(file_name):
            # open raster
            img_pointer = gdal.Open(file_name)

            # get tiepoint, pixel size, pixel rotation
            geometadata = img_pointer.GetGeoTransform()

            # make dictionary of all the metadata
            meta_dict = {'ulx': geometadata[0],
                         'uly': geometadata[3],
                         'xpixel': abs(geometadata[1]),
                         'ypixel': abs(geometadata[5]),
                         'rotationx': geometadata[2],
                         'rotationy': geometadata[4],
                         'datatype': img_pointer.GetRasterBand(1).DataType,
                         'columns': img_pointer.RasterXSize,  # columns from raster pointer
                         'rows': img_pointer.RasterYSize,  # rows from raster pointer
                         'bands': img_pointer.RasterCount,  # bands from raster pointer
                         'projection': img_pointer.GetProjection(),  # projection information from pointer
                         'name': os.path.basename(file_name)}  # file basename

            # remove pointer
            img_pointer = None

            return meta_dict
        else:
            raise ValueError("File does not exist.")

    def make_polygon_geojson_feature(self):
        """
        Make a feature geojson for the raster using its metaDict data
        """

        meta_dict = self.metadict

        if meta_dict is not None:
            return {"type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                             [meta_dict['ulx'], meta_dict['uly']],
                             [meta_dict['ulx'], meta_dict['uly'] - (meta_dict['ypixel'] * (meta_dict['rows'] + 1))],
                             [meta_dict['ulx'] + (meta_dict['xpixel'] * (meta_dict['columns'] + 1)),
                              meta_dict['uly'] - (meta_dict['ypixel'] * (meta_dict['rows'] + 1))],
                             [meta_dict['ulx'] + (meta_dict['xpixel'] * (meta_dict['columns'] + 1)), meta_dict['uly']],
                             [meta_dict['ulx'], meta_dict['uly']]
                             ]]
                        },
                    "properties": {
                        "name": meta_dict['name'].split('.')[0]
                        },
                    }
        else:
            raise AttributeError("Metadata dictionary does not exist.")
