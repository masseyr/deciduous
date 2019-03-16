import numpy as np
from common import *
from osgeo import gdal, gdal_array
np.set_printoptions(suppress=True)

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


__all__ = ['Raster']


class Raster:
    """
    Class to read and write rasters from/to files and numpy arrays
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
        self.datasource = None
        self.shape = shape
        self.transform = transform
        self.crs_string = crs_string
        self.name = name
        self.dtype = dtype
        self.metadict = metadict
        self.nodatavalue = None
        self.tile_grid = list()
        self.ntiles = None
        self.init = False

    def __repr__(self):
        if self.shape is not None:
            return "<raster {ras} of size {bands}x{rows}x{cols}>".format(ras=Handler(self.name).basename,
                                                                         bands=self.shape[0],
                                                                         rows=self.shape[1],
                                                                         cols=self.shape[2])
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
            outfile = Handler(filename=outfile).file_remove_check()

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

                if self.nodatavalue is not None:
                    fileptr.GetRasterBand(i + 1).SetNoDataValue(self.nodatavalue)

                print('Writing band: ' + self.bnames[i])

        fileptr.FlushCache()
        print('File written to disk!')
        fileptr = None

    def initialize(self,
                   get_array=False,
                   band_order=None,
                   finite_only=True,
                   nan_replacement=0.0,
                   use_dict=None,
                   sensor=None):

        """
        Initialize a raster object from a file
        :param get_array: flag to include raster as 3 dimensional array (bool)
        :param band_order: band location array (int starting at 0; ignored if get_array is False)
        :param finite_only: flag to remove non-finite values from array (ignored if get_array is False)
        :param nan_replacement: replacement for all non-finite replacements
        :param use_dict: Dictionary to use for renaming bands
        :param sensor: Sensor to be used with dictionary (resources.bname_dict)
        (ignored if finite_only, get_array is false)
        :return raster object
        """
        self.init = True
        raster_name = self.name

        if Handler(raster_name).file_exists():

            # open file
            fileptr = gdal.Open(raster_name)
            self.datasource = fileptr

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
                    print('Reading bands: {}'.format(" ".join([str(b) for b in band_order])))

                    # bands in array
                    n_array_bands = len(band_order)

                    # initialize array
                    array3d = np.zeros((n_array_bands, rows, cols),
                                       gdal_array.GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).DataType))

                    # read array and store the band values and name in array
                    for i, b in enumerate(band_order):
                        bandname = fileptr.GetRasterBand(b + 1).GetDescription()
                        print('Reading band {}'.format(bandname))
                        array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray()
                        names.append(bandname)

                    # if flag for finite values is present
                    if finite_only:
                        if np.isnan(array3d).any() or np.isinf(array3d).any():
                            array3d[np.isnan(array3d)] = nan_replacement
                            array3d[np.isinf(array3d)] = nan_replacement
                            print("Non-finite values replaced with " + str(nan_replacement))
                        else:
                            print("Non-finite values absent in file")

                # assign to empty class object
                self.array = array3d
                self.bnames = names
                self.shape = [bands, rows, cols]
                self.transform = fileptr.GetGeoTransform()
                self.crs_string = fileptr.GetProjection()
                self.dtype = fileptr.GetRasterBand(1).DataType
                self.metadict = Raster.get_raster_metadict(raster_name)

            # if get_array is false
            else:
                # get band names
                names = list()
                for i in range(0, bands):
                    names.append(fileptr.GetRasterBand(i + 1).GetDescription())

                # assign to empty class object without the array
                self.bnames = names
                self.shape = [bands, rows, cols]
                self.transform = fileptr.GetGeoTransform()
                self.crs_string = fileptr.GetProjection()
                self.dtype = fileptr.GetRasterBand(1).DataType
                self.metadict = Raster.get_raster_metadict(raster_name)

            # remap band names
            if use_dict is not None:
                self.bnames = [use_dict[sensor][b] for b in self.bnames]

        else:
            raise ValueError('No matching file found on disk')

    @property
    def chk_for_empty_tiles(self):
        """
        check the tile for empty bands, return true if one exists
        :return: bool
        """
        if Handler(self.name).file_exists():
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
                out_file_basename = Handler(in_file).basename.split('.')[0]

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
                            out_file_name = str(out_path) + Handler().sep + str(out_file_basename) + \
                                            "_" + str(i + 1) + "_" + str(j + 1) + ".tif"

                            # check if file already exists
                            out_file_name = Handler(filename=out_file_name).file_remove_check()

                            # get/calculate spatial parameters
                            new_ul = [ulx + i * px, uly + j * py]
                            new_lr = [new_ul[0] + px * tile_size_x, new_ul[1] + py * tile_size_y]
                            new_transform = (new_ul[0], px, rotx, new_ul[1], roty, py)

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
                            out_file_ptr.SetProjection(crs_string)

                            # delete pointers
                            out_file_ptr.FlushCache()  # save to disk
                            out_file_ptr = None
                            driver = None

                            # check for empty tiles
                            out_raster = Raster(out_file_name)
                            if out_raster.chk_for_empty_tiles:
                                print('Removing empty raster file: ' + Handler(out_file_name).basename)
                                Handler(out_file_name).file_delete()
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
    def get_raster_metadict(file_name=None,
                            file_ptr=None):
        """
        Function to get all the spatial metadata associated with a geotiff raster
        :param file_name: Name of the raster file (includes full path)
        :param file_ptr: Gdal file pointer
        :return: Dictionary of raster metadata
        """
        if file_name is not None:
            if Handler(file_name).file_exists():
                # open raster
                img_pointer = gdal.Open(file_name)
            else:
                raise ValueError("File does not exist.")

        elif file_ptr is not None:
            img_pointer = file_ptr

        else:
            raise ValueError("File or pointer not found")

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
                     'name': Handler(file_name).basename}  # file basename

        # remove pointer
        img_pointer = None

        return meta_dict

    def change_type(self,
                    out_type='int16'):

        """
        Method to change the raster data type
        :param out_type: Out data type. Options: int, int8, int16, int32, int64,
                                                float, float, float32, float64,
                                                uint, uint8, uint16, uint32, etc.
        :return: None
        """
        if gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(out_type)) != self.dtype:

            self.array = self.array.astype(out_type)
            self.dtype = gdal_array.NumericTypeCodeToGDALTypeCode(self.array.dtype)

            if self.nodatavalue is not None:
                self.nodatavalue = np.array(self.nodatavalue).astype(out_type).item()

            print('Changed raster data type to {}\n'.format(out_type))
        else:
            print('Raster data type already {}\n'.format(out_type))

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

    @staticmethod
    def get_coords(xy_list,
                   pixel_size=None,
                   tie_point=None,
                   pixel_center=True):

        """
        Method to convert pixel coord to image coords
        :param xy_list: List of tuples [(x1,y1), (x2,y2)....]
        :param pixel_size: tuple of x and y pixel size
        :param tie_point: tuple of x an y coordinates of tie point for the xy list
        :param pixel_center: If the center of the pixels should be returned instead of the top corners (default: True)
        :return: List of coordinates in tie point coordinate system
        """

        if type(xy_list).__name__ != 'list':
            xy_list = list(xy_list)

        if pixel_center:
            add_const = (float(pixel_size[0])/2.0, float(pixel_size[1])/2.0)
        else:
            add_const = (0.0, 0.0)

        coord_list = list()
        for xy in xy_list:

            xcoord = float(xy[0]) * float(pixel_size[0]) + tie_point[0] + add_const[0]
            ycoord = float(xy[1]) * float(pixel_size[1]) + tie_point[1] + add_const[1]

            coord_list.append([xcoord, ycoord])

        return coord_list

    def get_bounds(self):
        """
        Method to return a list of raster coordinates
        :return: List of lists
        """
        tie_pt = [self.transform[0], self.transform[3]]
        rast_coords = [tie_pt,
                       [tie_pt[0] + self.metadict['xpixel'] * self.shape[2], tie_pt[1]],
                       [tie_pt[0] + self.metadict['xpixel'] * self.shape[2],
                        tie_pt[1] - self.metadict['ypixel'] * self.shape[1]],
                       [tie_pt[0], tie_pt[1] - self.metadict['ypixel'] * self.shape[1]],
                       tie_pt]
        return rast_coords

    def make_tile_grid(self,
                       tile_xsize=1024,
                       tile_ysize=1024):
        """
        Returns the coordinates of the blocks to be extracted
        :param tile_xsize: Number of columns in the tile block
        :param tile_ysize: Number of rows in the tile block
        :return: list of lists
        """
        if not self.init:
            self.initialize()

        for y in xrange(0, self.shape[1], tile_ysize):
            if y + tile_ysize < self.shape[1]:
                rows = tile_ysize
            else:
                rows = self.shape[1] - y
            for x in xrange(0, self.shape[2], tile_xsize):
                if x + tile_xsize < self.shape[2]:
                    cols = tile_xsize
                else:
                    cols = self.shape[2] - x

                self.tile_grid.append({'block_coords': (x, y, cols, rows),
                                       'tie_point': self.get_coords([(x, y)],
                                                                    pixel_size=(self.transform[1], self.transform[5]),
                                                                    tie_point=(self.transform[0], self.transform[3]),
                                                                    pixel_center=False)[0]})

        self.ntiles = len(self.tile_grid)

    def get_next_tile(self,
                      tile_xsize=1024,
                      tile_ysize=1024,
                      band=1):

        """
        Generator to extract raster tile by tile
        :param tile_xsize: Number of columns in the tile block
        :param tile_ysize: Number of rows in the tile block
        :param band: Band to extract (default: 1)
        :return: Yields tuple: (tiepoint xy tuple, tile numpy array)
        """

        if not self.init:
            self.initialize()

        if self.ntiles is None:
            self.make_tile_grid(tile_xsize,
                                tile_ysize)

        temp_band = self.datasource.GetRasterBand(band)

        i = 0
        while i < self.ntiles:

            tile_arr = temp_band.ReadAsArray(*self.tile_grid[i]['block_coords'])

            yield self.tile_grid[i]['tie_point'], tile_arr

            i += 1

