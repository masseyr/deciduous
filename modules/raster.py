import numpy as np
from common import *
import warnings
from osgeo import gdal, gdal_array, ogr, osr, gdalconst
np.set_printoptions(suppress=True)

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


__all__ = ['Raster', 'MultiRaster']


class Raster(object):
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
        self.array_offsets = None  # (px, py, xoff, yoff)
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
        self.bounds = None
        self.init = False
        self.stats = dict()

    def __repr__(self):

        if self.shape is not None:
            return "<raster {ras} of size {bands}x{rows}x{cols} ".format(ras=Handler(self.name).basename,
                                                                         bands=self.shape[0],
                                                                         rows=self.shape[1],
                                                                         cols=self.shape[2]) + \
                "datatype {dt} 'no-data' value {nd}>".format(dt=str(self.dtype),
                                                             nd=str(self.nodatavalue))
        else:
            return "<raster with path {ras}>".format(ras=self.name,)

    def write_to_file(self,
                      outfile=None,
                      driver='GTiff',
                      add_overview=False,
                      resampling='nearest',
                      overviews=None,
                      verbose=False,
                      **kwargs):
        """
        Write raster to file, given all the properties
        :param self: Raster object
        :param driver: raster driver
        :param outfile: Name of output file
        :param add_overview: If an external overview should be added to the file (useful for display)
        :param resampling: resampling type for overview (nearest, cubic, average, mode, etc.)
        :param overviews: list of overviews to compute( default: [2, 4, 8, 16, 32, 64, 128, 256])
        :param verbose: If the steps should be displayed
        :param kwargs: keyword arguments for creation options
        """
        creation_options = []
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                creation_options.append('{}={}'.format(key.upper(),
                                                       value.upper()))
        if outfile is None:

            if driver == 'MEM':
                outfile = 'tmp'
            else:
                outfile = self.name
                outfile = Handler(filename=outfile).file_remove_check()

        if verbose:
            Opt.cprint('\nWriting {}\n'.format(outfile))

        gtiffdriver = gdal.GetDriverByName(driver)
        fileptr = gtiffdriver.Create(outfile, self.shape[2], self.shape[1],
                                     self.shape[0], self.dtype, creation_options)
        nbands = self.shape[0]
        fileptr.SetGeoTransform(self.transform)
        fileptr.SetProjection(self.crs_string)

        if len(self.bnames) > 0:
            for i, bname in enumerate(self.bnames):
                if len(bname) == 0:
                    self.bnames[i] = 'band_{}'.format(str(i + 1))
        else:
            for i in range(self.shape[0]):
                self.bnames[i] = 'band_{}'.format(str(i + 1))

        if self.array is None:
            self.read_array()

        if nbands == 1:
            fileptr.GetRasterBand(1).WriteArray(self.array, 0, 0)
            fileptr.GetRasterBand(1).SetDescription(self.bnames[0])

            if self.nodatavalue is not None:
                fileptr.GetRasterBand(1).SetNoDataValue(self.nodatavalue)
            if verbose:
                Opt.cprint('Writing band: ' + self.bnames[0])
        else:
            for i in range(0, nbands):
                fileptr.GetRasterBand(i + 1).WriteArray(self.array[i, :, :], 0, 0)
                fileptr.GetRasterBand(i + 1).SetDescription(self.bnames[i])

                if self.nodatavalue is not None:
                    fileptr.GetRasterBand(i + 1).SetNoDataValue(self.nodatavalue)
                if verbose:
                    Opt.cprint('Writing band: ' + self.bnames[i])

        fileptr.FlushCache()
        fileptr = None

        if verbose:
            Opt.cprint('File written to disk!')

        if add_overview:
            if verbose:
                Opt.cprint('\nWriting overview')

            self.add_overviews(resampling,
                               overviews,
                               **kwargs)

            if verbose:
                Opt.cprint('Overview written to disk!')

    def add_overviews(self,
                      resampling='nearest',
                      overviews=None,
                      **kwargs):
        """
        Method to create raster overviews
        :param resampling:
        :param overviews:
        :param kwargs:
        :return:
        """

        fileptr = gdal.Open(self.name, 0)

        if overviews is None:
            overviews = [2, 4, 8, 16, 32, 64, 128, 256]

        if type(overviews) not in (list, tuple):
            if type(overviews) in (str, float):
                try:
                    overviews = [int(overviews)]
                except Exception as e:
                    raise ValueError(e.message)
            elif type(overviews) == int:
                overviews = [overviews]
            else:
                raise ValueError('Unsupported data type for overviews list')
        else:
            if any(list(type(elem) != int for elem in overviews)):
                overviews_ = list()
                for elem in overviews:
                    try:
                        overviews_.append(int(elem))
                    except Exception as e:
                        Opt.cprint('Conversion error: {} -for- {}'.format(e.message, elem))

                overviews = overviews_

        for k, v in kwargs.items():
            gdal.SetConfigOption('{}_OVERVIEW'.format(k.upper()), v.upper())

        fileptr.BuildOverviews(resampling.upper(), overviews)
        fileptr = None

    def read_array(self,
                   offsets=None,
                   band_order=None):
        """
        read raster array with offsets
        :param offsets: tuple or list - (xoffset, yoffset, xcount, ycount)
        :param band_order: order of bands to read
        """

        if not self.init:
            self.initialize()

        fileptr = self.datasource

        nbands, nrows, ncols = self.shape

        if offsets is None:
            self.array_offsets = (0, 0, ncols, nrows)
        else:
            self.array_offsets = offsets

        array3d = np.zeros((nbands,
                            self.array_offsets[3],
                            self.array_offsets[2]),
                           gdal_array.GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).DataType))

        # read array and store the band values and name in array
        if band_order is not None:
            for b in band_order:
                self.bnames.append(self.datasource.GetRasterBand(b + 1).GetDescription())
        else:
            band_order = list(range(nbands))

        # read array and store the band values and name in array
        for i, b in enumerate(band_order):
            if self.array_offsets is None:
                array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray()
            else:
                array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray(*self.array_offsets,
                                                                            resample_alg=gdalconst.GRA_NearestNeighbour)

        if (self.shape[0] == 1) and (len(array3d.shape) > 2):
            self.array = array3d.reshape([self.array_offsets[3],
                                          self.array_offsets[2]])
        else:
            self.array = array3d

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
        :return None
        """
        self.init = True
        raster_name = self.name

        if Handler(raster_name).file_exists() or 'vsimem' in self.name:
            fileptr = gdal.Open(raster_name)  # open file
            self.datasource = fileptr
            self.metadict = Raster.get_raster_metadict(file_name=raster_name)

        elif self.datasource is not None:
            fileptr = self.datasource
            self.metadict = Raster.get_raster_metadict(file_ptr=fileptr)

        else:
            raise ValueError('No datasource found')

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
                        Opt.cprint("Non-finite values replaced with " + str(nan_replacement))
                    else:
                        Opt.cprint("Non-finite values absent in file")

                # get band names
                for i in range(0, bands):
                    names.append(fileptr.GetRasterBand(i + 1).GetDescription())

            # band order present
            else:
                Opt.cprint('Reading bands: {}'.format(" ".join([str(b) for b in band_order])))

                bands = len(band_order)

                # bands in array
                n_array_bands = len(band_order)

                # initialize array
                if self.array_offsets is None:
                    array3d = np.zeros((n_array_bands,
                                        rows,
                                        cols),
                                       gdal_array.GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).DataType))
                else:
                    array3d = np.zeros((n_array_bands,
                                        self.array_offsets[3],
                                        self.array_offsets[2]),
                                       gdal_array.GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).DataType))

                # read array and store the band values and name in array
                for i, b in enumerate(band_order):
                    bandname = fileptr.GetRasterBand(b + 1).GetDescription()
                    Opt.cprint('Reading band {}'.format(bandname))

                    if self.array_offsets is None:
                        array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray()
                    else:
                        array3d[i, :, :] = fileptr.GetRasterBand(b + 1).ReadAsArray(*self.array_offsets)

                    names.append(bandname)

                # if flag for finite values is present
                if finite_only:
                    if np.isnan(array3d).any() or np.isinf(array3d).any():
                        array3d[np.isnan(array3d)] = nan_replacement
                        array3d[np.isinf(array3d)] = nan_replacement
                        Opt.cprint("Non-finite values replaced with " + str(nan_replacement))
                    else:
                        Opt.cprint("Non-finite values absent in file")

            # assign to empty class object
            self.array = array3d
            self.bnames = names
            self.shape = [bands, rows, cols]
            self.transform = fileptr.GetGeoTransform()
            self.crs_string = fileptr.GetProjection()
            self.dtype = fileptr.GetRasterBand(1).DataType

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
            self.nodatavalue = fileptr.GetRasterBand(1).GetNoDataValue()

        self.bounds = self.get_bounds()

        # remap band names
        if use_dict is not None:
            self.bnames = [use_dict[sensor][b] for b in self.bnames]

    def set_nodataval(self,
                      in_nodataval=255,
                      out_nodataval=0,
                      outfile=None,
                      in_array=True,
                      **kwargs):
        """
        replace no data value in raster and write to tiff file
        :param in_nodataval: no data value in input raster
        :param out_nodataval: no data value in output raster
        :param in_array: if the no data value should be changed in raster array
        :param outfile: output file name
        """
        if in_array:
            if not self.init:
                self.initialize(get_array=True,
                                **kwargs)
            self.array[np.where(self.array == in_nodataval)] = out_nodataval

        self.nodatavalue = out_nodataval

        if outfile is not None:
            self.write_to_file(outfile)

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
                   pixel_size,
                   tie_point,
                   pixel_center=True):

        """
        Method to convert pixel locations to image coords
        :param xy_list: List of tuples [(x1,y1), (x2,y2)....]
        :param pixel_size: tuple of x and y pixel size
        :param tie_point: tuple of x an y coordinates of tie point for the xy list
        :param pixel_center: If the center of the pixels should be returned instead of the top corners (default: True)
        :return: List of coordinates in tie point coordinate system
        """

        if type(xy_list) != list:
            xy_list = [xy_list]

        if pixel_center:
            add_const = (float(pixel_size[0])/2.0, float(pixel_size[1])/2.0)
        else:
            add_const = (0.0, 0.0)

        return list((float(xy[0]) * float(pixel_size[0]) + tie_point[0] + add_const[0],
                     float(xy[1]) * float(pixel_size[1]) + tie_point[1] + add_const[1])
                    for xy in xy_list)

    @staticmethod
    def get_locations(coords_list,
                      pixel_size,
                      tie_point):
        """
        Method to convert global coordinates to image pixel locations
        :param coords_list: Lit of coordinates in image CRS [(x1,y1), (x2,y2)....]
        :param pixel_size: Pixel size
        :param tie_point: Tie point of the raster or tile
        :return: list of pixel locations
        """
        if type(coords_list) != list:
            coords_list = [coords_list]

        return list(((coord[0] - tie_point[0])//pixel_size[0],
                     (coord[1] - tie_point[1])//pixel_size[1])
                    if coord is not None else [None, None]
                    for coord in coords_list)

    def get_bounds(self,
                   xy_coordinates=True):
        """
        Method to return a list of raster coordinates
        :param xy_coordinates: return a list of xy coordinates if true, else return [xmin, xmax, ymin, ymax]
        :return: List of lists
        """
        if not self.init:
            self.initialize()
        tie_pt = [self.transform[0], self.transform[3]]

        if xy_coordinates:
            return [tie_pt,
                    [tie_pt[0] + self.metadict['xpixel'] * self.shape[2], tie_pt[1]],
                    [tie_pt[0] + self.metadict['xpixel'] * self.shape[2],
                     tie_pt[1] - self.metadict['ypixel'] * self.shape[1]],
                    [tie_pt[0], tie_pt[1] - self.metadict['ypixel'] * self.shape[1]],
                    tie_pt]
        else:
            return [tie_pt[0], tie_pt[0] + self.metadict['xpixel'] * self.shape[2],
                    tie_pt[1] - self.metadict['ypixel'] * self.shape[1], tie_pt[1]]

    def get_pixel_bounds(self,
                         bound_coords=None,
                         coords_type='pixel'):
        """
        Method to return image bounds in the format xmin, xmax, ymin, ymax
        :param bound_coords: (xmin, xmax, ymin, ymax)
        :param coords_type: type of coordinates specified in bound_coords: 'pixel' for pixel coordinates,
                                                                           'crs' for image reference system coordinates
        :return: tuple: (xmin, xmax, ymin, ymax) in pixel coordinates
        """
        if not self.init:
            self.initialize()

        if bound_coords is not None:
            if coords_type == 'pixel':
                xmin, xmax, ymin, ymax = bound_coords
            elif coords_type == 'crs':
                _xmin, _xmax, _ymin, _ymax = bound_coords
                coords_list = [(_xmin, _ymax), (_xmax, _ymax), (_xmax, _ymin), (_xmin, _ymin)]
                coords_locations = np.array(self.get_locations(coords_list,
                                                               (self.transform[1], self.transform[5]),
                                                               (self.transform[0], self.transform[3])))
                xmin, xmax, ymin, ymax = \
                    int(coords_locations[:, 0].min()), \
                    int(coords_locations[:, 0].max()), \
                    int(coords_locations[:, 1].min()), \
                    int(coords_locations[:, 1].max())
            else:
                raise ValueError("Unknown coordinate types")

            if xmin < 0:
                xmin = 0
            if xmax > self.shape[2]:
                xmax = self.shape[2]
            if ymin < 0:
                ymin = 0
            if ymax > self.shape[1]:
                ymax = self.shape[1]

            if xmin >= xmax:
                raise ValueError("Image x-size should be greater than 0")
            if ymin >= ymax:
                raise ValueError("Image y-size should be greater than 0")
        else:
            xmin, xmax, ymin, ymax = 0, self.shape[2], 0, self.shape[1]

        return xmin, xmax, ymin, ymax

    def make_tile_grid(self,
                       tile_xsize=1024,
                       tile_ysize=1024,
                       bound_coords=None,
                       coords_type='pixel',
                       tile_buffer=None):
        """
        Returns the coordinates of the blocks to be extracted
        :param tile_xsize: Number of columns in the tile block
        :param tile_ysize: Number of rows in the tile block
        :param bound_coords: (xmin, xmax, ymin, ymax)
        :param coords_type: type of coordinates specified in bound_coords: 'pixel' for pixel coordinates,
                                                                           'crs' for image reference system coordinates
        :param tile_buffer: Buffer outside the tile boundary in image projection units
        :return: list of lists
        """
        if not self.init:
            self.initialize()

        # convert to the number of pixels in the buffer region
        if buffer is not None:
            buf_size_x = np.ceil(float(tile_buffer)/abs(float(self.transform[1])))
            buf_size_y = np.ceil(float(tile_buffer)/abs(float(self.transform[5])))
        else:
            buf_size_x = buf_size_y = None

        xmin, xmax, ymin, ymax = self.get_pixel_bounds(bound_coords,
                                                       coords_type)

        for y in range(ymin, ymax, tile_ysize):

            if y + tile_ysize < ymax:
                rows = tile_ysize
            else:
                rows = ymax - y

            for x in range(xmin, xmax, tile_xsize):
                if x + tile_xsize < xmax:
                    cols = tile_xsize
                else:
                    cols = xmax - x

                tie_pt = self.get_coords([(x, y)],
                                         (self.transform[1], self.transform[5]),
                                         (self.transform[0], self.transform[3]),
                                         pixel_center=False)[0]

                bounds = [tie_pt,
                          [tie_pt[0] + self.transform[1] * cols, tie_pt[1]],
                          [tie_pt[0] + self.transform[1] * cols, tie_pt[1] + self.transform[5] * rows],
                          [tie_pt[0], tie_pt[1] + self.transform[5] * rows],
                          tie_pt]

                self.tile_grid.append({'block_coords': (x, y, cols, rows),
                                       'tie_point': tie_pt,
                                       'bound_coords': bounds,
                                       'first_pixel': (xmin, ymin)})

        self.ntiles = len(self.tile_grid)

    def get_tile(self,
                 bands=None,
                 block_coords=None,
                 finite_only=True,
                 nan_replacement=None):
        """
        Method to get raster numpy array of a tile
        :param bands: bands to get in the array, index starts from one. (default: all)
        :param finite_only:  If only finite values should be returned
        :param nan_replacement: replacement for NAN values
        :param block_coords: coordinates of tile to retrieve in image coords (x, y, cols, rows)
        :return: numpy array
        """

        if not self.init:
            self.initialize()

        if nan_replacement is None:
            if self.nodatavalue is None:
                nan_replacement = 0
            else:
                nan_replacement = self.nodatavalue

        if bands is None:
            bands = list(range(1, self.shape[0] + 1))

        if len(bands) == 1:
            temp_band = self.datasource.GetRasterBand(bands[0])
            tile_arr = temp_band.ReadAsArray(*block_coords)

        else:
            tile_arr = np.zeros((len(bands),
                                 block_coords[3],
                                 block_coords[2]),
                                gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))

            for jj, band in enumerate(bands):
                temp_band = self.datasource.GetRasterBand(band)
                tile_arr[jj, :, :] = temp_band.ReadAsArray(*block_coords)

            if finite_only:
                if np.isnan(tile_arr).any() or np.isinf(tile_arr).any():
                    tile_arr[np.isnan(tile_arr)] = nan_replacement
                    tile_arr[np.isinf(tile_arr)] = nan_replacement

        return tile_arr

    def get_next_tile(self,
                      tile_xsize=1024,
                      tile_ysize=1024,
                      bands=None,
                      get_array=True,
                      finite_only=True,
                      nan_replacement=None):

        """
        Generator to extract raster tile by tile
        :param tile_xsize: Number of columns in the tile block
        :param tile_ysize: Number of rows in the tile block
        :param bands: List of bands to extract (default: None, gets all bands; Index starts at 0)
        :param get_array: If raster array should be retrieved as well
        :param finite_only: If only finite values should be returned
        :param nan_replacement: replacement for NAN values
        :return: Yields tuple: (tiepoint xy tuple, tile numpy array(2d array if only one band, else 3d array)
        """

        if not self.init:
            self.initialize()

        if self.ntiles is None:
            self.make_tile_grid(tile_xsize,
                                tile_ysize)
        if nan_replacement is None:
            if self.nodatavalue is None:
                nan_replacement = 0
            else:
                nan_replacement = self.nodatavalue

        if bands is None:
            bands = range(1, int(self.shape[0]) + 1)
        elif type(bands) in (int, float):
            bands = [int(bands)]
        elif type(bands) in (list, tuple):
            bands = [int(ib + 1) for ib in bands]
        else:
            raise ValueError('Unknown/unsupported data type for "bands" keyword')

        ii = 0
        while ii < self.ntiles:
            if get_array:

                if len(bands) == 1:
                    temp_band = self.datasource.GetRasterBand(bands[0])
                    tile_arr = temp_band.ReadAsArray(*self.tile_grid[ii]['block_coords'])

                else:
                    tile_arr = np.zeros((len(bands),
                                         self.tile_grid[ii]['block_coords'][3],
                                         self.tile_grid[ii]['block_coords'][2]),
                                        gdal_array.GDALTypeCodeToNumericTypeCode(self.dtype))

                    for jj, band in enumerate(bands):
                        temp_band = self.datasource.GetRasterBand(band)
                        tile_arr[jj, :, :] = temp_band.ReadAsArray(*self.tile_grid[ii]['block_coords'])

                if finite_only:
                    if np.isnan(tile_arr).any() or np.isinf(tile_arr).any():
                        Opt.cprint('Non-finite values present in tile')

                        tile_arr[np.where(np.isnan(tile_arr))] = nan_replacement
                        tile_arr[np.where(np.isinf(tile_arr))] = nan_replacement

            else:
                tile_arr = None

            yield self.tile_grid[ii]['tie_point'], tile_arr

            ii += 1

    def extract_geom(self,
                     wkt_strings,
                     geom_id=None,
                     band_order=None,
                     **kwargs):
        """
        Extract all pixels that intersect a feature in a Raster.
        The raster object should be initialized before using this method.
        Currently this method only supports single geometries per query.
        :param wkt_strings: List or Tuple of Vector geometries (e.g. point) in WKT string format
                           this geometry should be in the same CRS as the raster
                           Currently only 'Point' or 'MultiPoint' geometry is supported.
                           Accepted wkt_strings: List of POINT or MULTIPOINT wkt(s)
        :param geom_id: List of geometry IDs
                        If for a MultiGeom only one ID is presented, it will be suffixed with a number
        :param band_order: Order of bands to be extracted (list)

        :param kwargs: List of additional arguments
                        tile_size : (256, 256) default

        :return: List of pixel band values as tuples for each pixel : [ (ID, [band vales] ), ]
        """

        # define tile size
        if 'tile_size' in kwargs:
            tile_size = kwargs['tile_size']
        else:
            tile_size = (self.shape[1], self.shape[2])

        # define band order
        if band_order is None:
            band_order = np.array(range(0, self.shape[0]))
        else:
            band_order = np.array(band_order)

        # initialize raster
        if not self.init or self.array is None:
            self.initialize()

        if type(wkt_strings) not in (list, tuple):
            wkt_strings = [wkt_strings]

        id_geom_list = list()
        if geom_id is None:
            geom_id = range(1, len(wkt_strings) + 1)

        for ii, wkt_string in enumerate(wkt_strings):
            if 'MULTI' in wkt_string:
                multi_geom = ogr.CreateGeometryFromWkt(wkt_strings)
                id_geom_list += list(('{}_{}'.format(geom_id[ii], str(jj+1)), multi_geom.GetGeometryRef(jj))
                                     for jj in range(multi_geom.GetGeometryCount()))
            else:
                id_geom_list.append(('{}_{}'.format(geom_id[ii], str(1)), ogr.CreateGeometryFromWkt(wkt_string)))

        self.make_tile_grid(*tile_size)

        tile_samp_output = list([] for _ in range(len(id_geom_list)))

        for tile in self.tile_grid:
            tile_wkt = 'POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                             for (x, y) in tile['bound_coords'])))
            tile_geom = ogr.CreateGeometryFromWkt(tile_wkt)

            samp_ids = list(ii for ii, elem in enumerate(id_geom_list) if tile_geom.Intersects(elem[1]))

            if len(samp_ids) > 0:
                self.read_array(tile['block_coords'])

                samp_coords = list(list(float(elem)
                                        for elem in id_geom_list[ii][1].ExportToWkt()
                                        .replace('POINT', '')
                                        .replace('(', '')
                                        .replace(')', '')
                                        .strip()
                                        .split(' '))
                                   for ii in samp_ids)

                if self.shape[0] == 1:
                    samp_values = list([self.array[int(y), int(x)]]
                                       for x, y in self.get_locations(samp_coords,
                                                                      (self.transform[1],
                                                                       self.transform[5]),
                                                                      tile['tie_point']))
                else:
                    samp_values = list(self.array[band_order, int(y), int(x)].tolist()
                                       for x, y in self.get_locations(samp_coords,
                                                                      (self.transform[1],
                                                                       self.transform[5]),
                                                                      tile['tie_point']))

                for ii, samp_id in enumerate(samp_ids):
                    tile_samp_output[samp_id] = (id_geom_list[samp_id][0], samp_values[ii])

        return tile_samp_output

    def get_stats(self,
                  print_stats=False,
                  approx=False):

        """
        Method to compute statistics of the raster object, and store as raster property
        :param print_stats: If the statistics should be printed to console
        :param approx: If approx statistics should be calculated instead to gain speed
        :return: None
        """

        for ib in range(self.shape[0]):
            band = self.datasource.GetRasterBand(ib+1)
            band.ComputeStatistics(approx)
            band_stats = dict(zip(['min', 'max', 'mean', 'stddev'], band.GetStatistics(int(approx), 0)))

            if print_stats:
                Opt.cprint('Band {} : {}'.format(self.bnames[ib],
                                                 str(band_stats)))

            self.stats[self.bnames[ib]] = band_stats

    def clip(self,
             cutline_file=None,
             cutline_blend=0,
             outfile=None,
             return_vrt=False,
             return_vrt_dict=False,
             **creation_options):
        """
        Method to clip a raster to a given geometry/vector
        This method only supports clipping to the first layer in the datasource
        :param cutline_file: Shapefile, etc. to clip raster
        :param cutline_blend: blend distance in pixels
        :param outfile: Output filename
        :param return_vrt: If a VRT object should be returned instead of raster
        :param return_vrt_dict: if a VRT options dictionary should be returned instead
        :param creation_options: Other creation options to input
        :return: Raster object

        valid warp options can be found at:
        https://gdal.org/python/osgeo.gdal-module.html#WarpOptions:
        """
        cutline_ds = ogr.Open(cutline_file)
        layer_count = cutline_ds.GetLayerCount()
        cutline_layer = cutline_ds.GetLayer(0)
        cutline_layer_name = cutline_layer.GetDescription()

        if layer_count > 1:
            warnings.warn('Using top layer {} as cutline, ignoring others'.format(cutline_layer_name))

        cutline_ds = cutline_layer = None

        vrt_dict = dict()

        vrt_dict['cutlineDSName'] = cutline_file
        vrt_dict['cutlineLayer'] = cutline_layer_name
        vrt_dict['cutlineBlend'] = cutline_blend
        vrt_dict['cropToCutline'] = True
        vrt_dict['copyMetadata'] = True

        creation_options_list = []
        if len(creation_options) > 0:
            for key, value in creation_options.items():
                creation_options_list.append('{}={}'.format(key.upper(),
                                                            value.upper()))
            vrt_dict['creationOptions'] = creation_options_list

        vrt_opt = gdal.WarpOptions(**vrt_dict)

        if return_vrt_dict:
            return vrt_dict
        else:
            if outfile is None:
                outfile = Handler(self.name).add_to_filename('_clipped')

            vrt_ds = gdal.Warp(outfile, self.name, options=vrt_opt)

            if return_vrt:
                return vrt_ds
            else:
                vrt_ds = None
                return Raster(outfile)

    def reproject(self,
                  outfile=None,
                  out_epsg=None,
                  out_wkt=None,
                  out_proj4=None,
                  out_spref=None,
                  output_res=None,
                  out_datatype=gdal.GDT_Float32,
                  resampling=None,
                  output_bounds=None,
                  out_format='GTiff',
                  out_nodatavalue=None,
                  verbose=False,
                  cutline_file=None,
                  cutline_blend=0,
                  return_vrt=False,
                  **creation_options):
        """
        Method to reproject raster object
        :param outfile: output file name
        :param out_epsg: EPSG code for output spatial reference
        :param out_wkt: WKT representation of output spatial reference
        :param out_proj4: PROJ4 representation of output spatial reference
        :param out_spref: Output spatial reference object
        :param output_res: output spatial resolution (xRes, yRes)
        :param out_datatype: output type (gdal.GDT_Byte, etc...)
        :param resampling: near, bilinear, cubic, cubicspline,
                           lanczos, average, mode, max, min, med, q1, q3
        :param out_nodatavalue: output no-data value to replace input no-data value
        :param output_bounds: output bounds as (minX, minY, maxX, maxY) in target SRS
        :param out_format: output format ("GTiff", etc...)
        :param verbose: If the steps should be displayed
        :param cutline_file: Shapefile, etc. to clip raster
        :param cutline_blend: blend distance in pixels
        :param return_vrt: If VRT object should be returned instead of raster
        :param creation_options: Creation options to be used while writing the raster
                                (example for geotiff: 'compress=lzw' , 'bigtiff=yes' )
        :return: VRT object or None (if output file is also specified)

        All the other valid warp options can be found at
        https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
        """

        vrt_dict = dict()

        if output_bounds is not None:
            vrt_dict['outputBounds'] = output_bounds

        if output_res is not None:
            vrt_dict['xRes'] = output_res[0]
            vrt_dict['yRes'] = output_res[1]

        if out_nodatavalue is not None:
            vrt_dict['dstNodata'] = out_nodatavalue
        else:
            vrt_dict['dstNodata'] = self.nodatavalue

        vrt_dict['srcNodata'] = self.nodatavalue

        if resampling is not None:
            vrt_dict['resampleAlg'] = resampling
        else:
            vrt_dict['resampleAlg'] = 'near'

        if verbose:
            Opt.cprint('Outfile: {}'.format(outfile))

        if out_spref is not None:
            sp = out_spref
        else:
            sp = osr.SpatialReference()

            if out_epsg is not None:
                res = sp.ImportFromEPSG(out_epsg)
            elif out_wkt is not None:
                res = sp.ImportFromWkt(out_wkt)
            elif out_proj4 is not None:
                res = sp.ImportFromProj4(out_proj4)
            else:
                raise ValueError("Output Spatial reference not provided")

        vrt_dict['srcSRS'] = self.crs_string
        vrt_dict['dstSRS'] = sp.ExportToWkt()

        vrt_dict['outputType'] = out_datatype
        vrt_dict['copyMetadata'] = True
        vrt_dict['format'] = out_format

        if cutline_file is not None:
            clip_opts = self.clip(cutline_file,
                                  cutline_blend,
                                  return_vrt_dict=True)

            vrt_dict.update(clip_opts)

        creation_options_list = []
        if len(creation_options) > 0:
            for key, value in creation_options.items():
                creation_options_list.append('{}={}'.format(key.upper(),
                                             value.upper()))

            vrt_dict['creationOptions'] = creation_options_list

        vrt_opt = gdal.WarpOptions(**vrt_dict)

        if outfile is None:
            outfile = Handler(self.name).dirname + Handler().sep + '_reproject.tif'

        vrt_ds = gdal.Warp(outfile, self.name, options=vrt_opt)

        if return_vrt:
            return vrt_ds
        else:
            vrt_ds = None


class MultiRaster:
    """
    Virtual raster class to manipulate GDAL virtual raster object
    """

    def __init__(self,
                 filelist=None,
                 initialize=True,
                 get_array=False):

        """
        Class constructor
        :param filelist: List of raster (.tif) files
        :param initialize: if the rasters from file list should be initialized
        :param get_array: if raster arrays should be read to memory
        """

        self.filelist = filelist
        self.rasters = list()

        if filelist is not None:

            if type(filelist).__name__ not in ('list', 'tuple'):
                filelist = [filelist]
            for filename in filelist:
                ras = Raster(filename)
                if initialize:
                    ras.initialize(get_array=get_array)
                self.rasters.append(ras)

        self.intersection = None
        self.nodatavalue = list(raster.nodatavalue for raster in self.rasters)
        self.resolutions = list((raster.transform[1], raster.transform[5]) for raster in self.rasters)

    def get_intersection(self,
                         index=None,
                         _return=True):
        """
        Method to get intersecting bounds of the raster objects
        :param index: list of indices of raster files/objects
        :param _return: Should the method return the bound coordinates
        :return: coordinates of intersection (minx, miny, maxx, maxy)
        """

        wkt_list = list()
        if index is not None:
            for ii in index:
                bounds = self.rasters[ii].get_bounds()
                wktstring = 'POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                                  for (x, y) in bounds)))
                wkt_list.append(wktstring)
        else:
            for raster in self.rasters:
                bounds = raster.get_bounds()
                wktstring = 'POLYGON(({}))'.format(', '.join(list(' '.join([str(x), str(y)])
                                                                  for (x, y) in bounds)))
                wkt_list.append(wktstring)

        geoms = list(ogr.CreateGeometryFromWkt(wktstring) for wktstring in wkt_list)

        temp_geom = geoms[0]

        for geom in geoms[1:]:
            temp_geom = temp_geom.Intersection(geom)

        temp_geom = temp_geom.ExportToWkt()

        temp_coords = list(list(float(elem.strip()) for elem in elem_.strip()
                                                                     .split(' '))
                           for elem_ in temp_geom.replace('POLYGON', '')
                                                 .replace('((', '')
                                                 .replace('))', '')
                                                 .split(','))

        minx = min(list(coord[0] for coord in temp_coords))
        miny = min(list(coord[1] for coord in temp_coords))

        maxx = max(list(coord[0] for coord in temp_coords))
        maxy = max(list(coord[1] for coord in temp_coords))

        self.intersection = (minx, miny, maxx, maxy)

        if _return:
            return self.intersection

    def layerstack(self,
                   order=None,
                   verbose=False,
                   outfile=None,
                   return_vrt=True,
                   **kwargs):

        """
        Method to layerstack rasters in a given order
        :param order: order of raster layerstack
        :param verbose: If some of the steps should be printed to console
        :param outfile: Name of the output file (.tif)
        :param return_vrt: If the file should be written to disk or vrt object should be returned
        :return: None

        valid build vrt options in kwargs
        (from https://gdal.org/python/osgeo.gdal-module.html#BuildVRT):

          options --- can be be an array of strings, a string or let empty and filled from other keywords..
          resolution --- 'highest', 'lowest', 'average', 'user'.
          outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS.
          xRes --- output horizontal resolution in target SRS.
          yRes --- output vertical resolution in target SRS.
          targetAlignedPixels --- whether to force output bounds to be multiple of output resolution.
          separate --- whether each source file goes into a separate stacked band in the VRT band.
          bandList --- array of band numbers (index start at 1).
          addAlpha --- whether to add an alpha mask band to the VRT when the source raster have none.
          resampleAlg --- resampling mode.
          outputSRS --- assigned output SRS.
          allowProjectionDifference --- whether to accept input datasets have not the same projection.
           Note: they will *not* be reprojected.
          srcNodata --- source nodata value(s).
          VRTNodata --- nodata values at the VRT band level.
          hideNodata --- whether to make the VRT band not report the NoData value.
          callback --- callback method.
          callback_data --- user data for callback.

        valid translate options in kwargs
        (from https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions):

          options --- can be be an array of strings, a string or let empty and filled from other keywords.
          format --- output format ("GTiff", etc...)
          outputType --- output type (gdal.GDT_Byte, etc...)
          bandList --- array of band numbers (index start at 1)
          maskBand --- mask band to generate or not ("none", "auto", "mask", 1, ...)
          width --- width of the output raster in pixel
          height --- height of the output raster in pixel
          widthPct --- width of the output raster in percentage (100 = original width)
          heightPct --- height of the output raster in percentage (100 = original height)
          xRes --- output horizontal resolution
          yRes --- output vertical resolution
          creationOptions --- list of creation options
          srcWin --- subwindow in pixels to extract: [left_x, top_y, width, height]
          projWin --- subwindow in projected coordinates to extract: [ulx, uly, lrx, lry]
          projWinSRS --- SRS in which projWin is expressed
          strict --- strict mode
          unscale --- unscale values with scale and offset metadata
          scaleParams --- list of scale parameters, each of the form [src_min,src_max] or [src_min,src_max,dst_min,dst_max]
          exponents --- list of exponentiation parameters
          outputBounds --- assigned output bounds: [ulx, uly, lrx, lry]
          metadataOptions --- list of metadata options
          outputSRS --- assigned output SRS
          GCPs --- list of GCPs
          noData --- nodata value (or "none" to unset it)
          rgbExpand --- Color palette expansion mode: "gray", "rgb", "rgba"
          stats --- whether to calculate statistics
          rat --- whether to write source RAT
          resampleAlg --- resampling mode
          callback --- callback method
          callback_data --- user data for callback

        """

        if order is None:
            order = list(range(len(self.rasters)))

        vrt_dict = dict()

        if 'output_bounds' in kwargs:
            vrt_dict['outputBounds'] = kwargs['output_bounds']
        else:
            vrt_dict['outputBounds'] = self.intersection

        output_res = min(list(np.abs(self.resolutions[i][0]) for i in order))

        if 'outputresolution' in kwargs:
            vrt_dict['xRes'] = kwargs['outputresolution'][0]
            vrt_dict['yRes'] = kwargs['outputresolution'][1]
        else:
            vrt_dict['xRes'] = output_res
            vrt_dict['yRes'] = output_res

        if 'nodatavalue' in kwargs:
            vrt_dict['srcNodata'] = kwargs['nodatavalue']
        else:
            vrt_dict['srcNodata'] = self.nodatavalue[0]

        if 'outnodatavalue' in kwargs:
            vrt_dict['VRTNodata'] = kwargs['outnodatavalue']
        else:
            vrt_dict['VRTNodata'] = self.nodatavalue[0]

        if 'resample' in kwargs:
            vrt_dict['resampleAlg'] = kwargs['resample']
        else:
            vrt_dict['resampleAlg'] = 'cubic'

        if verbose:
            Opt.cprint('Getting bounds ...')

        vrt_dict['outputBounds'] = self.get_intersection(index=order)
        vrt_dict['separate'] = True
        vrt_dict['hideNodata'] = False

        if verbose:
            Opt.cprint('Files: \n{}'.format('\n'.join(list(self.filelist[i] for i in order))))

        _vrt_opt_ = gdal.BuildVRTOptions(**vrt_dict)

        if outfile is None:
            vrtfile = Handler(self.filelist[0]).dirname + Handler().sep + 'layerstack1.vrt'
            outfile = Handler(self.filelist[0]).dirname + Handler().sep + 'layerstack1.tif'
        else:
            vrtfile = outfile.split('.tif')[0] + '.vrt'

        _vrt_ = gdal.BuildVRT(vrtfile, list(self.filelist[i] for i in order), options=_vrt_opt_)

        if not return_vrt:
            if verbose:
                Opt.cprint('Writing layer stack file : {} ...'.format(outfile))

            gdal.Translate(outfile, _vrt_, **kwargs)
            _vrt_ = None

            if verbose:
                Opt.cprint('Done!')
        else:
            return _vrt_

    def composite(self,
                  layer_indices=None,
                  verbose=False,
                  outfile=None,
                  composite_type='mean',
                  tile_size=1024,
                  write_raster=False,
                  **kwargs):
        """
        Method to calculate raster composite in a given order
        :param layer_indices: list of layer indices
        :param verbose: If some of the steps should be printed to console
        :param outfile: Name of the output file (.tif)
        :param tile_size: Size of internal tile
        :param write_raster: If the raster file should be written or a raster object be returned
        :param composite_type: mean, median, pctl_xxx (eg: pctl_5, pctl_99, pctl_100, etc.),
        :return: None
        """

        if layer_indices is None:
            layer_indices = list(range(len(self.rasters)))  # list of layer indices
            t_order = list(range(1, len(self.rasters) + 1))  # list of bands to include in raster tiles
        else:
            t_order = list(elem + 1 for elem in layer_indices)  # list of bands to include in raster tiles

        # layer stack vrt
        _ls_vrt_ = self.layerstack(order=layer_indices,
                                   verbose=verbose,
                                   return_vrt=True,
                                   **kwargs)

        # raster object from vrt
        lras = Raster('tmp_layerstack')
        lras.datasource = _ls_vrt_
        lras.initialize()

        if 'bound_coords' in kwargs:
            if 'coords_type' in kwargs:
                lras.make_tile_grid(tile_size,
                                    tile_size,
                                    bound_coords=kwargs['bound_coords'],
                                    coords_type=kwargs['coords_type'])
            else:
                lras.make_tile_grid(tile_size,
                                    tile_size,
                                    bound_coords=kwargs['bound_coords'],
                                    coords_type='crs')
        else:
            lras.make_tile_grid(tile_size,
                                tile_size)

        Opt.cprint(lras)

        # make numpy array to hold the final result
        out_arr = np.zeros((lras.shape[1], lras.shape[2]),
                           dtype=gdal_array.GDALTypeCodeToNumericTypeCode(lras.dtype))

        # loop through raster tiles
        count = 0
        for tie_pt, tile_arr in lras.get_next_tile(bands=t_order):

            Opt.cprint(lras.tile_grid[count]['block_coords'])

            _x, _y, _cols, _rows = lras.tile_grid[count]['block_coords']

            if composite_type == 'mean':
                temp_arr = np.apply_along_axis(lambda x: np.mean(x[x != lras.nodatavalue]), 0, tile_arr)
            elif composite_type == 'median':
                temp_arr = np.apply_along_axis(lambda x: np.median(x[x != lras.nodatavalue]), 0, tile_arr)
            elif composite_type == 'min':
                temp_arr = np.apply_along_axis(lambda x: np.min(x[x != lras.nodatavalue]), 0, tile_arr)
            elif composite_type == 'max':
                temp_arr = np.apply_along_axis(lambda x: np.max(x[x != lras.nodatavalue]), 0, tile_arr)
            elif 'pctl' in composite_type:
                pctl = int(composite_type.split('_')[1])
                temp_arr = np.apply_along_axis(lambda x: np.percentile(x[x != lras.nodatavalue], pctl), 0, tile_arr)
            else:
                raise ValueError('Unknown composite option')

            # update output array with tile composite
            out_arr[_y: (_y+_rows), _x: (_x+_cols)] = temp_arr
            count += 1

        # write array to raster
        lras.array = out_arr

        if write_raster:
            # write raster
            lras.write_to_file(outfile)
            Opt.cprint('Written {}'.format(outfile))
        else:
            return lras

    def mosaic(self,
               order=None,
               verbose=False,
               outfile=None,
               nodata_values=None,
               band_index=None,
               blend_images=True,
               blend_pixels=10,
               blend_cutline=None,
               **kwargs):
        """
        Under construction

        Method to mosaic rasters in a given order
        :param order: order of raster layerstack
        :param verbose: If some of the steps should be printed to console
        :param outfile: Name of the output file (.tif)
        :param nodata_values: Value or tuple (or list) of values used as nodata bands for each image to be mosaicked
        :param band_index: list of bands to be used in mosaic (default: all)
        :param blend_images: If blending should be used in mosaicking (default True)
        :param blend_cutline: OSGEO SWIG geometry of cutline
        :param blend_pixels: width of pixels to blend around the cutline or
                             raster boundary for multiple rasters (default: 10)

        :return: None

        valid warp options in kwargs
        (from https://gdal.org/python/osgeo.gdal-module.html#WarpOptions):

          options --- can be be an array of strings, a string or let empty and filled from other keywords.
          format --- output format ("GTiff", etc...)
          outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS
          outputBoundsSRS --- SRS in which output bounds are expressed, in the case they are not expressed in dstSRS
          xRes, yRes --- output resolution in target SRS
          targetAlignedPixels --- whether to force output bounds to be multiple of output resolution
          width --- width of the output raster in pixel
          height --- height of the output raster in pixel
          srcSRS --- source SRS
          dstSRS --- output SRS
          srcAlpha --- whether to force the last band of the input dataset to be considered as an alpha band
          dstAlpha --- whether to force the creation of an output alpha band
          outputType --- output type (gdal.GDT_Byte, etc...)
          workingType --- working type (gdal.GDT_Byte, etc...)
          warpOptions --- list of warping options
          errorThreshold --- error threshold for approximation transformer (in pixels)
          warpMemoryLimit --- size of working buffer in bytes
          resampleAlg --- resampling mode
          creationOptions --- list of creation options
          srcNodata --- source nodata value(s)
          dstNodata --- output nodata value(s)
          multithread --- whether to multithread computation and I/O operations
          tps --- whether to use Thin Plate Spline GCP transformer
          rpc --- whether to use RPC transformer
          geoloc --- whether to use GeoLocation array transformer
          polynomialOrder --- order of polynomial GCP interpolation
          transformerOptions --- list of transformer options
          cutlineDSName --- cutline dataset name
          cutlineLayer --- cutline layer name
          cutlineWhere --- cutline WHERE clause
          cutlineSQL --- cutline SQL statement
          cutlineBlend --- cutline blend distance in pixels
          cropToCutline --- whether to use cutline extent for output bounds
          copyMetadata --- whether to copy source metadata
          metadataConflictValue --- metadata data conflict value
          setColorInterpretation --- whether to force color interpretation of input bands to output bands
          callback --- callback method
          callback_data --- user data for callback

        For valid translate options, see MultiRaster.layerstack()


        steps:
        1) vectorize each layer in blend_layers according to vectorize_values
        2) calculate buffer in 1 pixel step from vectorized shapes
        3) Only two images at a time, the one on top and the once below it, are weighted, and weighted function calculated
        4) calculate weighting function output using vrt via tiling


        if vectorize_values is None:
            vectorize_values = list(self.nodatavalue[i-1] for i in blend_bands)

        elif type(vectorize_values) not in (list, tuple):
            if type(vectorize_values) != np.ndarray:
                vectorize_values = list(vectorize_values for _ in blend_bands)
            else:
                raise ValueError('Values to vectorize should be one of: tuple, list, int, or float')

        vector_list = list()
        for ii, val in enumerate(vectorize_values):
            temp_vec =

        """














