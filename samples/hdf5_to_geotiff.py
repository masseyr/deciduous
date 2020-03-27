from osgeo import gdal, osr
import h5py
import numpy as np
from modules import Vector, Handler
import argparse
import sys


if __name__ == "__main__":

    # script, filename = sys.argv

    in_folder = "D:/temp/above2017_0629_lvis2b/l2b/"
    out_folder = "D:/temp/above2017_0629_lvis2b_tif/"

    spref = osr.SpatialReference()
    spref.ImportFromEPSG(4326)
    spref_wkt = spref.ExportToWkt()

    filelist = Handler(dirname=in_folder).find_all('*.h5')

    for hdf5_file in filelist:
        outfile = out_folder + Handler(hdf5_file).basename.replace('.h5', '.tif')

        print('\n===============================================================================================')

        print('Input file: {}'.format(hdf5_file))
        print('Output file: {}'.format(outfile))

        fs = h5py.File(hdf5_file, 'r')

        file_keys = []
        fs.visit(file_keys.append)

        lat_arr = np.array(fs['GLAT'])
        lon_arr = np.array(fs['GLON']) - 360.0

        lat_limits = [lat_arr.min(), lat_arr.max()]
        lon_limits = [lon_arr.min(), lon_arr.max()]

        pos_arr = np.vstack([lon_arr, lat_arr]).T
        cover_arr = np.array(fs['cover'])
        cover_err_arr = np.array(fs['cover_error'])

        xpixel = 0.00027
        ypixel = 0.00027

        x_extent = lon_limits[1] - lon_limits[0]
        y_extent = lat_limits[1] - lat_limits[0]

        xsize = np.ceil(x_extent/xpixel)
        ysize = np.ceil(y_extent/ypixel)

        print('x extent {} : y extent {}'.format(str(x_extent), str(y_extent)))
        print('Num of 30m pixels in x {} | Num of 30m pixels in y {}'.format(str(xsize), str(ysize)))

        wkt_list = list(Vector.wkt_from_coords(coords, geom_type='point') for coords in pos_arr.tolist())

        attrib = {'cover': 'float', 'cover_error': 'float'}

        attr_list = list({'cover': cover[0], 'cover_error': cover[1]} for cover in zip(cover_arr, cover_err_arr))

        vector = Vector.vector_from_string(wkt_list,
                                           geom_string_type='wkt',
                                           out_epsg=4326,
                                           vector_type='point',
                                           attributes=attr_list,
                                           attribute_types=attrib,
                                           verbose=False)

        print(vector)

        vector.rasterize(outfile=outfile,
                         pixel_size=[xpixel, ypixel],
                         out_dtype=gdal.GDT_Float32,
                         compress='lzw',
                         attribute='cover')






