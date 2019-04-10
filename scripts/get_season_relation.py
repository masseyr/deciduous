from modules import *


if __name__ == '__main__':

    file_dir = ''
    out_dir = ''

    filelist = Handler(dirname=file_dir).find_all(pattern='.tif')

    band_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

    for filename in filelist:
        raster_name = Handler(filename).basename
        raster = Raster(filename)
        raster.initialize(get_array=False)








