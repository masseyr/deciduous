from modules import *
import numpy as np


if __name__ == '__main__':

    folder = 'C:/temp/tc/in/'
    files = Handler(dirname=folder).find_all(pattern='*.tif')

    outfolder = 'C:/temp/tc/out/'

    files = ["D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/treecover/" + \
        "ABoVE_median_SR_NDVI_boreal_2005_tc_prediction_vis_nd_250m.tif"]

    outfolder = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/treecover/"

    for file_ in files:
        print(file_)

        outfile = outfolder + Handler(file_).basename.split('.')[0] + '_full.tif'

        ras1 = Raster(file_)
        ras1.initialize(get_array=True)
        print(ras1.array)

        ras1.array[np.where(ras1.array == 255)] = 0

        print(ras1.array.min())
        print(ras1.array.max())

        print(ras1)

        ras1.nodatavalue = 0

        ras1.write_to_file(outfile)
