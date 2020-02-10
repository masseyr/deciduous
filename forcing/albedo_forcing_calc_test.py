from modules import *
from sys import argv
import numpy as np
import os


if __name__ == '__main__':
    '''
    script, infile, outdir, picklefile, band_name = argv
    '''
    infile = "C:/temp/decid_tc_2000_layerstack-0000026880-0000161280.tif"
    outdir = "C:/temp/"
    picklefile = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/" + \
                 "RFalbedo_deciduous_fraction_treecover_50000_cutoff_5_deg1_20191115T174618_spring.pickle"
    band_name = 'spr_albedo'

    outfile = Handler(outdir + Handler(infile).basename).add_to_filename(band_name)

    if os.path.exists(outfile):
        os.remove(outfile)

    # raster contains three bands: 1) decid 2) tree cover 3) land extent mask.
    # all the bands are in integer format
    raster = Raster(infile)
    raster.initialize()
    raster.bnames = ['decid', 'treecover', 'land']

    raster.get_stats(True)

    Opt.cprint(raster.shape)

    regressor = RFRegressor.load_from_pickle(picklefile)

    Opt.cprint(regressor)

    out_raster = RFRegressor.regress_raster(regressor,
                                            raster,
                                            output_type='median',
                                            mask_band='land',
                                            outfile=outfile,
                                            band_name=band_name,
                                            array_multiplier=0.01,
                                            internal_tile_size=512000,
                                            nodatavalue=0,
                                            verbose=True)

    Opt.cprint(out_raster)

    out_raster.write_to_file(compress='lzw')
