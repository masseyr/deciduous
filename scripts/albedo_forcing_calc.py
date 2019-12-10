from modules import *
from sys import argv
import numpy as np


if __name__ == '__main__':

    script, infile, outdir, picklefile, band_name = argv

    outfile = outdir + Handler(infile).basename

    Handler(outfile).file_remove_check()

    # raster contains three bands: 1) decid 2) tree cover 3) land extent mask.
    # all the bands are in integer format
    raster = Raster(infile)
    raster.initialize()
    raster.bnames = ['decid', 'treecover']

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

    out_raster.write_to_file(compress='lzw', bigtiff='yes')

    uncert_dir = outdir[:-1] + '_uncert/'

    Handler(dirname=uncert_dir).dir_create()

    outfile = uncert_dir + Handler(infile).basename

    Handler(outfile).file_remove_check()

    out_raster = RFRegressor.regress_raster(regressor,
                                            raster,
                                            output_type='sd',
                                            mask_band='land',
                                            outfile=outfile,
                                            band_name=band_name,
                                            array_multiplier=0.01,
                                            internal_tile_size=512000,
                                            nodatavalue=0,
                                            verbose=True)

    Opt.cprint(out_raster)

    out_raster.write_to_file(compress='lzw', bigtiff='yes')
