from modules import *
from sys import argv
import numpy as np


if __name__ == '__main__':
    '''
    script, infile, outdir, picklefile, band_name = argv
    '''
    # ------------------------------------------------------------------------------------------
    infile = "C:/temp/decid_tc_2000_layerstack-0000026880-0000161280.tif"
    outdir = "C:/temp/"
    picklefile = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/" + \
                 "RFalbedo_deciduous_fraction_treecover_50000_cutoff_5_deg1_20191115T174618_spring.pickle"
    band_name = 'spr_albedo'
    # ------------------------------------------------------------------------------------------

    outfile = outdir + Handler(infile).basename
    Handler(outfile).file_remove_check()

    # infile contains five bands: 1) decid
    #                             2) decid uncertainty
    #                             3) tree cover
    #                             4) tree cover uncertainty
    #                             5) land extent mask.
    # All the bands are in integer format
    # The uncertainty bands are std dev values around the mean value
    # We use 10 random values in the range mean +/- stddev
    # to calculate the output range (uncertainty) around the mean forcing value.

    raster = Raster(infile)
    raster.initialize()
    raster.bnames = ['decid', 'decidu', 'treecover', 'treecoveru', 'land']

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
