if __name__ == '__main__':
    import sys
    import os

    module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(module_path)
    from modules import *
    '''
    script, infile, outdir, picklefile, band_name = sys.argv
    '''
    # ------------------------------------------------------------------------------------------
    infile = "D:/temp/decid_tc_layerstack_2015_test250.tif"
    outdir = "D:/temp/"
    picklefile = "D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/" + \
                 "RFalbedo_deciduous_fraction_treecover_50000_cutoff_5_deg1_20191115T174618_spring.pickle"
    band_name = 'spr_albedo'
    # ------------------------------------------------------------------------------------------

    outfile = Handler(infile).add_to_filename('_{}'.format(band_name))
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
    Opt.cprint(raster.bnames)
    raster.bnames = ['decid', 'decidu', 'treecover', 'treecoveru', 'land']
    Opt.cprint(raster.bnames)
    raster.get_stats(True)

    Opt.cprint(raster.shape)

    regressor = RFRegressor.load_from_pickle(picklefile)

    Opt.cprint(regressor)

    data = regressor.data

    labels = data['labels']
    features = data['features']
    label_name = data['label_name']
    feature_names = data['feature_names']

    print(labels.shape)
    print(features.shape)

    exit(0)

    out_raster = RFRegressor.regress_raster(regressor,
                                            raster,
                                            outfile=outfile,
                                            band_name=band_name,
                                            output_type='median',
                                            mask_band='land',
                                            tile_size=min(raster.shape[1], raster.shape[2]),
                                            array_multiplier=0.01,
                                            nodatavalue=0,
                                            verbose=True)

    Opt.cprint(out_raster)

    exit()

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
