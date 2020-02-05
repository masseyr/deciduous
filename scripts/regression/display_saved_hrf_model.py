from modules import *


"""
This script is used to classify a raster using a selected RF model.
This script also generates the uncertainty raster.
"""


def make_final_regressor(pickle_file,
                         llim_,
                         ulim_,
                         over_adjust=1.0,
                         clip=0.01):

    # load classifier from file
    regressor = RFRegressor.load_from_pickle(pickle_file)

    data = regressor.data
    vdata = regressor.vdata

    all_data = dict()

    all_data['feature_names'] = data['feature_names']
    all_data['label_name'] = data['label_name']
    all_data['labels'] = list()
    all_data['features'] = list()

    for i, label in enumerate(data['labels']):
        all_data['labels'].append(label)
        all_data['features'].append(data['features'][i])

    for i, label in enumerate(vdata['labels']):
        all_data['labels'].append(label)
        all_data['features'].append(vdata['features'][i])

    param = {'trees': regressor.trees,
             'samp_split': regressor.samp_split,
             'samp_leaf': regressor.samp_leaf,
             'max_feat': regressor.max_feat}

    # initialize RF classifier
    model = RFRegressor(**param)

    # fit RF classifier using training data
    model.fit_data(all_data)

    model.get_adjustment_param(clip=clip,
                               data_limits=[llim_, ulim_],
                               over_adjust=over_adjust)

    return model


# main program
if __name__ == '__main__':

    # read in the input files
    # script, infile, outdir, rf_picklefile1, rf_picklefile2 = argv

    ulim = 1.0
    llim = 0.0

    active_dir = "D:/shared/"

    infile = active_dir + "Dropbox/projects/NAU/landsat_deciduous/data/temp/" + \
        "ABoVE_median_SR_NDVI_boreal_2015-0000023808-0000134912_subset.tif"

    outdir = active_dir + "Dropbox/projects/NAU/landsat_deciduous/data/temp/"

    rf_picklefile1 = active_dir + "Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/rf_test/" + \
        "gee_data_cleaning_v28_median2_RF_7.pickle"
    rf_picklefile2 = active_dir + "Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/rf_test/" + \
        "gee_data_cleaning_v28_median2_summer_RF_3.pickle"

    # ------------------common parameters-----------------
    over_adjust_ = 1.0
    clip_ = 0.01
    nodatavalue = -9999.0
    tile_size_multiplier = 10

    Opt.cprint('-----------------------------------------------------------------')
    # print('Script: ' + script)
    Opt.cprint('Random Forest file 1: ' + rf_picklefile1)
    Opt.cprint('Random Forest file 2: ' + rf_picklefile2)

    Opt.cprint('Raster: ' + infile)
    Opt.cprint('Outdir: ' + outdir)
    Opt.cprint('-----------------------------------------------------------------')

    # load classifier from file
    rf_regressor1 = make_final_regressor(rf_picklefile1,
                                         float(llim),
                                         float(ulim),
                                         over_adjust=over_adjust_,
                                         clip=clip_)

    rf_regressor2 = make_final_regressor(rf_picklefile2,
                                         float(llim),
                                         float(ulim),
                                         over_adjust=over_adjust_,
                                         clip=clip_)
    Opt.cprint(rf_regressor1)
    Opt.cprint(rf_regressor2)

    bandnames1_ = rf_regressor1.features
    Opt.cprint('Bands 1 : ')
    Opt.cprint(bandnames1_)

    bandnames2_ = rf_regressor2.features
    Opt.cprint('Bands 2 : ')
    Opt.cprint(bandnames2_)

