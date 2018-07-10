from modules import *
from sys import argv
import gc


"""
This script is used to classify a raster using a selected RF model.
This script also generates the uncertainty raster.
"""

#main program
if __name__ == '__main__':

    # read in the input files
    # script, rf_picklefile, infile, outdir = argv
    rf_picklefile="C:\\Users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\ABoVE_CAN_AK_all_2010_trn_samp_clean_md95.pickle"
    infile="C:\\temp\\ABoVE_img_am_2010_z1-0000000000-0000048384.tif"
    outdir="C:\\temp\\"

    print('-----------------------------------------------------------------')
    # print('Script: ' + script)
    print('Random Forest file: ' + rf_picklefile)
    print('Raster: ' + infile)
    print('Outdir: ' + outdir)
    print('-----------------------------------------------------------------')

    # load classifier from file
    rf_classifier = RFRegressor.load_from_pickle(rf_picklefile)
    print(rf_classifier)
    classif_bandnames = rf_classifier.features
    print('Bands : "' + '", "'.join(classif_bandnames) + '"')
    
    # get raster metadata
    ras = Raster(infile)
    ras.initialize()
    print(ras.shape)

    print(ras)
    bandnames = ras.bnames
    print('Raster bands: "' + '", "'.join(bandnames) + '"')
    band_order = Sublist(bandnames).sublistfinder(classif_bandnames)
    print('Band order: ' + ', '.join([str(b) for b in band_order]))

    # re-initialize raster
    ras.initialize(get_array=True,
                   band_order=band_order)

    print(ras)

    # classify raster and write to file
    classif = rf_classifier.classify_raster(ras,
                                            outdir=outdir)
    print(classif)
    classif.write_to_file()

    # standard deviation file
    varfile = rf_classifier.tree_variance_raster(ras,
                                                 array_multiplier=0.00001,
                                                 outdir=outdir)

    print(varfile)
    varfile.write_to_file()
    
    Opt.print_memory_usage()
    gc.collect()

    print('Done!')


