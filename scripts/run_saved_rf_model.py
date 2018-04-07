from modules import *
from sys import argv
import os
import time
import psutil
import gc


"""
This script is used to classify a raster using a selected RF model.
This script also generates the uncertainty raster.
"""

#main program
if __name__ == '__main__':

    # read in the input files
    script, rf_picklefile, infile, outdir = argv

    print('-----------------------------------------------------------------')
    # print('Script: ' + script)
    print('Random Forest file: ' + rf_picklefile)
    print('Raster: ' + infile)
    print('Outdir: ' + outdir)
    print('-----------------------------------------------------------------')

    # load classifier from file
    rf_classifier = Classifier.load_from_pickle(rf_picklefile)
    print(rf_classifier)
    classif_bandnames = rf_classifier.data['feature_names']
    print('Bands : ' + ' '.join(classif_bandnames))
    
    # get raster metadata
    ras = Raster.initialize(infile,
                            use_dict=bname_dict)
    print(ras)
    bandnames = ras.bnames
    print('Raster bands: ' + ' '.join(bandnames))
    band_order = Sublist(bandnames).sublistfinder(classif_bandnames)
    print('Band order: ' + ', '.join([str(b) for b in band_order]))

    # re-initialize raster
    ras = Raster.initialize(infile,
                            get_array=True,
                            band_order=band_order,
                            use_dict=bname_dict)
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


