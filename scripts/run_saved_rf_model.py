from modules import *
from sys import argv
import os
import time
import psutil
import gc

#main program
if __name__ == '__main__':

    print_memory_usage()
    
    # read in the input files
    #script, rf_picklefile, infile, outdir = argv
    infile = "C:\\temp\\AK_LS_coll1_2010-0000013824-0000076032_uncomp_4096_5568.tif"
    rf_picklefile = "C:\\temp\\rf_pickle_200.pickle"
    outdir= "c:\\temp"
    print('-----------------------------------------------------------------')
    # print('Script: ' + script)
    print('Random Forest file: ' + rf_picklefile)
    print('Raster: ' + infile)
    print('Outdir: ' + outdir)
    print('-----------------------------------------------------------------')
    
    print_memory_usage()
    
    # wait 10 mins
    # counter = 0
    # while True:
    #     counter = counter + 1
    #     if counter % 10:
    #         print('Elapsed time: ' + str(counter) + ' seconds')
    #     
    #     if counter == 60:
    #         break
    #     else:
    #         # add delay
    #         time.sleep(10)

    # load classifier from file
    rf_classifier = Classifier.load_from_pickle(rf_picklefile)
    print(rf_classifier)
    classif_bandnames = rf_classifier.data['feature_names']
    print('Bands : ' + ' '.join(classif_bandnames))

    print_memory_usage()
    
    # get raster metadata
    ras = Raster.initialize(infile, use_dict=bname_dict)
    print(ras)
    bandnames = ras.bnames
    print('Raster bands: ' + ' '.join(bandnames))
    band_order = sublistfinder(bandnames, classif_bandnames)
    print('Band order: ' + ', '.join([str(b) for b in band_order]))
    
    print_memory_usage()
    
    gc.collect()
    
    # re-initialize raster
    ras = Raster.initialize(infile, get_array=True, band_order=band_order, use_dict=bname_dict)
    print(ras)
    
    print_memory_usage()

    # classify raster and write to file
    # classif = rf_classifier.classify_raster(ras, outdir=outdir)
    # print(classif)
    # classif.write_to_file()

    # standard deviation file
    varfile = rf_classifier.tree_variance_raster(ras, outdir=outdir)
    
    print_memory_usage()
    
    print(varfile)
    varfile.write_to_file()
    
    print_memory_usage()

    print('Done!')


