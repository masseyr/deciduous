from modules import *
import numpy as np
from sys import argv
import os

"""
This script tests multiple RF models for different parameters:
Trees, min_samp_splits on RMSE and R-squared to calculate an
array for a surface plot.
"""

# main program
if __name__ == '__main__':

    sep = Handler().sep

    # read file names from commandline arguments: (in order)
    # training samples,
    # held out samples,
    # out file string,
    # directory to store RF classifier outputs
    script, infile, inCfile, outfile_str, outdir = argv

    outfile_rsq = outdir + sep + outfile_str + "_rsq.csv"
    outfile_rmse = outdir + sep + outfile_str + "_rmse.csv"

    if (not os.path.isfile(outfile_rmse)) or (not os.path.isfile(outfile_rsq)):

        # prepare training samples
        trn_samp = Samples(csv_file=infile,
                           label_colname='Decid_AVG')

        print(trn_samp)

        # prepare held out samples
        trn_Csamp = Samples(csv_file=inCfile,
                            label_colname='Decid_AVG')

        print(trn_Csamp)

        # define tree test list
        trees_min = 10
        trees_max = 1000
        tree_step = 10

        # define split test list
        split_min = 2
        split_max = 100
        split_step = 5

        # define trees (rows)
        trees = Sublist.custom_list(trees_min,
                                    trees_max,
                                    tree_step)

        rownames = [str(i) for i in trees]

        # define splits (columns)
        split = Sublist.custom_list(split_min,
                                    split_max,
                                    split_step)

        colnames = [str(i) for i in split]

        # 3d output array where array
        # 0 is rsq; 1 is rmse
        result = np.zeros((2,
                           len(trees),
                           len(split)),
                          dtype=float)

        result[1, :, :] = 1.0

        # iterate through all parameters
        for i in range(0, len(trees)):
            for j in range(0, len(split)):

                Opt.cprint('Making RF model with {} trees and {} sample splits'.format(str(trees[i]),
                                                                                       str(split[j])))

                # initialize RF clasifier
                rf_model = Classifier(trees=trees[i],
                                      samp_split=split[j],
                                      oob_score=False)

                # fit RF classifier using training data
                rf_model.fit_data(trn_samp.format_data())
                # print(rf_model)

                # get prediction from the RF model
                pred = rf_model.tree_predictions(trn_Csamp.format_data())

                # predict and store result in array
                result[0, i, j] = pred['rsq']
                result[1, i, j] = pred['rmse']

        # unassign
        rf_model = None
        pred = None

        # file handlers
        rsq_handler = Handler(filename=outfile_rsq)
        rmse_handler = Handler(filename=outfile_rmse)

        print('Writing arrays to files...')

        # write to file
        rsq_handler.write_numpy_array_to_file(result[0, :, :],
                                              rownames=rownames,
                                              colnames=colnames)

        rmse_handler.write_numpy_array_to_file(result[1, :, :],
                                               rownames=rownames,
                                               colnames=colnames)

    print('~Done!~')
