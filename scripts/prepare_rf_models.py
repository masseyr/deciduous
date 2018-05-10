from modules import *
from sys import argv

"""
This script initializes and fits training data to Random Forest (RF) models.
The data is pre prepared using analysis_initiate.py. The RF models are then
pickled and saved for later use. In addition this script also generates outputs
by classifying held-out samples using the RF model.
"""

# main program
if __name__ == '__main__':

    sep = Handler().sep

    # read file names from commandline arguments: (in order)
    # training samples,
    # held out samples,
    # out file to write,
    # directory to store RF classifier
    script, infile, inCfile, outfile, pickledir = argv

    # prepare training samples
    trn_samp = Samples(csv_file=infile, label_colname='Decid_AVG')
    print(trn_samp)

    # prepare held out samples
    trn_Csamp = Samples(csv_file=inCfile, label_colname='Decid_AVG')
    print(trn_Csamp)

    # initialize RF clasifier
    rf_model = RFRegressor(trees=500, samp_split=15, oob_score=False)
    print(rf_model)

    # fit RF classifier using training data
    rf_model.fit_data(trn_samp.format_data())
    print(rf_model)

    # save RF classifier using pickle
    picklefile = pickledir + sep + Handler(infile).basename.split('.')[0] + '.pickle'
    rf_model.pickle_it(picklefile)

    # predict using held out samples and print to file
    outfile1 = Handler(outfile).dirname + sep + \
        'val' + sep + 'val_samp_' + Handler(outfile).basename
    outfile1 = Handler(filename=outfile1).file_remove_check()
    pred = rf_model.sample_predictions(trn_Csamp.format_data(), outfile=outfile1, picklefile=picklefile)
    print(pred)

    # check training samples for fit
    outfile2 = Handler(outfile).dirname + sep + \
        'trn' + sep + 'trn_samp_' + Handler(outfile).basename
    outfile2 = Handler(filename=outfile2).file_remove_check()
    pred3 = rf_model.sample_predictions(trn_samp.format_data(), outfile=outfile2, picklefile=picklefile)
    print(pred3)

    # predict using all samples and print to file
    trn_Csamp.merge_data(trn_samp)
    print(trn_Csamp)

    # check all samples for fit
    outfile3 = Handler(outfile).dirname + sep + \
        'all' + sep + 'all_samp_' + Handler(outfile).basename
    outfile3 = Handler(filename=outfile3).file_remove_check()
    pred3 = rf_model.sample_predictions(trn_Csamp.format_data(), outfile=outfile3, picklefile=picklefile)
    print(pred3)
