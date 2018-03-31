from modules import *
from sys import argv

# main program
if __name__ == '__main__':

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
    rf_model = Classifier(trees=500, samp_split=15, oob_score=False)
    print(rf_model)

    # fit RF classifier using training data
    rf_model.fit_data(trn_samp.format_data())
    print(rf_model)

    # save RF classifier using pickle
    picklefile = pickledir + os.path.sep + os.path.basename(infile).split('.')[0] + '.pickle'
    rf_model.pickle_it(picklefile)

    # predict using held out samples and print to file
    outfile1 = os.path.dirname(outfile) + os.path.sep + 'val_samp_' + os.path.basename(outfile)
    pred = rf_model.tree_predictions(trn_Csamp.format_data(), outfile=outfile1, picklefile=picklefile)
    print(pred)

    # check training samples for fit
    outfile2 = os.path.dirname(outfile) + os.path.sep + 'trn_samp_' + os.path.basename(outfile)
    pred3 = rf_model.tree_predictions(trn_Csamp.format_data(), outfile=outfile2, picklefile=picklefile)
    print(pred3)

    # predict using all samples and print to file
    trn_Csamp.merge_data(trn_samp)
    print(trn_Csamp)

    # check all samples for fit
    outfile3 = os.path.dirname(outfile) + os.path.sep + 'all_samp_' + os.path.basename(outfile)
    pred3 = rf_model.tree_predictions(trn_Csamp.format_data(), outfile=outfile3, picklefile=picklefile)
    print(pred3)
