from modules import *
import multiprocessing
from sys import argv
import sys
import pandas as pd
import time


"""
This script initializes and fits training data to prepare models.
The data is pre prepared using analysis_initiate.py. The RF models are then
pickled and saved for later use. In addition this script also generates outputs
by classifying held-out samples using the RF model.
"""


def fit_regressor(args_list):

    """
    Method to train and validate classification models
    and if the R-squared is > 0.5 then store the model and its
    properties in a pickled file and csv file respectively

    :param args_list: List of list of args represents the following in the given order:

    (name: Name of the model,
    train_samp: Samples object for training the classifier,
    val_samp: Samples object for validating the classifier,
    infile: input file containing the samples
    pickle_dir: folder to store the pickled classifier in)

    :returns: tuple (r-squared*100 , model_name)
    """
    sep = Handler().sep

    result_list = list()

    for args in args_list:

        name, train_samp, valid_samp, in_file, pickle_dir = args

        # initialize RF classifier
        model = RFRegressor(trees=200, samp_split=2, oob_score=False)
        model.time_it = True

        # fit RF classifier using training data
        model.fit_data(train_samp.format_data())

        # predict using held out samples and print to file
        pred = model.sample_predictions(valid_samp.format_data(),
                                        regress_limit=[0.01, 0.99])

        rsq = pred['rsq'] * 100.0
        slope = pred['slope']
        intercept = pred['intercept']

        out_dict = {'name': Handler(in_file).basename.split('.')[0] + name,
                    'rsq': pred['rsq'] * 100.0,
                    'slope': pred['slope'],
                    'intercept': pred['intercept'],
                    'rmse': pred['rmse']}

        if rsq >= 50.0:

            model.adjustment['gain'] = 1.0/slope
            model.adjustment['bias'] = -1.0 * (intercept/slope)
            model.adjustment['upper_limit'] = 1.0
            model.adjustment['lower_limit'] = 0.0

            # file to write the model run output to
            outfile = pickle_dir + sep + \
                Handler(in_file).basename.split('.')[0] + name + '.txt'
            outfile = Handler(filename=outfile).file_remove_check()

            # save RF classifier using pickle
            picklefile = pickle_dir + sep + \
                Handler(in_file).basename.split('.')[0] + name + '.pickle'
            picklefile = Handler(filename=picklefile).file_remove_check()

            model.pickle_it(picklefile)

            # predict using the model to store results in a file
            pred = model.sample_predictions(valid_samp.format_data(),
                                            outfile=outfile,
                                            picklefile=picklefile,
                                            regress_limit=[0.01, 0.99])

            out_dict['rsq'] = pred['rsq'] * 100.0
            out_dict['slope'] = pred['slope']
            out_dict['intercept'] = pred['intercept']
            out_dict['rmse'] = pred['rmse']

        result_list.append(out_dict)

    return result_list


def display_time(seconds,
                 precision=1):
    """
    method to display time in human readable format
    :param seconds: Number of seconds
    :param precision: Decimal precision
    :return: String
    """

    # define denominations
    intervals = [('weeks', 604800),
                 ('days', 86400),
                 ('hours', 3600),
                 ('minutes', 60),
                 ('seconds', 1)]

    # initialize list
    result = list()

    # coerce to float
    dtype = type(seconds).__name__
    if dtype != 'int' or dtype != 'long' or dtype != 'float':
        try:
            seconds = float(seconds)
        except (TypeError, ValueError, NameError):
            sys.stdout.write("Type not coercible to Float")

    # break denominations
    for name, count in intervals:
        if name != 'seconds':
            value = seconds // count
            if value:
                seconds -= value * count
                if value == 1:
                    name = name.rstrip('s')
                value = str(int(value))
                result.append("{v} {n}".format(v=value,
                                               n=name))
        else:
            value = "{:.{p}f}".format(seconds,
                                      p=precision)
            result.append("{v} {n}".format(v=value,
                                           n=name))

    # join output
    return ' '.join(result)


# main program
if __name__ == '__main__':

    # script, infile, pickledir, codename = argv

    pickledir = "C:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/prepared/pickle/"
    infile = "C:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/prepared/" \
        "ABoVE_test_V8_L7_all_2010_samp.csv"
    codename = "v62"

    t = time.time()

    Handler(dirname=pickledir).dir_create()

    Opt.cprint(infile)
    Opt.cprint(pickledir)
    Opt.cprint(codename)

    label_colname = 'decid_frac'
    model_initials = 'RF'
    n_iterations = 100
    sample_partition = 70
    display = 10

    sep = Handler().sep

    cpus = multiprocessing.cpu_count()

    print('Number of CPUs: {}'.format(str(cpus)))

    Handler(dirname=pickledir).dir_create()

    # prepare training samples
    samp = Samples(csv_file=infile, label_colname=label_colname)
    samp_list = list()

    Opt.cprint('Randomizing samples...')

    for i in range(0, n_iterations):

        model_name = '_{}_{}'.format(model_initials,
                                     str(i+1))

        trn_samp, val_samp = samp.random_partition(sample_partition)

        samp_list.append([model_name,
                          trn_samp,
                          val_samp,
                          infile,
                          pickledir])

    Opt.cprint('Number of elements in sample list : {}'.format(str(len(samp_list))))

    sample_chunks = [samp_list[i::cpus] for i in xrange(cpus)]

    chunk_length = list(str(len(chunk)) for chunk in sample_chunks)

    Opt.cprint('Distribution of chunks : {}'.format(', '.join(chunk_length)))

    pool = multiprocessing.Pool(processes=cpus)

    results = pool.map(fit_regressor,
                       sample_chunks)

    Opt.cprint('Top {} models:'.format(str(display)))
    Opt.cprint('')
    Opt.cprint('R-sq, Model name')

    out_list = list()
    if len(results) > 0:
        for result in results:
            for vals in result:
                out_list.append(vals)
        out_list.sort(reverse=True, key=lambda elem: elem['rsq'])

        for output in out_list[0: (display-1)]:
            Opt.cprint(output)

    if len(out_list) > 0:
        df = pd.DataFrame(out_list)
        df.to_csv(pickledir + sep + 'results_summary_' + codename + '.csv')
    else:
        Opt.cprint('No results to summarize!')

    print('Time taken: {}'.format(display_time(time.time() - t)))
