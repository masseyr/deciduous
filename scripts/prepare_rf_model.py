from modules import *
import multiprocessing
from sys import argv
import sys
import time


"""
This script initializes and fits training data to prepare models.
The data is pre prepared using analysis_initiate.py. The RF models are then
pickled and saved for later use. In addition this script also generates outputs
by classifying held-out samples using the RF model.
"""


def fit_regressor(args):

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

    name, train_samp, valid_samp, in_file, pickle_dir, llim, ulim, param = args

    # initialize RF classifier
    model = RFRegressor(**param)
    model.time_it = True

    regress_limit = [0.025 * ulim, 0.975 * ulim]
    rsq_limit = 60.0

    # fit RF classifier using training data
    model.fit_data(train_samp.format_data())

    # predict using held out samples and print to file
    pred = model.sample_predictions(valid_samp.format_data(),
                                    regress_limit=regress_limit)

    out_dict = dict()
    out_dict['name'] = Handler(in_file).basename.split('.')[0] + name
    out_dict['rsq'] = pred['rsq'] * 100.0
    out_dict['slope'] = pred['slope']
    out_dict['intercept'] = pred['intercept']
    out_dict['rmse'] = pred['rmse']

    model.output = out_dict

    rsq = pred['rsq'] * 100.0
    slope = pred['slope']
    intercept = pred['intercept']
    rmse = pred['rmse']

    if rsq >= rsq_limit:
        if intercept > regress_limit[0]:
            model.adjustment['bias'] = -1.0 * (intercept / slope)

        model.adjustment['gain'] = 1.0 / slope
        model.adjustment['upper_limit'] = ulim
        model.adjustment['lower_limit'] = llim

        # file to write the model run output to
        outfile = pickle_dir + sep + \
                  Handler(in_file).basename.split('.')[0] + name + '.txt'
        outfile = Handler(filename=outfile).file_remove_check()

        # save RF classifier using pickle
        picklefile = pickle_dir + sep + \
                     Handler(in_file).basename.split('.')[0] + name + '.pickle'
        picklefile = Handler(filename=picklefile).file_remove_check()

        # predict using the model to store results in a file
        pred = model.sample_predictions(valid_samp.format_data(),
                                        outfile=outfile,
                                        picklefile=picklefile,
                                        regress_limit=regress_limit)

        out_dict['rsq'] = pred['rsq'] * 100.0
        out_dict['slope'] = pred['slope']
        out_dict['intercept'] = pred['intercept']
        out_dict['rmse'] = pred['rmse']

        out_dict['regress_low_limit'] = regress_limit[0]
        out_dict['regress_up_limit'] = regress_limit[1]

        model.output.update(out_dict)
        model.pickle_it(picklefile)

    return out_dict


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
    '''
    script, infile, pickledir, codename, n_iterations, cpus = argv

    '''
    pickledir = "C:/temp/decid/"
    infile = "D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/" + \
        "gee_data_clean_v15_corr.csv"
    codename = "v15"
    n_iterations = 50

    param = {"samp_split": 10, "max_feat": 19, "trees": 500, "samp_leaf": 1}
    # param = {"samp_split": 9, "max_feat": 17, "trees": 200, "samp_leaf": 2}

    min_decid = 0.0
    max_decid = 1.0

    label_colname = 'decid_frac'
    model_initials = 'RF'

    sample_partition = 70
    display = 10

    t = time.time()

    Handler(dirname=pickledir).dir_create()

    Opt.cprint(infile)
    Opt.cprint(pickledir)
    Opt.cprint(codename)

    cpus = multiprocessing.cpu_count() - 1
    n_iterations = int(n_iterations)

    sep = Handler().sep

    cpus = int(cpus)

    print('Number of CPUs: {}'.format(str(cpus)))

    Handler(dirname=pickledir).dir_create()

    # prepare training samples
    samp = Samples(csv_file=infile, label_colname=label_colname)
    samp_list = list()
    '''
    corr_samp = list()
    for elem in samp.x:
        corr_elem = list()
        for number in elem:
            if number < -32767:
                number = -32767
            elif number > 32768:
                number = 32768

            corr_elem.append(number)
        corr_samp.append(corr_elem)

    samp.x = corr_samp
    '''
    Opt.cprint('Randomizing samples...')

    for i in range(0, n_iterations):

        model_name = '_{}_{}'.format(model_initials,
                                     str(i+1))

        trn_samp, val_samp = samp.random_partition(sample_partition)

        samp_list.append([model_name,
                          trn_samp,
                          val_samp,
                          infile,
                          pickledir,
                          min_decid,
                          max_decid,
                          param])

    Opt.cprint('Number of elements in sample list : {}'.format(str(len(samp_list))))

    pool = multiprocessing.Pool(processes=cpus)

    results = pool.map(fit_regressor,
                       samp_list)

    Opt.cprint('Top {} models:'.format(str(display)))
    Opt.cprint('')
    Opt.cprint('R-sq, Model name')

    if len(results) > 0:

        Opt.cprint('Results:----------------------------------')
        for result in results:
            Opt.cprint(result)
        Opt.cprint('------------------------------------------')
        Opt.cprint('\nLength of results: {}\n'.format(len(results)))

        sep = Handler().sep

        Opt.cprint('Top {} models:'.format(str(display)))
        Opt.cprint('')
        Opt.cprint('R-sq, Model name')

        out_list = sorted(results,
                          key=lambda elem: elem['rsq'],
                          reverse=True)

        for output in out_list[0: (display - 1)]:
            Opt.cprint(output)

        summary_file = pickledir + 'results_summary_' + codename + '.csv'
        Opt.cprint('\nSummary file: {}\n'.format(summary_file))

        Handler.write_to_csv(out_list,
                             outfile=summary_file,
                             delimiter=',')

    else:
        Opt.cprint('\nNo results to summarize!\n')

    print('Time taken: {}'.format(display_time(time.time() - t)))
