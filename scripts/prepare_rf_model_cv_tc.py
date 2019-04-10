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

    name, cut_samp, in_file, pickle_dir, llim, ulim, param = args

    regress_limit = [0.025 * ulim, 0.975 * ulim]
    rsq_limit = 60.0
    n_folds = 5

    samp_folds = cut_samp.make_folds(n_folds)

    model_list = list()

    for train_samp, valid_samp in samp_folds:

        # initialize RF classifier
        model = RFRegressor(**param)
        model.time_it = True

        # fit RF classifier using training data
        model.fit_data(train_samp.format_data())
        model.vdata = valid_samp.format_data()

        model.get_adjustment_param(clip=0.01,
                                   data_limits=[ulim, llim],
                                   over_adjust=1.05)

        # predict using held out samples and print to file
        pred = model.sample_predictions(valid_samp.format_data(),
                                        regress_limit=regress_limit)

        out_dict = dict()
        out_dict['name'] = Handler(in_file).basename.split('.')[0] + name
        out_dict['rsq'] = pred['rsq'] * 100.0
        out_dict['slope'] = pred['slope']
        out_dict['intercept'] = pred['intercept']
        out_dict['rmse'] = pred['rmse']

        print(out_dict)

        model.output = out_dict

        model_list.append(model)

    rsq_list = list(model.output['rsq'] for model in model_list)

    median_model = model_list[0]
    for model in model_list:
        if model.output['rsq'] == Sublist(rsq_list).max():
            median_model = model

    median_model.all_cv_results = list(model.output for model in model_list)

    if Sublist(rsq_list).median() >= rsq_limit:

        median_model.get_training_fit()

        median_model.output['regress_low_limit'] = regress_limit[0]
        median_model.output['regress_up_limit'] = regress_limit[1]

        median_model.output['var_importance'] = ';'.join(list('{}:{}'.format(elem[0], str(elem[1]))
                                                         for elem in median_model.var_importance()))

        # file to write the model run output to
        outfile = pickle_dir + sep + \
                  Handler(in_file).basename.split('.')[0] + name + '.txt'
        outfile = Handler(filename=outfile).file_remove_check()

        # save RF classifier using pickle
        picklefile = pickle_dir + sep + \
                     Handler(in_file).basename.split('.')[0] + name + '.pickle'
        picklefile = Handler(filename=picklefile).file_remove_check()

        # predict using the model to store results in a file
        pred = median_model.sample_predictions(median_model.vdata,
                                               outfile=outfile,
                                               picklefile=picklefile,
                                               regress_limit=regress_limit)
        median_model.pickle_it(picklefile)

    return median_model.output


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
    pickledir = "C:/temp/tree_cover/"
    infile = "d:/shared/" \
             "Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/" + \
        "out_tc_2010_samp_v1.csv"
    codename = "v1_0"
    n_iterations = 6

    param = {"samp_split": 10, "max_feat": 24, "trees": 400, "samp_leaf": 5}
    # param = {"samp_split": 9, "max_feat": 17, "trees": 200, "samp_leaf": 2}

    min_tc = 0.0
    max_tc = 100.0

    label_colname = 'tc_value'
    model_initials = 'RF'

    bootstrap_partition = 90
    display = 10

    t = time.time()

    Handler(dirname=pickledir).dir_create()

    Opt.cprint(infile)
    Opt.cprint(pickledir)
    Opt.cprint(codename)

    cpus = multiprocessing.cpu_count() - 2
    n_iterations = int(n_iterations)

    sep = Handler().sep

    cpus = int(cpus)

    print('Number of CPUs: {}'.format(str(cpus)))

    Handler(dirname=pickledir).dir_create()

    # prepare training samples
    samp = Samples(csv_file=infile, label_colname=label_colname)

    print(samp)

    samp_list = list()

    Opt.cprint('Randomizing samples...')

    for i in range(0, n_iterations):

        model_name = '_{}_{}'.format(model_initials,
                                     str(i+1))

        reduced_samp, _ = samp.random_partition(bootstrap_partition)
        samp_list.append([model_name,
                          reduced_samp,
                          infile,
                          pickledir,
                          min_tc,
                          max_tc,
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
