from modules import *
from mpi4py import MPI
from sys import argv
import json


"""
This script initializes and fits training data to prepare models.
The data is pre prepared using analysis_initiate.py. The RF models are then
pickled and saved for later use if they meet a certain criteria (Rsq > 0.60).
In addition this script also generates outputs by classifying held-out samples 
using the RF model. This script is run in parallel using MPI libraries. 
"""


@Timer.timing(True)
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

    Opt.cprint('Length of arguments at {} is {}'.format(str(rank), len(args_list)))

    if args_list is None:
        args_list = list()

    for args in args_list:

        name, train_samp, valid_samp, in_file, pickle_dir, llim, ulim, param = args

        # initialize RF classifier
        model = RFRegressor(**param)
        model.time_it = True

        regress_limit = [0.025*ulim, 0.975*ulim]
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

        result_list.append(out_dict)

    return result_list


if __name__ == '__main__':

    script, infile, pickledir, codename, n_iterations = argv

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    min_decid = 0
    max_decid = 10000

    label_colname = 'DECID_FRAC'
    model_initials = 'RF'
    sample_partition = 70
    n_iterations = int(n_iterations)
    display = 10
    param = {"samp_split": 10, "max_feat": 19, "trees": 500, "samp_leaf": 1}

    if rank == 0:

        Opt.cprint('Number of CPUs : {} '.format(str(size)))

        Handler(dirname=pickledir).dir_create()

        # prepare training samples
        samp = Samples(csv_file=infile, label_colname=label_colname)

        # limit int values to 16 bit
        corr_samp = list()
        for elem in samp.x:
            corr_elem = list()
            for number in elem:
                if number < -32767:
                    number = -32767
                elif number > 32767:
                    number = 32767

                corr_elem.append(number)
            corr_samp.append(corr_elem)

        samp.x = corr_samp

        samp_list = list()

        Opt.cprint('Randomizing samples...')

        for i in range(0, n_iterations):
            model_name = '_{}_{}'.format(model_initials,
                                         str(i + 1))

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

        sample_chunks = [samp_list[i::size] for i in xrange(size)]

        chunk_length = list(str(len(chunk)) for chunk in sample_chunks)

        Opt.cprint(' Distribution of chunks : {}'.format(', '.join(chunk_length)))

    else:
        sample_chunks = None

    try:
        samples = comm.scatter(sample_chunks,
                               root=0)
    except OverflowError:
        Opt.cprint('Overflow error while scattering samples at rank {}'.format(rank))
        samples = None

    result = fit_regressor(samples)

    result_array = comm.gather(result,
                               root=0)

    if rank == 0:

        results = [item for sublist in result_array for item in sublist if item is not None]

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

            summary_file = pickledir + sep + 'results_summary_' + codename + '.csv'
            Opt.cprint('\nSummary file: {}\n'.format(summary_file))

            Handler.write_to_csv(out_list,
                                 outfile=summary_file,
                                 delimiter=',')

        else:
            Opt.cprint('\nNo results to summarize!\n')
