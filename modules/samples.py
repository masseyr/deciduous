import numpy as np
from common import *
import random
from timer import Timer
from resources import bname_dict
from scipy.stats.stats import pearsonr
import warnings

__all__ = ['Samples']


class Samples:
    """
    Class to read and arrange sample data.
    Stores label and label names in y and y_names
    Stores feature and feature names in x and x_names.
    Currently the user has to provide sample csv files with one column as label (output)
    and the rest of the columns as feature attributes. There should be no index number column.
    All columns should be data only.
    """

    def __init__(self,
                 csv_file=None,
                 label_colname=None,
                 x=None,
                 y=None,
                 x_name=None,
                 y_name=None,
                 weights=None,
                 weights_colname=None,
                 use_band_dict=None,
                 **kwargs):

        """
        :param csv_file: csv file that contains the features (training or validation samples)
        :param label_colname: column in csv file that contains the feature label (output value)
        :param x: 2d array containing features (samples) without the label
        :param y: 1d array of feature labels (same order as x)
        :param x_name: 1d array of feature names (bands)
        :param y_name: name of label
        :param use_band_dict: list of attribute (band) names
        """
        self.csv_file = csv_file
        self.label_colname = label_colname
        self.x = x
        self.x_name = x_name
        self.y = y
        self.y_name = y_name

        self.weights = weights
        self.weights_colname = weights_colname
        self.use_band_dict = use_band_dict

        self.index = None
        self.nfeat = None

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        # either of label name or csv file is provided without the other
        if (csv_file is None) and (label_colname is None):
            pass  # warnings.warn("Samples initiated without data file or label")

        # label name or csv file are provided
        elif (label_colname is not None) and (csv_file is not None):

            temp = Handler(filename=csv_file).read_from_csv()

            # label name doesn't match
            if any(label_colname in s for s in temp['name']):
                loc = temp['name'].index(label_colname)
            else:
                raise ValueError("Label name mismatch.\nAvailable names: " + ', '.join(temp['name']))

            # read from data dictionary
            self.x_name = Sublist(elem.strip() for elem in temp['name'][:loc] + temp['name'][(loc + 1):])
            self.x = Sublist(feat[:loc] + feat[(loc + 1):] for feat in temp['feature'])
            self.y = Sublist(feat[loc] for feat in temp['feature'])
            self.y_name = temp['name'][loc].strip()

            # if band name dictionary is provided
            if use_band_dict is not None:
                self.y_name = [use_band_dict[b] for b in self.y_name]

        elif (label_colname is None) and (csv_file is not None):

            temp = Handler(filename=csv_file).read_from_csv()

            # read from data dictionary
            self.x_name = Sublist(elem.strip() for elem in temp['name'])
            self.x = Sublist(feat for feat in temp['feature'])

        else:
            ValueError("No data found for label.")

        if weights is None:
            if weights_colname is not None:
                if csv_file is not None:

                    temp = Handler(filename=csv_file).read_from_csv()

                    # label name doesn't match
                    if any(weights_colname in n for n in temp['name']):
                        loc = temp['name'].index(weights_colname)
                    else:
                        raise ValueError("Weight column name mismatch.\nAvailable names: " + ', '.join(temp['name']))

                    self.weights = Sublist(feat[loc] for feat in self.x)
                    self.x = Sublist(self.x).remove_by_loc(loc)

                else:
                    raise ValueError("No csv_file specified for weights")

        # if keywords are supplied
        if kwargs is not None:

            # columns containing data
            if 'columns' in kwargs:
                self.columns = kwargs['columns']
            else:
                self.columns = None

            # IDs of samples
            if 'ids' in kwargs:
                self.ids = kwargs['ids']
            else:
                self.ids = None

        else:
            self.columns = None
            self.ids = None

        if self.x is not None:

            if self.columns is None:
                self.columns = Sublist(range(0, len(self.x[0])))

            self.nsamp = len(self.x)
            self.nvar = len(self.x[0])
        else:
            self.nsamp = 0
            self.nvar = 0

        self.index = Sublist(range(0, self.nsamp))

        if self.x is not None:
            self.nfeat = len(self.x[0])

            self.xmin = list()
            self.xmax = list()

            for i in range(0, self.nfeat):
                self.xmin.append(min(list(x_elem[i] for x_elem in self.x)))
                self.xmax.append(max(list(x_elem[i] for x_elem in self.x)))

        if self.y is not None:
            self.ymin = min(self.y)
            self.ymax = max(self.y)

    def __repr__(self):
        """
        Representation of the Samples object
        :return: Samples class representation
        """
        if self.csv_file is not None:
            return "<Samples object from {cf} with {v} variables, {n} samples>".format(cf=Handler(self.csv_file).basename,
                                                                                       n=len(self.x),
                                                                                       v=len(self.x_name))
        elif self.csv_file is None and self.x is not None:
            return "<Samples object with {n} samples>".format(n=len(self.x))
        else:
            return "<Samples object: __empty__>"

    def format_data(self):
        """
        Method to format the samples to the RF model fit method
        :param self
        :return: dictionary of features and labels
        """
        if self.columns is not None:
            out_x = list()
            for tsamp in self.x:
                tsamp = Sublist(tsamp)[self.columns]
                out_x.append(tsamp)

            out_x_name = self.x_name[self.columns]
        else:
            out_x = self.x
            out_x_name = self.x_name

        return {
            'features': Opt.__copy__(out_x),
            'labels': Opt.__copy__(self.y),
            'label_name': Opt.__copy__(self.y_name),
            'feature_names': Opt.__copy__(out_x_name),
        }

    def correlation_matrix(self,
                           sensor='ls57',
                           display_progress=True):
        """
        Method to return a dictionary with correlation data
        rows = columns = variables (or dimensions)
        :param sensor: Sensor parameters to be used (current options: 'ls57' for Landsat 5 and 7,
                        and 'ls8' for Landsat 8)
        :param display_progress: Should the elements of correlation matrix
        be displayed while being calculated? (default: True)

        :return: Dictionary
        """
        # get data from samples
        data_mat = np.matrix(self.x)
        nsamp, nvar = data_mat.shape
        print(nsamp, nvar)

        # get names of variables variables
        var_names = list()
        i = 0
        for i, name in enumerate(self.x_name):
            print(str(i)+' '+name)
            if name in bname_dict[sensor]:
                var_names.append(bname_dict[sensor][name.upper()])
            else:
                var_names.append(name.upper())
            i = i + 1

        # initialize correlation matrix
        corr = np.zeros([nvar, nvar], dtype=float)

        # calculate correlation matrix
        for i in range(0, nvar):
            for j in range(0, nvar):
                corr[i, j] = np.abs(pearsonr(Sublist.column(data_mat, i),
                                             Sublist.column(data_mat, j))[0])
                if display_progress:
                    str1 = '{row} <---> {col} = '.format(row=var_names[i], col=var_names[j])
                    str2 = '{:{w}.{p}f}'.format(corr[i, j], w=3, p=2)
                    print(str1 + str2)

        return {'data': corr, 'names': var_names}

    def merge_data(self,
                   samp):
        """
        Merge two sample sets together
        :param self, samp:
        """
        for item in samp.x:
            self.x.append(item)
        for item in samp.y:
            self.y.append(item)

    @Timer.timing(True)
    def delete_column(self,
                      column_id=None,
                      column_name=None):
        """
        Function to remove a data column from the samples object
        :param column_id: ID (index) of the column
        :param column_name: Column label or name
        :return: Samples object with a column removed
        """
        if column_name is None and column_id is None:
            raise AttributeError('No argument for delete operation')

        elif column_id is None and column_name is not None:
            column_id = Sublist(self.x_name) == column_name

        temp = self.x

        self.x = list([val for j, val in enumerate(temp[i]) if j != column_id]
                      for i in range(0, self.nsamp))

        self.x_name = Sublist(self.x_name).remove(column_id)

        self.columns = list(range(0, len(self.x_name)))
        self.nvar = len(self.columns)

    def extract_column(self,
                       column_id=None,
                       column_name=None):
        """
        Function to extract a data column from the samples object
        :param column_id: ID (index) of the column
        :param column_name: Column label or name
        :return: Samples object with only one column
        """

        if column_name is None and column_id is None:
            raise AttributeError('No argument for extract operation')

        elif column_id is None and column_name is not None:
            column_id = Sublist(self.x_name) == column_name

        temp = [samp[column_id] for samp in self.x]
        temp_name = self.x_name[column_id]

        return {'name': temp_name, 'value': temp}

    def add_column(self,
                   column_name=None,
                   column_data=None,
                   column_order=None):
        """
        Function to add a column to the samples matrix
        :param column_name: Name of column to be added
        :param column_data: List of column values to be added
        :param column_order: List of numbers specifying column order for the column to be added
                            (e.g. if for three samples, the first value in column_data
                            is for second column, second value for first, third value for third,
                            the column_order is [1, 0, 2]
        :return: Samples object with added column
        """

        if column_data is None:
            raise AttributeError('No argument for add operation')

        if column_name is None:
            column_name = 'Column_{}'.format(str(len(self.x_name) + 1))

        if column_order is None or len(column_order) != len(self.x):
            print('Inconsistent or missing order - ignored')
            column_order = list(range(0, len(self.x)))

        temp = list()
        for i, j in enumerate(column_order):
            prev_samp = list(val for val in self.x[j])
            prev_samp.append(column_data[i])
            temp.append(prev_samp)

        self.x = temp
        self.x_name.append(column_name)
        self.columns = list(range(0, len(self.x_name)))
        self.nvar = len(self.columns)

    def save_to_file(self,
                     out_file):
        """
        Function to save sample object to csv file
        :param out_file: CSV file full path (string)
        :return: Write to file
        """

        samp_matrix = list()
        for i, j in enumerate(range(0, len(self.x))):
            samples = list(val for val in self.x[j])
            samples.append(self.y)
            samp_matrix.append(samples)

        out_arr = np.array(samp_matrix)
        out_names = self.x_name
        out_names.append(self.y_name)

        Handler(out_file).write_numpy_array_to_file(np_array=out_arr,
                                                    colnames=out_names)

    def random_partition(self,
                         percentage=75):

        """
        Method to randomly partition the samples based on a percentage
        :param percentage: Partition percentage (default: 75)
        (e.g. 75 for 75% training samples and 25% validation samples)
        :return: Tuple (Training sample object, validation sample object)
        """

        ntrn = int((percentage * self.nsamp) / 100.0)

        # randomly select training samples based on number
        trn_sites = random.sample(self.index, ntrn)
        val_sites = self.index.remove(trn_sites)

        # training sample object
        trn_samp = Samples()
        trn_samp.x_name = self.x_name
        trn_samp.y_name = self.y_name
        trn_samp.x = [self.x[i] for i in trn_sites]
        trn_samp.y = [self.y[i] for i in trn_sites]
        trn_samp.nsamp = len(trn_samp.x)
        trn_samp.index = Sublist(range(0, trn_samp.nsamp))
        trn_samp.nfeat = len(trn_samp.x[0])

        trn_samp.xmin = list()
        trn_samp.xmax = list()

        for i in range(0, trn_samp.nfeat):
            trn_samp.xmin.append(min(list(x_elem[i] for x_elem in trn_samp.x)))
            trn_samp.xmax.append(max(list(x_elem[i] for x_elem in trn_samp.x)))

        trn_samp.ymin = min(trn_samp.y)
        trn_samp.ymax = max(trn_samp.y)

        # validation sample object
        val_samp = Samples()
        val_samp.x_name = self.x_name
        val_samp.y_name = self.y_name
        val_samp.x = [self.x[i] for i in val_sites]
        val_samp.y = [self.y[i] for i in val_sites]
        val_samp.nsamp = len(val_samp.x)
        val_samp.index = Sublist(range(0, val_samp.nsamp))
        val_samp.nfeat = len(val_samp.x[0])

        val_samp.xmin = list()
        val_samp.xmax = list()

        for i in range(0, val_samp.nfeat):
            val_samp.xmin.append(min(list(x_elem[i] for x_elem in val_samp.x)))
            val_samp.xmax.append(max(list(x_elem[i] for x_elem in val_samp.x)))

        val_samp.ymin = min(val_samp.y)
        val_samp.ymax = max(val_samp.y)

        return trn_samp, val_samp

    def random_selection(self,
                         num=10):

        """
        Method to select a smaller number of samples from the Samples object
        :param num: Number of samples to select
        :return: Samples object
        """

        if num >= len(self.index):
            print('Number larger than population: {} specified for {} samples'.format(str(num),
                                                                                      str(len(self.index))))
            ran_samp_n = self.index
        else:
            ran_samp_n = random.sample(self.index, num)

        # training sample object
        ran_samp = Samples()
        ran_samp.x_name = self.x_name
        ran_samp.y_name = self.y_name
        ran_samp.x = [self.x[i] for i in ran_samp_n]
        ran_samp.y = [self.y[i] for i in ran_samp_n]
        ran_samp.nsamp = len(ran_samp.x)
        ran_samp.nfeat = len(ran_samp.x[0])
        ran_samp.index = Sublist(range(0, ran_samp.nsamp))

        ran_samp.xmin = list()
        ran_samp.xmax = list()

        for i in range(0, ran_samp.nfeat):
            ran_samp.xmin.append(min(list(x_elem[i] for x_elem in ran_samp.x)))
            ran_samp.xmax.append(max(list(x_elem[i] for x_elem in ran_samp.x)))

        ran_samp.ymin = min(ran_samp.y)
        ran_samp.ymax = max(ran_samp.y)

        return ran_samp

    def selection(self,
                  index_list):
        """
        Method to select samples based on an index list
        :param index_list:
        :return: Samples object
        """

        samp = Samples()
        samp.x_name = self.x_name
        samp.y_name = self.y_name
        samp.x = [self.x[i] for i in index_list]
        samp.y = [self.y[i] for i in index_list]
        samp.nsamp = len(samp.x)
        samp.nfeat = len(samp.x[0])
        samp.index = Sublist(range(0, samp.nsamp))

        samp.xmin = list()
        samp.xmax = list()

        for i in range(0, samp.nfeat):
            samp.xmin.append(min(list(x_elem[i] for x_elem in samp.x)))
            samp.xmax.append(max(list(x_elem[i] for x_elem in samp.x)))

        samp.ymin = min(samp.y)
        samp.ymax = max(samp.y)

        return samp

    def add_samp(self,
                 samp):
        """
        merge s Samples object into another
        :param samp:
        :return: None
        """

        for i in range(samp.nsamp):
            self.x.append(samp.x[i])
            self.y.append(samp.y[i])

        self.nsamp += samp.nsamp
        self.index = Sublist(range(0, self.nsamp))

        for i in range(0, self.nfeat):
            self.xmin.append(min(list(x_elem[i] for x_elem in self.x)))
            self.xmax.append(max(list(x_elem[i] for x_elem in self.x)))

        self.ymin = min(self.y)
        self.ymax = max(self.y)

    def make_folds(self,
                   n_folds=5):

        """
        Make n folds in sample sets
        :param n_folds:
        :return: list of tuples [(training samp, validation samp)...]
        """

        nsamp_list = list(len(self.index) // n_folds for _ in range(n_folds))
        if len(self.index) % n_folds > 0:
            nsamp_list[-1] += len(self.index) % n_folds

        index_list = Opt.__copy__(self.index)
        fold_samples = list()

        for fold_samp in nsamp_list:

            val_index = random.sample(index_list, fold_samp)

            index_list = index_list.remove(val_index)

            trn_index = self.index.remove(val_index)

            fold_samples.append((self.selection(trn_index), self.selection(val_index)))

        return fold_samples














