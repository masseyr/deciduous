from common import *
from raster import Raster

import os
import pickle
import numpy as np
from math import sqrt
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Classifier:
    """Classifier object to be used with scikit-learn Random Forest classifier"""

    def __init__(self,
                 trees=1,
                 samp_split=2,
                 oob_score=True,
                 criterion='mse',
                 classifier=None):

        """
        Initialize RF classifier using class parameters
        :param trees: Number of trees
        :param samp_split: Minimum number of samples for split
        :param oob_score: (bool) calculate out of bag score
        :param criterion: criterion to be used (default: 'mse', options: 'mse', 'mae')
        :param classifier: Random forest classifier object
        (as detailed in http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
        """
        self.trees = trees
        self.samp_split = samp_split
        self.oob_score = oob_score
        self.criterion = criterion
        self.data = None
        self.classifier = RandomForestRegressor(n_estimators=trees,
                                                min_samples_split=samp_split,
                                                criterion=criterion,
                                                oob_score=oob_score)

    def __repr__(self):
        # gather which attributes exist
        attr_truth = [hasattr(self.classifier, 'estimators_'),
                      hasattr(self.classifier, 'n_features_'),
                      hasattr(self.classifier, 'n_outputs_'),
                      hasattr(self.classifier, 'oob_score_')]

        # if any exist print them
        if any(attr_truth):

            print_str_list = list("Random Forest Regressor:\n")

            # strings to be printed for each attribute
            if attr_truth[0]:
                print_str_list.append("Estmators: {}\n".format(len(self.classifier.estimators_)))

            if attr_truth[1]:
                print_str_list.append("Features: {}\n".format(self.classifier.n_features_))

            if attr_truth[2]:
                print_str_list.append("Output: {}\n".format(self.classifier.n_outputs_))

            if attr_truth[3]:
                print_str_list.append("OOB Score: {:{w}.{p}f} %".format(self.classifier.oob_score_ * 100.0, w=3, p=2))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Random Forest Regressor: __empty__>"

    def fit_data(self,
                 data):
        """
        Train the classifier
        :param data: dictionary with values (generated using Samples.format_data())
        :return:
        """
        self.data = data
        self.classifier.fit(data['features'], data['labels'])

    def predict_from_feat(self,
                          data):
        """Predict using classifier
        :param data: a feature (sample) with attribures (bands)
        """
        data['labels'] = self.classifier.predict(data['features'])
        return data

    def pickle_it(self,
                  outfile):
        """
        Save classifier
        :param outfile: File to save the classifier to
        """
        outfile = file_remove_check(outfile)
        with open(outfile, 'wb') as fileptr:
            pickle.dump(self, fileptr)

    @staticmethod
    def load_from_pickle(infile):
        """
        Reload classifier from file
        :param infile: File to load classifier from
        """
        with open(infile, 'rb') as fileptr:
            classifier_obj = pickle.load(fileptr)
            return classifier_obj

    def var_importance(self):
        """
        Return list of tuples of band names and their importance
        """

        return [(band, importance) for band, importance in
                zip(self.data['feature_names'], self.classifier.feature_importances_)]

    def classify_raster(self,
                        raster_obj,
                        outfile=None,
                        outdir=None,
                        array_multiplier=1.0,
                        array_additive=0.0,
                        data_type=gdal.GDT_Float32):

        """Model predictions from the RF classifier
        :param raster_obj: Raster object with a 3d array
        :param outfile: Name of output classification file
        :param array_multiplier: Rescale data using this value
        :param array_additive: Rescale data using this value
        :param data_type: Raster output data type
        :param outdir: output folder
        :returns: classification as raster object
        """
        # resolving output name
        if outdir is None:
            if outfile is None:
                outfile = os.path.dirname(raster_obj.name) + os.path.sep + \
                          '.'.join(os.path.basename(raster_obj.name).split('.')[0:-1]) + '_classif.' + \
                          os.path.basename(raster_obj.name).split('.')[-1]
        elif outfile is None:
            outfile = outdir + os.path.sep + \
                      '.'.join(os.path.basename(raster_obj.name).split('.')[0:-1]) + '_classif.' + \
                      os.path.basename(raster_obj.name).split('.')[-1]
        else:
            outfile = outdir + os.path.sep + os.path.basename(outfile)

        # initialize raster object
        out_ras = Raster(outfile)

        # get shape of raster array
        nbands = raster_obj.shape[0]
        nrows = raster_obj.shape[1]
        ncols = raster_obj.shape[2]

        # reshape into a long 2d array (nband, nrow * ncol) for classification,
        new_shape = [nbands, nrows * ncols]
        temp_arr = raster_obj.array
        temp_arr = temp_arr.reshape(new_shape) * array_multiplier + array_additive
        temp_arr = temp_arr.swapaxes(0, 1)

        # output 1d array after prediction
        out_arr = self.calc_arr(temp_arr, output='pred')

        # output raster
        out_ras.dtype = data_type
        out_ras.transform = raster_obj.transform
        out_ras.crs_string = raster_obj.crs_string
        out_ras.array = out_arr.reshape([nrows, ncols])
        out_ras.shape = [1, nrows, ncols]
        out_ras.bnames = [self.data['label_name']]

        # return raster object
        return out_ras

    def tree_predictions(self,
                         dataarray,
                         picklefile=None,
                         outfile=None):
        """
        Get tree predictions from the RF classifier
        :param dataarray: input 2d data array
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        """

        # calculate variance of tree predictions
        var_y = [np.var([tree.predict(np.array(feat).reshape(1, -1)) for tree in self.classifier.estimators_])
                 for feat in dataarray['features']]

        # calculate mean of tree predictions
        y = [self.classifier.predict(np.array(feat).reshape(1, -1)).item() for feat in dataarray['features']]

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(dataarray['labels'], y))

        # r-squared of predicted versus actual
        rsq = r2_score(dataarray['labels'], y)

        # write to file
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in dataarray['labels']]),
                        'mean_y,' + ', '.join([str(elem) for elem in y]),
                        'var_y,' + ', '.join([str(elem) for elem in var_y]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(rsq),
                        'rf_file,' + picklefile]

            # write the list to file
            write_list_to_file(outfile, out_list)

        return {
            'var_y': var_y,
            'mean_y': y,
            'obs_y': dataarray['labels'],
            'rmse': rmse,
            'rsq': rsq
        }

    def tree_variance_raster(self,
                             raster_obj,
                             outfile=None,
                             outdir=None,
                             sd=True,
                             array_multiplier=1.0,
                             array_additive=0.0,
                             data_type=gdal.GDT_Float32):

        """Tree variance from the RF classifier
        :param raster_obj: object with a 3d array
        :param outfile: name of output classification file
        :param array_multiplier: rescale data using this value
        :param array_additive: Rescale data using this value
        :param data_type: output raster data type
        :param outdir: output folder
        :param sd: (bool) flag for standard deviation
        :returns: classification as raster object
        """
        # resolving output name
        if outdir is None:
            if outfile is None:
                outfile = os.path.dirname(raster_obj.name) + os.path.sep + \
                          '.'.join(os.path.basename(raster_obj.name).split('.')[0:-1]) + '_var.' + \
                          os.path.basename(raster_obj.name).split('.')[-1]
        elif outfile is None:
            outfile = outdir + os.path.sep + \
                      '.'.join(os.path.basename(raster_obj.name).split('.')[0:-1]) + '_var.' + \
                      os.path.basename(raster_obj.name).split('.')[-1]
        else:
            outfile = outdir + os.path.sep + os.path.basename(outfile)

        out_ras = Raster(outfile)

        # classify by line
        nbands = raster_obj.shape[0]
        nrows = raster_obj.shape[1]
        ncols = raster_obj.shape[2]

        print('Shape: ' + ', '.join([str(elem) for elem in raster_obj.shape]))

        # reshape into a long 2d array (nband, nrow * ncol) for classification,
        new_shape = [nbands, nrows * ncols]

        print('New Shape: ' + ', '.join([str(elem) for elem in new_shape]))
        temp_arr = raster_obj.array
        temp_arr = temp_arr.reshape(new_shape) * array_multiplier + array_additive
        temp_arr = temp_arr.swapaxes(0, 1)

        # apply the variance calculating function on the array
        if sd:
            out_arr = self.calc_arr(temp_arr, output='sd')
        else:
            out_arr = self.calc_arr(temp_arr, output='var')

        # output raster and metadata
        out_ras.dtype = data_type
        out_ras.transform = raster_obj.transform
        out_ras.crs_string = raster_obj.crs_string

        out_ras.array = out_arr.reshape([nrows, ncols])
        out_ras.shape = [1, nrows, ncols]
        out_ras.bnames = ['variance']

        # return raster object
        return out_ras

    def calc_arr(self,
                 arr,
                 ntile_max=4,
                 ntile_size=64,
                 output='pred'):
        """
        Calculate random forest tree variance. Tiling is necessary in this step because
        large numpy arrays can cause memory issues leading to large memory usage during
        creation.

        :param arr: inout image reshaped to 2d array (axis 0: all pixels, axis 1: all bands)
        :param ntile_max: Maximum number of tiles upto which the
                          input image is processed without tiling (default = 4).
                          You can choose any (small) number that suits the available memory.
        :param ntile_size: Size of each square tile (default = 64)
        :param output: which output to produce,
                       choices: ['sd', 'var', 'pred']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'pred' is for prediction
        :return: 1d image array (that will need reshaping for image output)
        """

        # define output array
        out_arr = arr[:, 0] * 0.0

        # input image size
        npx_inp = long(arr.shape[0])  # number of pixels in input image
        nb_inp = long(arr.shape[1])  # number of bands in input image

        # size of tiles
        npx_tile = long(ntile_size * ntile_size)  # pixels in each tile
        npx_last = npx_inp % npx_tile  # pixels in last tile
        ntiles = long(npx_inp) / long(npx_tile) + 1  # total tiles

        # if number of tiles in the image
        # are less than the specified number
        if ntiles > ntile_max:

            # define tile array
            tile_arr = np.array([list(range(0, npx_tile)) for _ in range(0, self.trees)], dtype=float)

            for i in range(0, ntiles - 1):

                # calculate tree predictions for each pixel in a 2d array
                for j, tree in enumerate(self.classifier.estimators_):
                    temp = tree.predict(arr[i * npx_tile:(i + 1) * npx_tile, :])
                    tile_arr[j, :] = temp

                # calculate standard dev or variance or prediction for each tree
                if output == 'sd':
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = np.sqrt(np.var(tile_arr, axis=0))
                elif output == 'var':
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = np.var(tile_arr, axis=0)
                else:
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = np.mean(tile_arr, axis=0)

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 2

                # calculate tree predictions for each pixel in a 2d array
                for j, tree in enumerate(self.classifier.estimators_):
                    temp = tree.predict(arr[i * npx_last:(i + 1) * npx_last, :])
                    tile_arr[j, :] = temp

                # calculate standard dev or variance or prediction for each tree
                if output == 'sd':
                    out_arr[i * npx_last:(i + 1) * npx_last] = np.sqrt(np.var(tile_arr, axis=0))
                elif output == 'var':
                    out_arr[i * npx_last:(i + 1) * npx_last] = np.var(tile_arr, axis=0)
                else:
                    out_arr[i * npx_last:(i + 1) * npx_last] = np.mean(tile_arr, axis=0)

        else:

            # initialize output array
            tree_pred_arr = np.array([list(range(0, arr.shape[0])) for _ in range(0, self.trees)], dtype=float)

            # calculate tree predictions for each pixel in a 2d array
            for i, tree in enumerate(self.classifier.estimators_):
                tree_pred_arr[i, :] = tree.predict(arr)

            # calculate standard dev or variance or prediction for each tree
            if output == 'sd':
                out_arr = np.sqrt(np.var(tree_pred_arr, axis=0))
            elif output == 'var':
                out_arr = np.var(tree_pred_arr, axis=0)
            else:
                out_arr = np.mean(tree_pred_arr, axis=0)

        return out_arr


class Samples:
    """
    Class to read and arrange sample data for RF classifier.
    Stores label and label names in y and y_names
    Stores feature and feature names in x and x_names
    """

    def __init__(self,
                 csv_file=None,
                 label_colname=None,
                 x=None,
                 y=None,
                 x_name=None,
                 y_name=None,
                 use_band_dict=None):

        """
        :param csv_file: csv file that contains the features (training or validation samples)
        :param label_colname: column in csv file that contains the feature label (output value)
        :param x: 2d array containing features (samples) without the label
        :param y: 1d array of feature labels (same order as x)
        :param x_name: 1d array of feature attributes (bands)
        :param y_name: name of label
        :param use_dict: list of attribute (band) names
        """
        self.csv_file = csv_file
        self.label_colname = label_colname
        self.x = x
        self.x_name = x_name
        self.y = y
        self.y_name = y_name
        self.use_band_dict = use_band_dict

        # either of label name or csv file is provided without the other
        if (label_colname is not None) != (csv_file is not None):
            raise ValueError("Missing label or file")

        # label name or csv file are provided
        elif (label_colname is not None and
              csv_file is not None):
            temp = read_from_csv(csv_file)

            # label name doesn't match
            if any(label_colname in s for s in temp['name']):
                loc = temp['name'].index(label_colname)
            else:
                raise ValueError("Label name mismatch.\nAvailable names: " + ', '.join(temp['name']))

            # read from data dictionary
            self.x = [feat[:loc] + feat[(loc + 1):] for feat in temp['feature']]
            self.x_name = [elem.strip() for elem in temp['name'][:loc] + temp['name'][(loc + 1):]]
            self.y = [feat[loc] for feat in temp['feature']]
            self.y_name = temp['name'][loc].strip()

            # if band name dictionary is provided
            if use_band_dict is not None:
                self.y_name = [use_band_dict[b] for b in self.y_name]

        # no label name or csv file, but either feature or label list is provided without the other
        elif (x is not None) != (y is not None):
            raise ValueError("Missing label or feature")

    def __repr__(self):
        """
        Representation of the Samples object
        :return: Samples class representation
        """
        if self.csv_file is not None:
            return "<Samples object from {cf} with {n} samples>".format(cf=os.path.basename(self.csv_file),
                                                                        n=len(self.x))
        elif self.csv_file is None and self.x is not None:
            return "<Samples object with {n} samples>".format(n=len(self.x))
        else:
            "<Samples object: __empty__>"

    def format_data(self):
        """
        Method to format the samples to the RF model fit method
        :param self
        :return: dictionary of features and labels
        """
        return {
            'features': self.x,
            'labels': self.y,
            'label_name': self.y_name,
            'feature_names': self.x_name
        }

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
