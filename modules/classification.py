import pickle
import numpy as np
from scipy import stats
from math import sqrt
from osgeo import gdal
from common import *
from raster import Raster
from timer import Timer
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


__all__ = ['RFRegressor',
           'MRegressor']


sep = Handler().sep


class _Classifier(object):

    time_it = False

    def __init__(self,
                 data=None,
                 classifier=None,
                 **kwargs):
        self.data = data
        self.classifier = classifier
        self.features = None
        self.label = None

        self.adjustment = dict()

        if kwargs is not None:
            if 'timer' in kwargs:
                _Classifier.time_it = kwargs['timer']

    def __repr__(self):
        return "<Classifier base class>"

    def fit_data(self,
                 data,
                 use_weights=False):
        """
        Train the classifier
        :param data: dictionary with values (generated using Samples.format_data())
        :param use_weights: If the sample weights provided should be used? (default: False)
        :return: Nonetype
        """
        self.data = data

        if 'weights' not in data or not use_weights:
            self.classifier.fit(data['features'], data['labels'])
        else:
            self.classifier.fit(data['features'], data['labels'], data['weights'])

        self.features = data['feature_names']
        self.label = data['label_name']

    def predict(self, *args, **kwargs):
        """Placeholder function"""
        return

    def pickle_it(self,
                  outfile):
        """
        Save classifier
        :param outfile: File to save the classifier to
        """
        outfile = Handler(filename=outfile).file_remove_check()
        with open(outfile, 'wb') as fileptr:
            pickle.dump(self, fileptr)

    @classmethod
    @Timer.timing(time_it)
    def load_from_pickle(cls,
                         infile):
        """
        Reload classifier from file
        :param infile: File to load classifier from
        """
        with open(infile, 'rb') as fileptr:
            classifier_obj = pickle.load(fileptr)
            return classifier_obj

    @Timer.timing(time_it)
    def classify_raster(self,
                        raster_obj,
                        outfile=None,
                        outdir=None,
                        band_name='prediction',
                        output_type='pred',
                        array_multiplier=1.0,
                        array_additive=0.0,
                        data_type=gdal.GDT_Float32):

        """Tree variance from the RF classifier
        :param raster_obj: Initialized Raster object with a 3d array
        :param outfile: name of output classification file
        :param array_multiplier: rescale data using this value
        :param array_additive: Rescale data using this value
        :param data_type: output raster data type
        :param band_name: Name of the output raster band
        :param outdir: output folder
        :param output_type: Should the output be standard deviation ('sd'),
                            variance ('var'), or prediction ('pred')
        :returns: classification as raster object
        """

        # file handler object
        handler = Handler(raster_obj.name)

        # resolving output name
        if outdir is None:
            if outfile is None:
                outfile = handler.add_to_filename('_{}'.format(band_name))
        elif outfile is None:
            handler.dirname = outdir
            outfile = handler.add_to_filename('_{}'.format(band_name))
        else:
            outfile = Handler(outfile).file_remove_check()

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
        out_arr = self.predict(temp_arr,
                               output=output_type)

        # output raster and metadata
        out_ras.dtype = data_type
        out_ras.transform = raster_obj.transform
        out_ras.crs_string = raster_obj.crs_string

        out_ras.array = out_arr.reshape([nrows, ncols])
        out_ras.shape = [1, nrows, ncols]
        out_ras.bnames = [band_name]

        # return raster object
        return out_ras

    @staticmethod
    def linear_regress(x,
                       y,
                       xlim=None,
                       ylim=None):
        """
        Calculate linear regression attributes
        :param x: Vector of independent variables
        :param y: Vector of dependent variables
        :param xlim: 2 element list [lower limit, upper limit] of starting and ending values for x vector
        :param ylim: 2 element list [lower limit, upper limit] of starting and ending values for y vector
        """
        if xlim is not None:
            x_index_list = Sublist(x).range(*xlim,
                                            index=True)
            x = list(x[i] for i in x_index_list)
            y = list(y[i] for i in x_index_list)

        if ylim is not None:
            y_index_list = Sublist(y).range(*ylim,
                                            index=True)
            x = list(x[i] for i in y_index_list)
            y = list(y[i] for i in y_index_list)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        rsq = r_value ** 2
        return {
            'rsq': rsq,
            'slope': slope,
            'intercept': intercept,
            'pval': p_value,
            'stderr': std_err
        }


class MRegressor(_Classifier):
    """Multiple linear regressor object for scikit-learn linear model"""

    time_it = False

    def __init__(self,
                 data=None,
                 classifier=None,
                 intercept=True,
                 jobs=1,
                 normalize=False,
                 **kwargs):

        super(MRegressor, self).__init__(data,
                                         classifier)
        if self.classifier is None:
            self.classifier = linear_model.LinearRegression(copy_X=True,
                                                            fit_intercept=intercept,
                                                            n_jobs=jobs,
                                                            normalize=normalize)

        self.intercept = self.classifier.intercept_ if hasattr(self.classifier, 'intercept_') else None
        self.coefficient = self.classifier.coef_ if hasattr(self.classifier, 'coef_') else None

        if kwargs is not None:
            if 'timer' in kwargs:
                MRegressor.time_it = kwargs['timer']

    def __repr__(self):
        # gather which attributes exist
        attr_truth = [hasattr(self.classifier, 'coef_'),
                      hasattr(self.classifier, 'intercept_')]

        if any(attr_truth):

            print_str_list = list("Multiple Linear Regressor:\n")

            # strings to be printed for each attribute
            if attr_truth[0]:
                print_str_list.append("Coefficients: {}\n".format(len(self.classifier.coef_)))

            if attr_truth[1]:
                print_str_list.append("Intercept: {}\n".format(self.classifier.intercept_))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Multiple Linear Regressor: __empty__>"

    @Timer.timing(time_it)
    def predict(self,
                arr,
                ntile_max=9,
                tile_size=128,
                **kwargs):
        """
        Calculate random forest model prediction, variance, or standard deviation.
        Variance or standard deviation is calculated across all trees.
        Tiling is necessary in this step because large numpy arrays can cause
        memory issues during creation.

        :param arr: input numpy 2d array (axis 0: features (pixels), axis 1: bands)
        :param ntile_max: Maximum number of tiles up to which the
                          input image or array is processed without tiling (default = 9).
                          You can choose any (small) number that suits the available memory.
        :param tile_size: Size of each square tile (default = 128)
                :param kwargs: Keyword arguments:
                       'gain': Adjustment of the predicted output by linear adjustment of gain (slope)
                       'bias': Adjustment of the predicted output by linear adjustment of bias (intercept)
                       'upper_limit': Limit of maximum value of prediction
                       'lower_limit': Limit of minimum value of prediction
        :return: 1d image array (that will need reshaping if image output)
        """
        if kwargs is not None:
            for key, value in kwargs.items():
                self.adjustment[key] = value

        # define output array
        out_arr = arr[:, 0] * 0.0

        # input image size
        npx_inp = long(arr.shape[0])  # number of pixels in input image
        nb_inp = long(arr.shape[1])  # number of bands in input image

        # size of tiles
        npx_tile = long(tile_size * tile_size)  # pixels in each tile
        npx_last = npx_inp % npx_tile  # pixels in last tile
        ntiles = long(npx_inp) / long(npx_tile) + 1  # total tiles

        # if number of tiles in the image
        # are less than the specified number
        if ntiles > ntile_max:

            for i in range(0, ntiles - 1):

                # calculate predictions for each pixel in a 2d array
                out_arr[i * npx_tile:(i + 1) * npx_tile] = \
                    self.classifier.predict(arr[i * npx_tile:(i + 1) * npx_tile, :])

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 2
                out_arr[i * npx_last:(i + 1) * npx_last] = \
                    self.classifier.predict(arr[i * npx_last:(i + 1) * npx_last, :])

        else:
            out_arr = self.classifier.predict(arr)

        if 'gain' in self.adjustment:
            out_arr = out_arr * self.adjustment['gain']

        if 'bias' in self.adjustment:
            out_arr = out_arr + self.adjustment['bias']

        if 'upper_limit' in self.adjustment:
            out_arr[out_arr > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

        if 'lower_limit' in self.adjustment:
            out_arr[out_arr < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

        return out_arr

    def sample_predictions(self,
                           data,
                           picklefile=None,
                           outfile=None):
        """
        Get tree predictions from the RF classifier
        :param data: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        """

        # calculate variance of tree predictions
        y = self.predict(np.array(data['features']))

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(data['labels'], y))

        # r-squared of predicted versus actual
        lm = self.linear_regress(data['labels'], y)

        # if either one of outfile or pickle file are available
        # then raise error
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")

        # if outfile and pickle file are both available
        # then write to file and proceed to return
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in data['labels']]),
                        'mean_y,' + ', '.join([str(elem) for elem in y]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(lm['rsq']),
                        'slope,' + str(lm['slope']),
                        'intercept,' + str(lm['intercept']),
                        'rf_file,' + picklefile]

            # write the list to file
            Handler(filename=outfile).write_list_to_file(out_list)

        # if outfile and pickle file are not provided,
        # then only return values
        return {
            'mean_y': y,
            'obs_y': data['labels'],
            'rmse': rmse,
            'rsq': lm['rsq'],
            'slope': lm['slope'],
            'intercept': lm['intercept']
        }


class RFRegressor(_Classifier):
    """Random Forest Regressor class for scikit-learn Random Forest regressor"""

    time_it = False

    def __init__(self,
                 data=None,
                 classifier=None,
                 trees=10,
                 samp_split=2,
                 oob_score=True,
                 criterion='mse',
                 **kwargs):
        """
        Initialize RF classifier using class parameters
        :param trees: Number of trees
        :param samp_split: Minimum number of samples for split
        :param oob_score: (bool) calculate out of bag score
        :param criterion: criterion to be used (default: 'mse', options: 'mse', 'mae')
        (some parameters are as detailed in
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
        """

        super(RFRegressor, self).__init__(data,
                                          classifier)

        if self.classifier is None:
            self.classifier = RandomForestRegressor(n_estimators=trees,
                                                    min_samples_split=samp_split,
                                                    criterion=criterion,
                                                    oob_score=oob_score)
        self.trees = trees
        self.samp_split = samp_split
        self.oob_score = oob_score
        self.criterion = criterion

        if kwargs is not None:
            if 'timer' in kwargs:
                MRegressor.time_it = kwargs['timer']

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
                print_str_list.append("Estimators: {}\n".format(len(self.classifier.estimators_)))

            if attr_truth[1]:
                print_str_list.append("Features: {}\n".format(self.classifier.n_features_))

            if attr_truth[2]:
                print_str_list.append("Output: {}\n".format(self.classifier.n_outputs_))

            if attr_truth[3]:
                print_str_list.append("OOB Score: {:{w}.{p}f} %".format(self.classifier.oob_score_ * 100.0,
                                                                        w=3, p=2))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Random Forest Regressor: __empty__>"

    @Timer.timing(time_it)
    def predict(self,
                arr,
                ntile_max=9,
                tile_size=128,
                output='pred',
                **kwargs):
        """
        Calculate random forest model prediction, variance, or standard deviation.
        Variance or standard deviation is calculated across all trees.
        Tiling is necessary in this step because large numpy arrays can cause
        memory issues during creation.

        :param arr: input 2d array (axis 0: features (pixels), axis 1: bands)
        :param ntile_max: Maximum number of tiles up to which the
                          input image or array is processed without tiling (default = 9).
                          You can choose any (small) number that suits the available memory.
        :param tile_size: Size of each square tile (default = 128)
        :param output: which output to produce,
                       choices: ['sd', 'var', 'pred', 'full']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'pred' is for prediction or mean of tree outputs
                       'full' is for the full spectrum of the leaf nodes' prediction
        :param kwargs: Keyword arguments:
                       'gain': Adjustment of the predicted output by linear adjustment of gain (slope)
                       'bias': Adjustment of the predicted output by linear adjustment of bias (intercept)
                       'upper_limit': Limit of maximum value of prediction
                       'lower_limit': Limit of minimum value of prediction
        :return: 1d image array (that will need reshaping if image output)
        """

        if kwargs is not None:
            for key, value in kwargs.items():
                self.adjustment[key] = value

        # define output array
        out_arr = Opt.__copy__(arr[:, 0]) * 0.0
        arr_shp = out_arr.shape

        full_arr = None
        if output == 'full':
            full_arr = np.empty([self.trees, arr_shp[0]])

        # input image size
        npx_inp = long(arr.shape[0])  # number of pixels in input image
        nb_inp = long(arr.shape[1])  # number of bands in input image

        # size of tiles
        npx_tile = long(tile_size * tile_size)  # pixels in each tile
        npx_last = npx_inp % npx_tile  # pixels in last tile
        ntiles = long(npx_inp) / long(npx_tile) + 1  # total tiles

        # if number of tiles in the image
        # are less than the specified number
        if ntiles > ntile_max:

            # define tile array
            tile_arr = np.array([list(range(0, npx_tile)) for _ in range(0, self.trees)], dtype=float)

            for i in range(0, ntiles - 1):

                print('Processing tile {} of {}'.format(str(i+1), ntiles))

                # calculate tree predictions for each pixel in a 2d array
                for j, tree in enumerate(self.classifier.estimators_):
                    temp = tree.predict(arr[i * npx_tile:(i + 1) * npx_tile, :])
                    tile_arr[j, :] = temp

                # calculate standard dev or variance or prediction for each tree
                if output == 'sd':
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = np.sqrt(np.var(tile_arr, axis=0))
                elif output == 'var':
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = np.var(tile_arr, axis=0)
                elif output == 'pred':
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = np.median(tile_arr, axis=0)

                elif output == 'full':
                    full_arr[:, i * npx_tile:(i + 1) * npx_tile] = tile_arr
                else:
                    return RuntimeError("No output type specified")

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 2

                print('Processing tile {} of {}'.format(str(i+1), ntiles))

                # calculate tree predictions for each pixel in a 2d array
                for j, tree in enumerate(self.classifier.estimators_):
                    temp = tree.predict(arr[i * npx_tile:(i * npx_tile + npx_last), :])
                    tile_arr = temp

                # calculate standard dev or variance or prediction for each tree
                if output == 'sd':
                    out_arr[i * npx_tile:(i * npx_tile + npx_last)] = np.sqrt(np.var(tile_arr, axis=0))
                elif output == 'var':
                    out_arr[i * npx_tile:(i * npx_tile + npx_last)] = np.var(tile_arr, axis=0)
                elif output == 'pred':
                    out_arr[i * npx_tile:(i * npx_tile + npx_last)] = np.mean(tile_arr, axis=0)

                elif output == 'full':
                    full_arr[:, i * npx_tile:(i * npx_tile + npx_last)] = tile_arr
                else:
                    return RuntimeError("No output type specified")
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
            elif output == 'pred':
                out_arr = np.mean(tree_pred_arr, axis=0)

            elif output == 'full':
                full_arr = tree_pred_arr
            else:
                return RuntimeError("No output type specified")

        if output == 'full':

            if 'gain' in self.adjustment:
                full_arr = full_arr * self.adjustment['gain']

            if 'bias' in self.adjustment:
                full_arr = full_arr + self.adjustment['bias']

            if 'upper_limit' in self.adjustment:
                full_arr[full_arr > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

            if 'lower_limit' in self.adjustment:
                full_arr[full_arr < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

            return full_arr

        else:

            if 'gain' in self.adjustment:
                out_arr = out_arr * self.adjustment['gain']

            if 'bias' in self.adjustment:
                out_arr = out_arr + self.adjustment['bias']

            if 'upper_limit' in self.adjustment:
                out_arr[out_arr > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

            if 'lower_limit' in self.adjustment:
                out_arr[out_arr < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

            return out_arr

    def sample_predictions(self,
                           data,
                           picklefile=None,
                           outfile=None,
                           **kwargs):
        """
        Get tree predictions from the RF classifier
        :param data: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        :param kwargs: Keyword arguments:
               'gain': Adjustment of the predicted output by linear adjustment of gain (slope)
               'bias': Adjustment of the predicted output by linear adjustment of bias (intercept)
               'upper_limit': Limit of maximum value of prediction
               'lower_limit': Limit of minimum value of prediction
               'regress_limit': 2 element list of Minimum and Maximum limits of the label array [min, max]
               'all_y': Boolean (if all lef outputs should be calculated)
               'var_y': Boolean (if variance of leaf nodes should be calculated)

        """

        if kwargs is not None:
            for key, value in kwargs.items():
                self.adjustment[key] = value

        if 'regress_limit' in kwargs:
            regress_limit = kwargs['regress_limit']
        else:
            regress_limit = None

        # calculate variance of tree predictions
        var_y = None
        if 'var_y' in self.adjustment:
            if self.adjustment['var_y']:
                var_y = self.predict(np.array(data['features']),
                                     output='var')

        # calculate mean of tree predictions
        pred_y = self.predict(np.array(data['features']),
                              output='pred')

        # calculate mean of tree predictions
        all_y = None
        if 'all_y' in self.adjustment:
            if self.adjustment['all_y']:
                all_y = self.predict(np.array(data['features']),
                                     output='full')

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(data['labels'], pred_y))

        # r-squared of predicted versus actual
        lm = self.linear_regress(data['labels'],
                                 pred_y,
                                 xlim=regress_limit)

        # if either one of outfile or pickle file are available
        # then raise error
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")

        # if outfile and pickle file are both available
        # then write to file and proceed to return
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in data['labels']]),
                        'pred_y,' + ', '.join([str(elem) for elem in pred_y]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(lm['rsq']),
                        'slope,' + str(lm['slope']),
                        'intercept,' + str(lm['intercept']),
                        'rf_file,' + picklefile]

            if all_y is not None:
                out_list.append('all_y,' + '[' + ', '.join(['[' + ', '.join(str(elem) for
                                                            elem in arr) + ']' for arr in list(all_y)]) + ']')
            if var_y is not None:
                out_list.append('var_y,' + ', '.join([str(elem) for elem in var_y]))

            # write the list to file
            Handler(filename=outfile).write_list_to_file(out_list)

        # if outfile and pickle file are not provided,
        # then only return values
        out_dict = {
            'pred_y': pred_y,
            'obs_y': data['labels'],
            'rmse': rmse,
            'rsq': lm['rsq'],
            'slope': lm['slope'],
            'intercept': lm['intercept']
        }

        if all_y is not None:
            out_dict['all_y'] = all_y
        if var_y is not None:
            out_dict['var_y'] = var_y

        return out_dict

    def var_importance(self):
        """
        Return list of tuples of band names and their importance
        """

        return [(band, importance) for band, importance in
                zip(self.data['feature_names'], self.classifier.feature_importances_)]

