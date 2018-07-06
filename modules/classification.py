import pickle
import numpy as np
from math import sqrt
from osgeo import gdal
from common import *
from raster import Raster
from timer import Timer
from samples import Samples
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

        if kwargs is not None:
            if 'timer' in kwargs:
                _Classifier.time_it = kwargs['timer']

    def __repr__(self):
        return "<Classifier base class>"

    def fit_data(self,
                 data):
        """
        Train the classifier
        :param data: dictionary with values (generated using Samples.format_data())
        :return:
        """
        self.data = data
        self.classifier.fit(data['features'], data['labels'])

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
        :return: 1d image array (that will need reshaping if image output)
        """

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

        return out_arr

    def sample_predictions(self,
                           dataarray,
                           picklefile=None,
                           outfile=None):
        """
        Get tree predictions from the RF classifier
        :param dataarray: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        """

        # calculate variance of tree predictions
        y = self.predict(np.array(dataarray['features']))

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(dataarray['labels'], y))

        # r-squared of predicted versus actual
        rsq = r2_score(dataarray['labels'], y)

        # if either one of outfile or pickle file are available
        # then raise error
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")

        # if outfile and pickle file are both available
        # then write to file and proceed to return
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in dataarray['labels']]),
                        'mean_y,' + ', '.join([str(elem) for elem in y]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(rsq),
                        'rf_file,' + picklefile]

            # write the list to file
            Handler(filename=outfile).write_list_to_file(out_list)

        # if outfile and pickle file are not provided,
        # then only return values
        return {
            'mean_y': y,
            'obs_y': dataarray['labels'],
            'rmse': rmse,
            'rsq': rsq
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
                print_str_list.append("Estmators: {}\n".format(len(self.classifier.estimators_)))

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
                output='mean',
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
                       choices: ['sd', 'var', 'pred']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'pred' is for prediction or mean of tree outputs
        :return: 1d image array (that will need reshaping if image output)
        """

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

    def sample_predictions(self,
                           dataarray,
                           picklefile=None,
                           outfile=None):
        """
        Get tree predictions from the RF classifier
        :param dataarray: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        """

        # calculate variance of tree predictions
        var_y = self.predict(np.array(dataarray['features']),
                             output='var')

        # calculate mean of tree predictions
        mean_y = self.predict(np.array(dataarray['features']),
                              output='pred')

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(dataarray['labels'], mean_y))

        # r-squared of predicted versus actual
        rsq = r2_score(dataarray['labels'], mean_y)

        # if either one of outfile or pickle file are available
        # then raise error
        if (outfile is not None) != (picklefile is not None):
            raise ValueError("Missing outfile or picklefile")

        # if outfile and pickle file are both available
        # then write to file and proceed to return
        elif outfile is not None:
            # write y, y_hat_bar, var_y to file (<- rows in this order)
            out_list = ['obs_y,' + ', '.join([str(elem) for elem in dataarray['labels']]),
                        'mean_y,' + ', '.join([str(elem) for elem in mean_y]),
                        'var_y,' + ', '.join([str(elem) for elem in var_y]),
                        'rmse,' + str(rmse),
                        'rsq,' + str(rsq),
                        'rf_file,' + picklefile]

            # write the list to file
            Handler(filename=outfile).write_list_to_file(out_list)

        # if outfile and pickle file are not provided,
        # then only return values
        return {
            'var_y': var_y,
            'mean_y': mean_y,
            'obs_y': dataarray['labels'],
            'rmse': rmse,
            'rsq': rsq
        }

    def var_importance(self):
        """
        Return list of tuples of band names and their importance
        """

        return [(band, importance) for band, importance in
                zip(self.data['feature_names'], self.classifier.feature_importances_)]

