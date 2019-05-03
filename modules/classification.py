import pickle
import numpy as np
from scipy import stats
from math import sqrt
from osgeo import gdal, gdal_array
from common import *
from raster import Raster
from timer import Timer
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


__all__ = ['RFRegressor',
           'MRegressor']


sep = Handler().sep


class _Regressor(object):

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 **kwargs):
        self.data = data
        self.vdata = None
        self.regressor = regressor
        self.features = None
        self.feature_index = None
        self.label = None
        self.output = None
        self.training_results = dict()
        self.fit = False
        self.all_cv_results = None

        self.adjustment = dict()

        if kwargs is not None:
            if 'timer' in kwargs:
                _Regressor.time_it = kwargs['timer']

    def __repr__(self):
        return "<Classifier base class>"

    def fit_data(self,
                 data,
                 use_weights=False):
        """
        Train the regressor
        :param data: dictionary with values (generated using Samples.format_data())
        :param use_weights: If the sample weights provided should be used? (default: False)
        :return: Nonetype
        """
        self.data = data

        if self.regressor is not None:

            if 'weights' not in data or not use_weights:
                self.regressor.fit(data['features'], data['labels'])
            else:
                self.regressor.fit(data['features'], data['labels'], data['weights'])

        self.features = data['feature_names']
        self.label = data['label_name']
        self.fit = True

    def predict(self, *args, **kwargs):
        """Placeholder function"""
        return

    def pickle_it(self,
                  outfile):
        """
        Save regressor
        :param outfile: File to save the regressor to
        """
        outfile = Handler(filename=outfile).file_remove_check()
        with open(outfile, 'wb') as fileptr:
            pickle.dump(self, fileptr)

    @classmethod
    def load_from_pickle(cls,
                         infile):
        """
        Reload regressor from file
        :param infile: File to load regressor from
        """
        with open(infile, 'rb') as fileptr:
            regressor_obj = pickle.load(fileptr)
            return regressor_obj

    @staticmethod
    @Timer.timing(time_it)
    def regress_raster(regressor,
                       raster_obj,
                       outfile=None,
                       outdir=None,
                       band_name='prediction',
                       output_type='median',
                       array_multiplier=1.0,
                       array_additive=0.0,
                       out_data_type=gdal.GDT_Float32,
                       nodatavalue=-32768.0,
                       **kwargs):

        """Tree variance from the RF regressor
        :param regressor: _Regressor object
        :param raster_obj: Initialized Raster object with a 3d array
        :param outfile: name of output file
        :param array_multiplier: rescale data using this value
        :param array_additive: Rescale data using this value
        :param out_data_type: output raster data type
        :param nodatavalue: No data value for output raster
        :param band_name: Name of the output raster band
        :param outdir: output folder
        :param output_type: Should the output be standard deviation ('sd'),
                            variance ('var'), or median ('median'), or mean ('mean')
                            or 'conf' for confidence interval
        :returns: Output as raster object
        """
        if 'z_val' in kwargs:
            z_val = kwargs['z_val']
        else:
            z_val = 1.96

        if 'band_multiplier' in kwargs:
            band_multiplier = kwargs['band_multiplier']
        else:
            band_multiplier = zip(list((elem, 1.0) for elem in raster_obj.bnames))

        if 'out_nodatavalue' in kwargs:
            out_nodatavalue = kwargs['out_nodatavalue']
        else:
            out_nodatavalue = nodatavalue

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

        if regressor.feature_index is None:
            regressor.feature_index = list(raster_obj.bnames.index(feat) for feat in regressor.features)

        # reshape into a long 2d array (nband, nrow * ncol) for classification,
        new_shape = [nbands, nrows * ncols]

        multiplier = np.array([band_multiplier[elem] if elem in band_multiplier else 1.0
                               for elem in raster_obj.bnames])

        def _with_data(pixel_vec):
            pixel_vec[np.where(pixel_vec != nodatavalue)] = \
                pixel_vec[np.where(pixel_vec != nodatavalue)] * multiplier[np.where(pixel_vec != nodatavalue)]
            return pixel_vec

        temp_arr = np.apply_along_axis(_with_data,
                                       0,
                                       raster_obj.array.astype(gdal_array.GDALTypeCodeToNumericTypeCode(out_data_type)))

        temp_arr = temp_arr.reshape(new_shape) * array_multiplier + array_additive
        temp_arr = temp_arr.swapaxes(0, 1)

        # apply the variance calculating function on the array
        out_arr = regressor.predict(temp_arr,
                                    output_type=output_type,
                                    z_val=z_val,
                                    nodatavalue=nodatavalue,
                                    out_nodatavalue=out_nodatavalue)

        # output raster and metadata
        if out_data_type != gdal_array.NumericTypeCodeToGDALTypeCode(out_arr.dtype):
            out_arr = out_arr.astype(gdal_array.GDALTypeCodeToNumericTypeCode(out_data_type))

        out_ras.dtype = out_data_type
        out_ras.transform = raster_obj.transform
        out_ras.crs_string = raster_obj.crs_string

        out_ras.array = out_arr.reshape([nrows, ncols])
        out_ras.shape = [1, nrows, ncols]
        out_ras.bnames = [band_name]
        out_ras.nodatavalue = out_nodatavalue

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
            value_arr = list(xlim[0] <= elem <= xlim[1] for elem in x)
            if not any(value_arr):
                raise ValueError("Minimum or maximum argument for regression outside data limits")

            x_index_list = Sublist(x).range(*xlim,
                                            index=True)
            x = list(x[i] for i in x_index_list)
            y = list(y[i] for i in x_index_list)

        if ylim is not None:
            value_arr = list(ylim[0] <= elem <= ylim[1] for elem in y)
            if not any(value_arr):
                raise ValueError("Minimum or maximum argument for regression outside data limits")

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


class MRegressor(_Regressor):
    """Multiple linear regressor object for scikit-learn linear model"""

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 intercept=True,
                 jobs=1,
                 normalize=False,
                 **kwargs):

        super(MRegressor, self).__init__(data,
                                         regressor)
        if self.regressor is None:
            self.regressor = linear_model.LinearRegression(copy_X=True,
                                                           fit_intercept=intercept,
                                                           n_jobs=jobs,
                                                           normalize=normalize)

        self.intercept = self.regressor.intercept_ if hasattr(self.regressor, 'intercept_') else None
        self.coefficient = self.regressor.coef_ if hasattr(self.regressor, 'coef_') else None

        if kwargs is not None:
            if 'timer' in kwargs:
                MRegressor.time_it = kwargs['timer']

    def __repr__(self):
        # gather which attributes exist
        attr_truth = [hasattr(self.regressor, 'coef_'),
                      hasattr(self.regressor, 'intercept_')]

        if any(attr_truth):

            print_str_list = list("Multiple Linear Regressor:\n")

            # strings to be printed for each attribute
            if attr_truth[0]:
                print_str_list.append("Coefficients: {}\n".format(len(self.regressor.coef_)))

            if attr_truth[1]:
                print_str_list.append("Intercept: {}\n".format(self.regressor.intercept_))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Multiple Linear Regressor: __empty__>"

    @Timer.timing(time_it)
    def predict(self,
                arr,
                ntile_max=5,
                tile_size=1024,
                **kwargs):
        """
        Calculate multiple regression model prediction, variance, or standard deviation.
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
        nodatavalue = None

        if kwargs is not None:
            for key, value in kwargs.items():
                if key in ('gain', 'bias', 'upper_limit', 'lower_limit'):
                    self.adjustment[key] = value

                if key == 'nodatavalue':
                    nodatavalue = value

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
                    self.regressor.predict(arr[i * npx_tile:(i + 1) * npx_tile, self.feature_index])

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 2
                out_arr[i * npx_last:(i + 1) * npx_last] = \
                    self.regressor.predict(arr[i * npx_tile:(i * npx_tile + npx_last), self.feature_index])

        else:
            out_arr = self.regressor.predict(arr[:, self.feature_index])

        if len(self.adjustment) > 0:

            if 'gain' in self.adjustment:
                out_arr = out_arr * self.adjustment['gain']

            if 'bias' in self.adjustment:
                out_arr = out_arr + self.adjustment['bias']

            if 'upper_limit' in self.adjustment:
                out_arr[out_arr > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

            if 'lower_limit' in self.adjustment:
                out_arr[out_arr < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

        output = out_arr

        if nodatavalue is not None:
            for ii in range(arr.shape[0]):
                output[np.unique(np.where(arr[ii, :, :] == nodatavalue)[0])] = nodatavalue

        return output

    def sample_predictions(self,
                           data,
                           picklefile=None,
                           outfile=None,
                           **kwargs):
        """
        Get predictions from the multiple regressor
        :param data: Dictionary object from Samples.format_data
        :param picklefile: Random Forest pickle file
        :param outfile: output csv file name
        """

        self.feature_index = list(data['feature_names'].index(feat) for feat in self.features)

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

    def get_training_fit(self,
                         regress_limit=None):

        """
        Find out how well the training samples fit the model
        :param regress_limit: List of upper and lower regression limits for training fit prediction
        :return: None
        """
        if self.fit:
            # predict using held out samples and print to file
            pred = self.sample_predictions(self.data,
                                           regress_limit=regress_limit)

            self.training_results['rsq'] = pred['rsq'] * 100.0
            self.training_results['slope'] = pred['slope']
            self.training_results['intercept'] = pred['intercept']
            self.training_results['rmse'] = pred['rmse']
        else:
            raise ValueError("Model not initialized with samples")

    def get_adjustment_param(self,
                             clip=0.025,
                             data_limits=None,
                             over_adjust=1.0):
        """
        get the model adjustment parameters based on training fit
        :param clip:
        :param data_limits:
        :param over_adjust

        :return: None
        """

        if data_limits is None:
            data_limits = [max(self.data['labels']), min(self.data['labels'])]

        regress_limit = [clip * data_limits[0],
                         (1.0 - clip) * data_limits[0]]

        self.get_training_fit(regress_limit=regress_limit)

        if self.training_results['intercept'] > regress_limit[0]:
            self.adjustment['bias'] = -1.0 * (self.training_results['intercept'] / self.training_results['slope'])

        self.adjustment['gain'] = (1.0 / self.training_results['slope']) * over_adjust
        self.adjustment['upper_limit'] = data_limits[0]
        self.adjustment['lower_limit'] = data_limits[1]


class RFRegressor(_Regressor):
    """Random Forest Regressor class for scikit-learn Random Forest regressor"""

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 trees=10,
                 samp_split=2,
                 samp_leaf=1,
                 max_depth=None,
                 max_feat='auto',
                 oob_score=False,
                 criterion='mse',
                 n_jobs=1,
                 **kwargs):
        """
        Initialize RF regressor using class parameters
        :param trees: Number of trees
        :param samp_split: Minimum number of samples for split
        :param oob_score: (bool) calculate out of bag score
        :param criterion: criterion to be used (default: 'mse', options: 'mse', 'mae')
        (some parameters are as detailed in
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
        """

        super(RFRegressor, self).__init__(data,
                                          regressor)

        if self.regressor is None:
            self.regressor = RandomForestRegressor(n_estimators=trees,
                                                   max_depth=max_depth,
                                                   min_samples_split=samp_split,
                                                   min_samples_leaf=samp_leaf,
                                                   max_features=max_feat,
                                                   criterion=criterion,
                                                   oob_score=oob_score,
                                                   n_jobs=n_jobs)
        self.trees = trees
        self.max_depth = max_depth
        self.samp_split = samp_split
        self.samp_leaf = samp_leaf
        self.max_feat = max_feat
        self.oob_score = oob_score
        self.criterion = criterion
        self.n_jobs = n_jobs

        self.dec_paths = list()

        if kwargs is not None:
            if 'timer' in kwargs:
                MRegressor.time_it = kwargs['timer']

    def __repr__(self):
        # gather which attributes exist
        attr_truth = [hasattr(self.regressor, 'estimators_'),
                      hasattr(self.regressor, 'n_features_'),
                      hasattr(self.regressor, 'n_outputs_'),
                      hasattr(self.regressor, 'oob_score_')]

        # if any exist print them
        if any(attr_truth):

            print_str_list = list("Random Forest Regressor:\n")

            # strings to be printed for each attribute
            if attr_truth[0]:
                print_str_list.append("Estimators: {}\n".format(len(self.regressor.estimators_)))

            if attr_truth[1]:
                print_str_list.append("Features: {}\n".format(self.regressor.n_features_))

            if attr_truth[2]:
                print_str_list.append("Output: {}\n".format(self.regressor.n_outputs_))

            if attr_truth[3]:
                print_str_list.append("OOB Score: {:{w}.{p}f} %".format(self.regressor.oob_score_ * 100.0,
                                                                        w=3, p=2))

            # combine all strings into one print string
            print_str = ''.join(print_str_list)

            return print_str

        else:
            # if empty return empty
            return "<Random Forest Regressor: __empty__>"

    @staticmethod
    def regress_tile(arr,
                     tile_start,
                     tile_end,
                     regressor,
                     feature_index,
                     npx_tile=1024*1024,
                     nodatavalue=None,
                     output_type='median',
                     intvl=95.0,
                     min_variance=0.01):

        """
        Method to preocess each tile of the image internally
        :param tile_start: pixel location of tile start
        :param tile_end: pixel loation of tile end
        :param arr: input array to process
        :param regressor: RFRegressor
        :param feature_index: List of list of feature indices corresponding to input array
                              i.e. index of bands to be used for regression
        :param npx_tile: numper to pixels in each tile
        :param nodatavalue: No data value
        :param output_type: Type of output to produce,
                       choices: ['sd', 'var', 'pred', 'full', 'conf']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'median' is for median of tree outputs
                       'mean' is for mean of tree coutputs
                       'conf' stands for confidence interval
        :param intvl: Prediction interval width (default: 95 percentile)
        :param min_variance: Minimum variance after which to cutoff
        :return: numpy 1-D array
        """
        if min_variance is None:
            min_variance = 0.05 * np.min(arr.astype(np.float32))

        temp_arr = arr[tile_start:tile_end, feature_index]

        out_tile = np.empty([tile_end - tile_start + 1]) * 0.0

        tile_arr = np.array([list(range(0, npx_tile)) for _ in range(0, regressor.trees)], dtype=float)

        if output_type in ('mean', 'median', 'full'):

            # calculate tree predictions for each pixel in a 2d array
            for jj, tree_ in enumerate(regressor.regressor.estimators_):
                tile_arr[jj, :] = tree_.predict(temp_arr)

            if output_type == 'median':
                out_tile = np.median(tile_arr, axis=0)
            elif output_type == 'mean':
                out_tile = np.mean(tile_arr, axis=0)
            elif output_type == 'full':
                return tile_arr

        elif output_type in ('sd', 'var'):

            for jj, tree_ in enumerate(regressor.regressor.estimators_):
                tile_arr[jj, :] = tree_.predict(temp_arr)

                var_tree = tree_.tree_.impurity[tree_.apply(temp_arr)]

                var_tree[var_tree < min_variance] = min_variance
                mean_tree = tree_.predict(temp_arr)
                out_tile += var_tree + mean_tree ** 2

            predictions = np.mean(tile_arr, axis=0)

            out_tile /= len(regressor.regressor.estimators_)
            out_tile -= predictions ** 2.0
            out_tile[out_tile < 0.0] = 0.0

            if output_type == 'var':
                return out_tile
            elif output_type == 'sd':
                return out_tile ** 0.5
        else:
            raise RuntimeError("Unknown output type or no output type specified")

        if len(regressor.adjustment) > 0:

            if 'gain' in regressor.adjustment:
                out_tile = out_tile * regressor.adjustment['gain']

            if 'bias' in regressor.adjustment:
                out_tile = out_tile + regressor.adjustment['bias']

            if 'upper_limit' in regressor.adjustment:
                out_tile[out_tile > regressor.adjustment['upper_limit']] = regressor.adjustment['upper_limit']

            if 'lower_limit' in regressor.adjustment:
                out_tile[out_tile < regressor.adjustment['lower_limit']] = regressor.adjustment['lower_limit']

        if nodatavalue is not None:
            out_tile[np.unique(np.where(temp_arr == nodatavalue)[0])] = nodatavalue

        return out_tile

    @Timer.timing(time_it)
    def predict(self,
                arr,
                ntile_max=5,
                tile_size=1024,
                output_type='pred',
                intvl=95.0,
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

        :param output_type: which output to produce,
                       choices: ['sd', 'var', 'pred', 'full', 'conf']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'median' is for median of tree outputs
                       'mean' is for mean of tree coutputs
                       'full' is for the full spectrum of the leaf nodes' prediction
                       'conf' stands for confidence interval

        :param intvl: Prediction interval width (default: 95 percentile)

        :param kwargs: Keyword arguments:
                       'gain': Adjustment of the predicted output by linear adjustment of gain (slope)
                       'bias': Adjustment of the predicted output by linear adjustment of bias (intercept)
                       'upper_limit': Limit of maximum value of prediction
                       'lower_limit': Limit of minimum value of prediction

        :return: 1d image array (that will need reshaping if image output)
        """
        nodatavalue = None

        if kwargs is not None:
            for key, value in kwargs.items():
                if key in ('gain', 'bias', 'upper_limit', 'lower_limit'):
                    self.adjustment[key] = value
                if key == 'nodatavalue':
                    nodatavalue = value

        # define output array
        if output_type == 'full':
            out_arr = np.empty([self.trees, arr.shape[0]])
        else:
            out_arr = Opt.__copy__(arr[:, 0]) * 0.0

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

                Opt.cprint('Processing tile {} of {}'.format(str(i+1), ntiles))

                if output_type == 'full':
                    out_arr[:, i * npx_tile:(i + 1) * npx_tile] = self.regress_tile(arr,
                                                                                    i * npx_tile,
                                                                                    (i + 1) * npx_tile,
                                                                                    self,
                                                                                    self.feature_index,
                                                                                    npx_tile=npx_tile,
                                                                                    nodatavalue=nodatavalue,
                                                                                    output_type=output_type,
                                                                                    intvl=intvl)
                else:
                    out_arr[i * npx_tile:(i + 1) * npx_tile] = self.regress_tile(arr,
                                                                                 i * npx_tile,
                                                                                 (i + 1) * npx_tile,
                                                                                 self,
                                                                                 self.feature_index,
                                                                                 npx_tile=npx_tile,
                                                                                 nodatavalue=nodatavalue,
                                                                                 output_type=output_type,
                                                                                 intvl=intvl)

            if npx_last > 0:  # number of total pixels for the last tile

                i = ntiles - 2

                Opt.cprint('Processing tile {} of {}'.format(str(i+2), ntiles))

                if output_type == 'full':
                    out_arr[:, i * npx_tile:(i * npx_tile + npx_last)] = self.regress_tile(arr,
                                                                                           i * npx_tile,
                                                                                           (i + 1) * npx_tile,
                                                                                           self,
                                                                                           self.feature_index,
                                                                                           npx_tile=npx_tile,
                                                                                           nodatavalue=nodatavalue,
                                                                                           output_type=output_type,
                                                                                           intvl=intvl)
                else:
                    out_arr[i * npx_tile:(i * npx_tile + npx_last)] = self.regress_tile(arr,
                                                                                        i * npx_tile,
                                                                                        (i + 1) * npx_tile,
                                                                                        self,
                                                                                        self.feature_index,
                                                                                        npx_tile=npx_tile,
                                                                                        nodatavalue=nodatavalue,
                                                                                        output_type=output_type,
                                                                                        intvl=intvl)

        else:

            out_arr = self.regress_tile(arr,
                                        0,
                                        npx_inp - 1,
                                        self,
                                        self.feature_index,
                                        npx_tile=npx_tile,
                                        nodatavalue=nodatavalue,
                                        output_type=output_type,
                                        intvl=intvl)

        if output_type == 'full':
            if nodatavalue is not None:
                out_arr[:, np.unique(np.where(arr == nodatavalue)[0])] = nodatavalue
        else:
            if len(self.adjustment) > 0:

                if 'gain' in self.adjustment:
                    out_arr = out_arr * self.adjustment['gain']

                if 'bias' in self.adjustment:
                    out_arr = out_arr + self.adjustment['bias']

                if 'upper_limit' in self.adjustment:
                    out_arr[out_arr > self.adjustment['upper_limit']] = self.adjustment['upper_limit']

                if 'lower_limit' in self.adjustment:
                    out_arr[out_arr < self.adjustment['lower_limit']] = self.adjustment['lower_limit']

            if nodatavalue is not None:
                out_arr[np.unique(np.where(arr == nodatavalue)[0])] = nodatavalue

        return out_arr

    def sample_predictions(self,
                           data,
                           picklefile=None,
                           outfile=None,
                           **kwargs):
        """
        Get tree predictions from the RF regressor
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
                if key in ('gain', 'bias', 'upper_limit', 'lower_limit'):
                    self.adjustment[key] = value

        self.feature_index = list(data['feature_names'].index(feat) for feat in self.features)

        if 'regress_limit' in kwargs:
            regress_limit = kwargs['regress_limit']
        else:
            regress_limit = None

        # calculate variance of tree predictions
        var_y = None
        if 'var_y' in kwargs:
            if kwargs['var_y']:
                var_y = self.predict(np.array(data['features']),
                                     output='var')

        # calculate mean of tree predictions
        all_y = None
        if 'all_y' in kwargs:
            if kwargs['all_y']:
                all_y = self.predict(np.array(data['features']),
                                     output='full')

        # calculate sd of tree predictions
        sd_y = None
        if 'sd_y' in kwargs:
            if kwargs['sd_y']:
                sd_y = self.predict(np.array(data['features']),
                                    output='sd')

        # calculate sd of tree predictions
        se_y = None
        if 'se_y' in kwargs:
            if kwargs['se_y']:
                se_y = self.predict(np.array(data['features']),
                                    output='se')

        conf_y = None
        if 'conf_y' in kwargs:
            if kwargs['conf_y']:
                if 'intvl' in kwargs:
                    intvl = kwargs['intvl']
                else:
                    intvl = 95.0
                conf_y = self.predict(np.array(data['features']),
                                      intvl=intvl,
                                      output='conf')

        # calculate mean of tree predictions
        pred_y = self.predict(np.array(data['features']),
                              output='pred')

        # rms error of the predicted versus actual
        rmse = sqrt(mean_squared_error(data['labels'], pred_y))

        # r-squared of predicted versus actual
        if regress_limit is not None:
            lm = self.linear_regress(data['labels'],
                                     pred_y,
                                     xlim=regress_limit)
        else:
            lm = self.linear_regress(data['labels'],
                                     pred_y)

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

            if sd_y is not None:
                out_list.append('sd_y,' + ', '.join([str(elem) for elem in sd_y]))

            if conf_y is not None:
                out_list.append('conf_y,' + ', '.join([str(elem) for elem in conf_y]))

            if se_y is not None:
                out_list.append('se_y,' + ', '.join([str(elem) for elem in se_y]))

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
            'intercept': lm['intercept'],
        }

        if all_y is not None:
            out_dict['all_y'] = all_y
        if var_y is not None:
            out_dict['var_y'] = var_y
        if sd_y is not None:
            out_dict['sd_y'] = sd_y
        if conf_y is not None:
            out_dict['conf_y'] = conf_y
        if se_y is not None:
            out_dict['se_y'] = se_y

        return out_dict

    def var_importance(self):
        """
        Return list of tuples of band names and their importance
        """

        return [(band, importance) for band, importance in
                zip(self.data['feature_names'], self.regressor.feature_importances_)]

    def get_training_fit(self,
                         regress_limit=None):

        """
        Find out how well the training samples fit the model
        :param regress_limit: List of upper and lower regression limits for training fit prediction
        :return: None
        """
        if self.fit:
            # predict using held out samples and print to file
            pred = self.sample_predictions(self.data,
                                           regress_limit=regress_limit)

            self.training_results['rsq'] = pred['rsq'] * 100.0
            self.training_results['slope'] = pred['slope']
            self.training_results['intercept'] = pred['intercept']
            self.training_results['rmse'] = pred['rmse']
        else:
            raise ValueError("Model not initialized with samples")

    def get_adjustment_param(self,
                             clip=0.025,
                             data_limits=None,
                             over_adjust=1.0):
        """
        get the model adjustment parameters based on training fit
        :param clip: Ratio of samples not to be used at each tailend
        :param data_limits: Limits of output data
        :param over_adjust: Amount of over adjustment needed to slope
        :return: None
        """

        if data_limits is None:
            data_limits = [max(self.data['labels']), min(self.data['labels'])]

        regress_limit = [clip * data_limits[0],
                         (1.0-clip) * data_limits[0]]

        self.get_training_fit(regress_limit=regress_limit)

        if self.training_results['intercept'] > regress_limit[0]:
            self.adjustment['bias'] = -1.0 * (self.training_results['intercept'] / self.training_results['slope'])

        self.adjustment['gain'] = (1.0 / self.training_results['slope'])*over_adjust
        self.adjustment['upper_limit'] = data_limits[0]
        self.adjustment['lower_limit'] = data_limits[1]


class HRFRegressor(RFRegressor):

    """
    Hierarchical Random Forest Regressor
    """

    time_it = False

    def __init__(self,
                 data=None,
                 regressor=None,
                 **kwargs):

        super(RFRegressor, self).__init__(data,
                                          regressor)
        if type(regressor).__name__ not in ('list', 'tuple'):
            regressor = [regressor]

        if type(data).__name__ not in ('list', 'tuple'):
            data = [data]

        feature_list_ = list(reg.features for reg in regressor)
        feature_index_ = list(reversed(sorted(range(len(feature_list_)),
                                              key=lambda x: len(feature_list_[x]))))

        self.regressor = list(regressor[idx] for idx in feature_index_)
        self.data = list(data[idx] for idx in feature_index_)
        self.features = list(feature_list_[idx] for idx in feature_index_)
        self.feature_index = None

    def __repr__(self):

        if self.regressor is None:
            repr_regressor = ['<empty>']
        elif type(self.regressor).__name__ in ('list', 'tuple'):
            repr_regressor = list(regressor.__repr__() for regressor in self.regressor)
        else:
            repr_regressor = [self.regressor.__repr__()]

        return "Hierarchical regressor object" + \
            "\n---\n  Regressors: \n{}".format(list('\n'.join(repr_regressor))) + \
            "\n---"

    @Timer.timing(time_it)
    def regress_raster(self,
                       raster_obj,
                       outfile=None,
                       outdir=None,
                       band_name='prediction',
                       output_type='median',
                       out_data_type=gdal.GDT_Float32,
                       nodatavalue=-32768.0,
                       **kwargs):

        """Tree variance from the RF regressor
        :param raster_obj: Initialized Raster object with a 3d array
        :param outfile: name of output file
        :param out_data_type: output raster data type
        :param nodatavalue: No data value for output raster
        :param band_name: Name of the output raster band
        :param outdir: output folder
        :param output_type: Should the output be standard deviation ('sd'),
                            variance ('var'), or prediction ('pred'),
                            or 'conf' for confidence interval
        :returns: Output as raster object
        """
        self.feature_index = list(list(raster_obj.bnames.index(feat) for feat in feat_grp)
                                  for feat_grp in self.features)

        _Regressor.regress_raster(self,
                                  raster_obj,
                                  outfile=outfile,
                                  outdir=outdir,
                                  band_name=band_name,
                                  output_type=output_type,
                                  out_data_type=out_data_type,
                                  nodatavalue=nodatavalue,
                                  **kwargs)

    @staticmethod
    def regress_tile(arr,
                     tile_start,
                     tile_end,
                     regressors,
                     feature_index,
                     npx_tile=1024*1024,
                     nodatavalue=None,
                     output_type='median',
                     intvl=95.0,
                     min_variance=None):

        """
        Method to preocess each tile of the image internally
        :param tile_start: pixel location of tile start
        :param tile_end: pixel loation of tile end
        :param arr: input array to process
        :param regressors: RFRegressor
        :param feature_index: List of list of feature indices corresponding to input array
                              i.e. index of bands to be used for regression
        :param npx_tile: numper to pixels in each tile
        :param nodatavalue: No data value
        :param output_type: Type of output to produce,
                       choices: ['sd', 'var', 'pred', 'full', 'conf']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'median' is for median of tree outputs
                       'mean' is for mean of tree coutputs
                       'conf' stands for confidence interval
        :param intvl: Prediction interval width (default: 95 percentile)
        :param min_variance: Minimum variance after which to cutoff
        :return: numpy 1-D array
        """
        if min_variance is None:
            min_variance = 0.05 * np.min(arr.astype(np.float32))

        out_tile = np.empty([tile_end - tile_start + 1]) * 0.0
        if nodatavalue is not None:
            out_tile += nodatavalue

        for ii, regressor in enumerate(regressors):
            Opt.cprint(' . {}'.format(str(ii + 1)))

            temp_arr = arr[tile_start:tile_end, feature_index[ii]]
            tile_arr = np.array([list(range(0, npx_tile)) for _ in range(0, regressor.trees)], dtype=float)

            if output_type in ('mean', 'median', 'full'):

                # calculate tree predictions for each pixel in a 2d array
                for jj, tree_ in enumerate(regressor.regressor.estimators_):
                    tile_arr[jj, :] = tree_.predict(temp_arr)

                if output_type == 'median':
                    out_tile = np.median(tile_arr, axis=0)
                elif output_type == 'mean':
                    out_tile = np.mean(tile_arr, axis=0)
                elif output_type == 'full':
                    return tile_arr

            elif output_type in ('sd', 'var'):

                for jj, tree_ in enumerate(regressor.regressor.estimators_):
                    tile_arr[jj, :] = tree_.predict(temp_arr)

                    var_tree = tree_.tree_.impurity[tree_.apply(temp_arr)]

                    var_tree[var_tree < min_variance] = min_variance
                    mean_tree = tree_.predict(temp_arr)
                    out_tile += var_tree + mean_tree ** 2

                predictions = np.mean(tile_arr, axis=0)

                out_tile /= len(regressor.regressor.estimators_)
                out_tile -= predictions ** 2.0
                out_tile[out_tile < 0.0] = 0.0

                if output_type == 'var':
                    return out_tile
                elif output_type == 'sd':
                    return out_tile ** 0.5
            else:
                raise RuntimeError("Unknown output type or no output type specified")

            if len(regressor.adjustment) > 0:

                if 'gain' in regressor.adjustment:
                    out_tile = out_tile * regressor.adjustment['gain']

                if 'bias' in regressor.adjustment:
                    out_tile = out_tile + regressor.adjustment['bias']

                if 'upper_limit' in regressor.adjustment:
                    out_tile[out_tile > regressor.adjustment['upper_limit']] = regressor.adjustment['upper_limit']

                if 'lower_limit' in regressor.adjustment:
                    out_tile[out_tile < regressor.adjustment['lower_limit']] = regressor.adjustment['lower_limit']

            if nodatavalue is not None:
                out_tile[np.unique(np.where(temp_arr == nodatavalue)[0])] = nodatavalue

        return out_tile

    def predict(self,
                arr,
                ntile_max=5,
                tile_size=1024,
                output_type='median',
                intvl=95.0,
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

        :param output_type: which output to produce,
                       choices: ['sd', 'var', 'pred', 'full', 'conf']
                       where 'sd' is for standard deviation,
                       'var' is for variance
                       'median' is for median of tree outputs
                       'mean' is for mean of tree coutputs
                       'conf' stands for confidence interval

        :param intvl: Prediction interval width (default: 95 percentile)
        :return: 1d image array (that will need reshaping if image output)
        """

        nodatavalue = None

        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'nodatavalue':
                    nodatavalue = value

        # define output array
        out_arr = Opt.__copy__(arr[:, 0]) * 0.0
        if nodatavalue is not None:
            out_arr += nodatavalue

        # input image size
        npx_inp = long(arr.shape[0])  # number of pixels in input image
        nb_inp = long(arr.shape[1])  # number of bands in input image

        # size of tiles
        npx_tile = long(tile_size * tile_size)  # pixels in each tile
        npx_last = npx_inp % npx_tile  # pixels in last tile
        ntiles = long(npx_inp) / long(npx_tile) + 1  # total tiles

        if ntiles > ntile_max:

            for i in range(0, ntiles - 1):

                Opt.cprint('Processing tile {} of {}'.format(str(i + 1), ntiles), newline='')

                out_arr[i * npx_tile:(i + 1) * npx_tile] = self.regress_tile(arr,
                                                                             i * npx_tile,
                                                                             (i + 1) * npx_tile,
                                                                             self.regressor,
                                                                             self.feature_index,
                                                                             npx_tile=npx_tile,
                                                                             nodatavalue=nodatavalue,
                                                                             output_type=output_type,
                                                                             intvl=intvl)

            if npx_last > 0:

                i = ntiles - 2
                Opt.cprint('Processing tile {} of {}'.format(str(i + 2), ntiles), newline='')

                out_arr[i * npx_tile:(i * npx_tile + npx_last)] = self.regress_tile(arr,
                                                                                    i * npx_tile,
                                                                                    i * npx_tile + npx_last,
                                                                                    self.regressor,
                                                                                    self.feature_index,
                                                                                    npx_tile=npx_last,
                                                                                    nodatavalue=nodatavalue,
                                                                                    output_type=output_type,
                                                                                    intvl=intvl)
        else:
            Opt.cprint('Processing image as one tile', newline='')

            out_arr = self.regress_tile(arr,
                                        0,
                                        npx_inp - 1,
                                        self.regressor,
                                        self.feature_index,
                                        npx_tile=npx_last,
                                        nodatavalue=nodatavalue,
                                        output_type=output_type,
                                        intvl=intvl)

        return out_arr
