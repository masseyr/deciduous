from common import Handler, Sublist
from classification import _Regressor
from exceptions import ObjectNotFound
import numpy as np
# import pandas as pd
import matplotlib
import warnings
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.interpolate import interpn
from scipy.signal import savgol_filter
import statsmodels.api as sm
import statsmodels.formula.api as smf

# plt.rcParams['agg.path.chunksize'] = 100000
# plt.rcParams["patch.force_edgecolor"] = True
# plt.interactive(False)
plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})
plt.rcParams['axes.labelweight'] = 'bold'


"""
plot_dictionary = {
    'type': None,  # plot types: histogram, boxwhisker, regression
    'names': None,  # list of names of 2d array columns
    'xvar': None,  # variable on the x axis
    'yvar': None,  # variable on the y axis
    'data': None,  # 2d array containing all data
    'bins': None,  # list or 1d array for box whisker plot binning
    'xlim': None,  # list of two elements (min, max) of plot, tuple of two elements
    'ylim': None,  # list of two elements(min, max) of plot, tuple of two elements
    'xlabel': None,  # x axis labels
    'ylabel': None,  # y axis labels
    'xtitle': None,  # title of x axis
    'ytitle': None,  # title of y axis
    'title': None,  # plot title
    'text': None,  # text to display in plot
    'text_loc': None,  # location of text on plot
    'datafile': None,  # data file name
    'plotfile': None,  # output plot file name
    'color': None,  # plot color
    'group_names': None  # group names for boxwhisker plots
    'show_values': None  # to show values in heatmap
    'heat_range': None  # to show range of heat values in heatmap
    'color_str': None  # color string for heatmap colors
    'label_colname': None  # column name in datafile to ignore
}
"""


class Plot:
    """
    Class to plot data
    """
    def __init__(self, in_dict):
        """
        Initialize the plot class
        :param in_dict: Input dictionary of parameters
        """
        self.dict = in_dict

        if 'plotfile' in self.dict:
            self.filename = self.dict['plotfile']
        else:
            self.filename = None

    def __repr__(self):
        if 'type' in self.dict:
            return "<Plot object of type {}>".format(self.dict['type'])
        else:
            return "<Plot object -empty- >"

    def draw(self):
        """
        Draw a plot object based on its type in plot dictionary
        :return: Plot object
        """

        if 'type' in self.dict:
            if self.dict['type'] == 'histogram':
                plot = self.histogram()
        #    elif self.dict['type'] == 'boxwhisker':
        #        plot = self.boxwhisker()
        #    elif self.dict['type'] == 'mheatmap':
        #        plot = self.matrix_heatmap()
            elif self.dict['type'] == 'rheatmap':
                plot = self.regression_heatmap()
            elif self.dict['type'] == 'rheatmap2':
                plot = self.regression_heatmap2()
            else:
                plot = None

            if plot is not None:
                # save to file or show in window
                if self.filename is not None:
                    self.filename = Handler(self.dict['plotfile']).file_remove_check()
                    plot.savefig(self.filename, dpi=1200)
                    plot.close()
                else:
                    plot.show()

            else:
                raise ObjectNotFound

        else:
            raise ValueError("Plot type not found in dictionary")

    def histogram(self):
        """
        Makes plot object of type histogram
        :return: Plot object
        """

        # data
        data = self.dict['data']
        dataset = [elem[self.dict['xvar']] for elem in data]

        # color
        if 'color' not in self.dict:
            color = 'green'
        else:
            color = self.dict['color']

        if 'binwidth' in self.dict:
            binwidth = self.dict['binwidth']
        elif 'nbins' in self.dict:
            binwidth = float(max(dataset)-min(dataset))/float(self.dict['nbins'])
        else:
            binwidth = float(max(dataset)-min(dataset))/float(10.0)

        # add axis limits
        if 'xlim' in self.dict:
            xlim = list(self.dict['xlim'])
        else:
            xlim = [min(dataset) - (max(dataset)-min(dataset)) * 0.05,
                    max(dataset) + (max(dataset)-min(dataset)) * 0.05]

        if 'ylim' in self.dict:
            ylim = list(self.dict['ylim'])
        else:
            max_freq = 0

            # find bin count
            for i in Sublist.frange(xlim[0], xlim[1], binwidth):

                count = Sublist(dataset).count_in_range(i, i + binwidth)

                if count > max_freq:
                    max_freq = count

            # find ylim
            ylim = [0, max_freq*1.05]

        # plot histogram
        plt.hist(dataset,
                 alpha=0.8,
                 color=color,
                 ec='k',
                 bins=Sublist.frange(xlim[0], xlim[1], binwidth))

        # add labels
        if 'xtitle' in self.dict:
            plt.xlabel(self.dict['xtitle'])
        else:
            plt.xlabel('x_variable')

        if 'ytitle' in self.dict:
            plt.ylabel(self.dict['ytitle'])
        else:
            plt.ylabel('Count')

        # add plot title
        if 'title' in self.dict:
            plt.title(self.dict['title'])
        elif 'ytitle' in self.dict:
            plt.title(self.dict['ytitle'])
        else:
            plt.title('Histogram')

        # add text in plot
        if ('text' in self.dict) == ('text_loc' in self.dict) and ('text' in self.dict):
            plt.text(self.dict['text_loc'][0],
                     self.dict['text_loc'][1],
                     self.dict['text'])

        # plot with default
        plt.axis(xlim + ylim)

        return plt

    def boxwhisker(self):
        """
        Make plot object of type bowhisker
        :return: Plot object
        """

        # get data and the column names
        data = self.dict['data']
        names = self.dict['names']

        # variables on x and y axes
        xvar = self.dict['xvar']
        yvar = self.dict['yvar']

        # column indices
        x_index = next(i for i in range(0, len(names)) if names[i] == xvar)
        y_index = next(i for i in range(0, len(names)) if names[i] == yvar)

        # get data to be plotted
        out_data = np.array([[elem[x_index], elem[y_index]] for elem in data])

        # bins to plot box whisker plot
        if 'bins' in self.dict:
            bins = self.dict['bins']

            # name of the bin groups
            group_names = self.dict['group_names']

            # initialize lists
            pdata = list()
            gname = list()
            # data bining
            for i in range(0, len(group_names)):
                temp = out_data[(bins[i] <= out_data[:, 0]) & (out_data[:, 0] < bins[i + 1]), 1]
                for j in range(0, len(temp)):
                    pdata.append(temp[j])
                    gname.append(group_names[i])



        # wideform list of lists with bins
        df = [gname, pdata]

        f, (ax) = plt.subplots(1, 1, figsize=(8, 4))
        f.suptitle(self.dict['title'], fontsize=14)

        sns.boxplot(data=df, ax=ax, color=self.dict['color'], showfliers=False)
        ax.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
        ax.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

        return plt

    '''

    def matrix_heatmap(self):
        """
        Make plot object of type heatmap
        :return: Plot object
        """

        if 'data' in self.dict:
            corr = self.dict['data']
            nvar = corr.shape[0]

        else:
            raise ObjectNotFound("No data found")

        if 'xlabel' in self.dict:
            var_names = self.dict['xlabel']
        else:
            var_names = list(str(i+1) for i in range(0, nvar))

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask, k=0)] = True

        xticklabels = var_names[:-1] + ['']
        yticklabels = [''] + var_names[1:]

        font = FontProperties()
        font.set_family('Times')
        font.set_style('normal')
        font.set_variant('normal')
        font.set_weight('bold')
        font.set_size('small')

        sns.set(font=font.get_fontconfig_pattern())
        sns.set(font_scale=2)

        plot_dict = dict()
        plot_dict['mask'] = mask
        plot_dict['xticklabels'] = xticklabels
        plot_dict['yticklabels'] = yticklabels

        if 'show_values' in self.dict:
            plot_dict['annot'] = self.dict['show_values']
            plot_dict['annot_kws'] = {"size": 16}

        if 'heat_range' in self.dict:
            plot_dict['vmin'] = self.dict['heat_range'][0]
            plot_dict['vmax'] = self.dict['heat_range'][1]

        if 'color_str' in self.dict:
            plot_dict['cmap'] = self.dict['color_str']

        plt.figure(figsize=(20, 18))

        with sns.axes_style("white"):
            ax = sns.heatmap(corr,
                             square=True,
                             **plot_dict)

            if 'title' in self.dict:
                plt.title(self.dict['title'])

        return plt
        '''

    def regression_heatmap(self):

        if 'points' in self.dict:
            x = [pt[0] for pt in self.dict['points']]
            y = [pt[1] for pt in self.dict['points']]
        else:
            raise ValueError('No data found')

        if 'xlim' in self.dict:
            xlim = self.dict['xlim']
            xloc = list(k for k in range(0, len(self.dict['points'])) if xlim[0] <= x[k] <= xlim[1])

        else:
            xloc = list(range(0, len(self.dict['points'])))
            xlim = (min(x), max(x))

        if 'ylim' in self.dict:
            ylim = self.dict['ylim']
            yloc = list(k for k in range(0, len(self.dict['points'])) if ylim[0] <= y[k] <= ylim[1])

        else:
            yloc = list(range(0, len(self.dict['points'])))
            ylim = (min(y), max(y))

        loc = list(set(xloc) & set(yloc))

        points_ = list(self.dict['points'][k] for k in loc)

        x_ = list(pt[0] for pt in points_)
        y_ = list(pt[1] for pt in points_)

        if 'xbins' in self.dict and 'ybins' in self.dict:
            xbins = self.dict['xbins']
            ybins = self.dict['ybins']
        elif 'ybins' in self.dict and 'xbins' not in self.dict:
            ybins = self.dict['ybins']
            xbins = self.dict['ybins']
        elif 'xbins' in self.dict and 'ybins' not in self.dict:
            ybins = self.dict['xbins']
            xbins = self.dict['xbins']
        else:
            xbins = 50
            ybins = 50

        if 'xbinvals' in self.dict and 'ybinvals' in self.dict:
            zi, xedges, yedges = np.histogram2d(x_, y_, bins=[self.dict['xbinvals'],
                                                              self.dict['ybinvals']])
        else:
            zi, xedges, yedges = np.histogram2d(x_, y_, bins=[xbins, ybins])

        xi, yi = np.meshgrid(xedges, yedges)

        if 'color' in self.dict:
            color = self.dict['color']
        else:
            color = plt.cm.ocean_r

        plt.pcolormesh(xi, yi, zi, cmap=color, vmin=zi.min(), vmax=int(0.8*zi.max()))

        if 'xlabel' in self.dict:
            xlabel = self.dict['xlabel']
        else:
            xlabel = 'x'

        if 'ylabel' in self.dict:
            ylabel = self.dict['ylabel']
        else:
            ylabel = 'y'

        if 'title' in self.dict:
            title = self.dict['title']
        else:
            title = 'Regression heatmap'

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if 'color_bar_label' in self.dict:
            plt.colorbar().set_label(self.dict['color_bar_label'])

        if 'line' in self.dict:
            if self.dict['line']:
                fit = np.polyfit(x_, y_, 1)
                print('Fitting coefficients: ' + ' '.join([str(i) for i in fit]))
                plt.plot(x_, [fit[0] * x_[i] + fit[1] for i in range(0, len(x_))], 'r-', lw=0.5)

                res = _Regressor.linear_regress(x_, y_, xlim, ylim)
                rsq = res['rsq']

                if 'legend' in self.dict:
                    if self.dict['legend']:
                        out_str = 'Y = {:3.2f} x + {:3.2f}'.format(fit[0], fit[1]) + \
                            '\n$R^2$ = {:3.2f}'.format(rsq) +\
                            '\nN = {:,}'.format(len(x_))

                        plt.text(0.05*xlim[1], 0.95*ylim[1], out_str,
                                 horizontalalignment='left',
                                 verticalalignment='top',)
        plt.xlim(xlim)
        plt.ylim(ylim)

        return plt

    def regression_heatmap2(self):

        plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})
        plt.rcParams['axes.labelweight'] = 'bold'

        if 'bins' in self.dict:
            bins = self.dict['bins']
        else:
            bins = (500, 500)

        if 'data' in self.dict:
            x_p = [pt[0] for pt in self.dict['data']]
            y_p = [pt[1] for pt in self.dict['data']]
        else:
            raise ValueError('No data found')

        if 'xlim' in self.dict:
            xlim = self.dict['xlim']
            xloc = list(k for k in range(0, len(self.dict['data'])) if xlim[0] <= x_p[k] <= xlim[1])
        else:
            xloc = list(range(0, len(self.dict['data'])))
            xlim = [min(x_p) - (max(x_p)-min(x_p)) * 0.05,
                    max(x_p) + (max(x_p)-min(x_p)) * 0.05]

        if 'ylim' in self.dict:
            ylim = self.dict['ylim']
            yloc = list(k for k in range(0, len(self.dict['data'])) if ylim[0] <= y_p[k] <= ylim[1])
        else:
            yloc = list(range(0, len(self.dict['data'])))
            ylim = [min(y_p) - (max(y_p)-min(y_p)) * 0.05,
                    max(y_p) + (max(y_p)-min(y_p)) * 0.05]

        loc = list(set(xloc) & set(yloc))

        data_ = list(self.dict['data'][k] for k in loc)

        x_ = list(pt[0] for pt in data_)
        y_ = list(pt[1] for pt in data_)

        if 'fill_value' in self.dict:
            fill_value = self.dict['fill_value']
        else:
            fill_value = 0

        data, x_e, y_e = np.histogram2d(x_, y_, bins=bins)

        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x_p, y_p]).T,
                    method="splinef2d",
                    bounds_error=False,
                    fill_value=fill_value)

        # Sort the points by density, so that the densest points are plotted last
        idx = (np.array(z).argsort()).tolist()

        x = list(x_p[i] for i in idx)
        y = list(y_p[i] for i in idx)
        z = list(z[i] for i in idx)

        if 'figsize' in self.dict:
            plt.figure(figsize=self.dict['figsize'])
        else:
            plt.figure(figsize=(6, 5))

        zorder = 0
        if 'color' in self.dict:
            color = self.dict['color']
        else:
            color = z

        plt.scatter(x, y, c=color, alpha=0.2, s=2, zorder=zorder)
        zorder += 1

        if 'line' in self.dict:
            if self.dict['line']:
                res = _Regressor.linear_regress(x, y, xlim, ylim)
                rsq = res['rsq']

                if 'plot_quantiles' in self.dict:
                    if len(self.dict['plot_quantiles']) > 0:
                        q_results = _Regressor.quantile_regress(x_, y_, self.dict['plot_quantiles'],
                                                                print_summary=True)
                        for q_result in q_results:
                            print('Fitting coefficients: ' + ' '.join(
                                [str(i) for i in (q_result['slope'], q_result['intercept'])]))
                            plt.plot(x_, [q_result['slope'] * x_[i] + q_result['intercept'] for i in range(0, len(x_))],
                                     'r-', linestyle='dashed', lw=1, zorder=zorder)
                            zorder += 1
                print('Fitting coefficients: ' + ' '.join([str(i) for i in (res['slope'], res['intercept'])]))
                plt.plot(x_, [res['slope'] * x_[i] + res['intercept'] for i in range(0, len(x_))], 'r-', lw=1.5,
                         zorder=zorder)
                zorder += 1

                if 'legend' in self.dict:
                    if self.dict['legend']:
                        if res['intercept'] >= 0:
                            out_str = 'Y = {:3.2f} x + {:3.2f}'.format(res['slope'], res['intercept']) + \
                                '\n$R^2$ = {:3.2f}'.format(rsq) +\
                                '\nN = {:,}'.format(len(x_))
                        else:
                            out_str = 'Y = {:3.2f} x - {:3.2f}'.format(res['slope'], -1.0*res['intercept']) + \
                                '\n$R^2$ = {:3.2f}'.format(rsq) + \
                                '\nN = {:,}'.format(len(x_))

                        rect = patches.Rectangle((0.04*xlim[1], 0.96*ylim[1]), 0.4*xlim[1], -0.2*ylim[1],
                                                 linewidth=1, fill=True, alpha=0.8,
                                                 edgecolor='black', facecolor='whitesmoke')

                        # Add the patch to the Axes
                        plt.gca().add_patch(rect)
                        zorder += 1
                        plt.text(0.05 * xlim[1], 0.95 * ylim[1], out_str,
                                 horizontalalignment='left',
                                 verticalalignment='top',
                                 zorder=zorder)
                        zorder += 1

        elif 'poly1d' in self.dict:
            if self.dict['poly1d']:
                if 'poly_degree' in self.dict:
                    poly_degree = self.dict['poly_degree']
                else:
                    warnings.warn('No polynomial degree specified. Using polynomial degree 2')
                    poly_degree = 2

                res = np.polyfit(x, y, poly_degree)
                p = np.poly1d(res)

                print('Fitting coefficients: ' + ' '.join([str(i) for i in res]))
                plt.plot(x_, [p(x_[i]) for i in range(0, len(x_))], 'r-', lw=1.5)

                if 'legend' in self.dict:
                    if self.dict['legend']:
                        rhs = list()
                        for i, coeff in enumerate(res):
                            if i != (len(res)-1):
                                rhs.append('{:3.3f} $x^{}$'.format(coeff, (len(res)-i-1)))
                            else:
                                rhs.append('{:3.2f}'.format(coeff))

                        out_str = 'Y = {}'.format(' + '.join(rhs))

                        rect = patches.Rectangle((0.04*xlim[1], 0.96*ylim[1]), 0.4*xlim[1], -0.2*ylim[1],
                                                 linewidth=1, fill=True, alpha=0.8,
                                                 edgecolor='black', facecolor='whitesmoke')

                        # Add the patch to the Axes
                        plt.gca().add_patch(rect)

                        plt.text(0.05 * xlim[1], 0.95 * ylim[1], out_str,
                                 horizontalalignment='left',
                                 verticalalignment='top', )

        plt.xlim(xlim)
        plt.ylim(ylim)

        return plt

