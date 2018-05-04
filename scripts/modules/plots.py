from common import Handler, Sublist
from exceptions import ObjectNotFound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


plt.rcParams["patch.force_edgecolor"] = True
plt.interactive(False)

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
    def __init__(self, dict):
        """
        Initialize the plot class
        :param dict: Input dictionary of parameters
        """
        self.dict = dict
        if 'plotfile' in self.dict:
            self.filename = Handler(self.dict['plotfile']).file_remove_check()
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
            elif self.dict['type'] == 'boxwhisker':
                plot = self.boxwhisker()
            elif self.dict['type'] == 'mheatmap':
                plot = self.matrix_heatmap()
            elif self.dict['type'] == 'rheatmap':
                plot = self.regression_heatmap()
            else:
                plot = None

            if plot is not None:
                # save to file or show in window
                if self.filename is not None:
                    plot.savefig(self.filename)
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
        names = self.dict['names']
        try:
            indx = next(i for i in range(0, len(names)) if names[i] == self.dict['xvar'])
        except StopIteration:
            print('Histogram variable not found')
            return

        dataset = [elem[indx] for elem in data]

        # color
        if 'color' not in self.dict:
            color = 'green'
        else:
            color = self.dict['color']

        # plot histogram
        plt.hist(dataset,
                 alpha=0.8,
                 color=color)

        # add labels
        if 'xtitle' in self.dict:
            plt.xlabel(self.dict['xtitle'])
        if 'ytitle' in self.dict:
            plt.ylabel(self.dict['ytitle'])

        # add plot title
        if 'title' in self.dict:
            plt.title(self.dict['title'])

        # add text in plot
        if ('text' in self.dict) == ('text_loc' in self.dict) and ('text' in self.dict):
            plt.text(self.dict['text_loc'][0],
                     self.dict['text_loc'][1],
                     self.dict['text'])

        # add axis limits
        if ('xlim' in self.dict) == ('ylim' in self.dict) and ('xlim' in self.dict):
            plt.axis(self.dict['xlim'] + self.dict['ylim'])
        else:
            # default axis limits
            mean_val = np.mean(dataset)

            # x range upto 5 standard deviations
            num_sd = 5
            max_freq = 0

            # width of each interval/bin
            width = np.sqrt(np.var(dataset))

            # default x lim values
            xlim = [mean_val - num_sd*width, mean_val + num_sd*width]

            # find bin count
            for i in Sublist.frange(xlim[0], xlim[1], width):
                count = Sublist(dataset).count_in_range(i, i + width)
                if count > max_freq:
                    max_freq = count

            # find ylim
            ylim = [0, max_freq]

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

        # bins to plot box whisker plot
        bins = self.dict['bins']

        # name of the bin groups
        group_names = self.dict['group_names']

        # variables on x and y axes
        xvar = self.dict['xvar']
        yvar = self.dict['yvar']

        # column indices
        x_index = next(i for i in range(0, len(names)) if names[i] == xvar)
        y_index = next(i for i in range(0, len(names)) if names[i] == yvar)

        # get data to be plotted
        out_data = np.array([[elem[x_index], elem[y_index]] for elem in data])

        # initialize lists
        pdata = list()
        gname = list()

        # data bining
        for i in range(0, len(group_names)):
            temp = out_data[(bins[i] <= out_data[:, 0]) & (out_data[:, 0] < bins[i + 1]), 1]
            for j in range(0, len(temp)):
                pdata.append(temp[j])
                gname.append(group_names[i])

        # dataframe with bins
        df = pd.DataFrame({xvar: gname, yvar: pdata})

        f, (ax) = plt.subplots(1, 1, figsize=(8, 4))
        f.suptitle(self.dict['title'], fontsize=14)

        sns.boxplot(x=xvar, y=yvar,
                    data=df, ax=ax, color=self.dict['color'], showfliers=False)
        ax.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
        ax.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

        return plt

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
        font.set_family('serif')
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
            plot_dict['annot_kws'] = {"size": 15}

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

    def regression_heatmap(self):

        if 'points' in self.dict:
            x = [pt[0] for pt in self.dict['points']]
            y = [pt[1] for pt in self.dict['points']]
        else:
            raise ValueError('No data found')

        if 'xlim' in self.dict:
            xlim = self.dict['xlim']
            xloc = list(k for k in range(0, len(x)) if xlim[0] <= x[k] <= xlim[1])

        else:
            xloc = list(range(0, len(x)))
            xlim = (min(x), max(x))

        if 'ylim' in self.dict:
            ylim = self.dict['ylim']
            yloc = list(k for k in range(0, len(y)) if ylim[0] <= y[k] <= ylim[1])

        else:
            yloc = list(range(0, len(y)))
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
            color = plt.cm.gnuplot2_r

        plt.pcolormesh(xi, yi, zi, cmap=color)

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
                fit = np.polyfit(x, y, 1)
                print('Fitting coefficients: ' + ' '.join([str(i) for i in fit]))
                plt.plot(x_, [fit[0] * x_[i] + fit[1] for i in range(0, len(x_))], 'r-', lw=0.5)
                plt.xlim(xlim)
                plt.ylim(ylim)

        return plt
