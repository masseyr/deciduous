from common import Handler, Sublist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
}
"""


class Plot:
    """Class to plot data"""
    def __init__(self, dict):
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
        if 'type' in self.dict:
            if self.dict['type'] == 'histogram':
                self.histogram()
            elif self.dict['type'] == 'boxwhisker':
                self.boxwhisker()
            elif self.dict['type'] == 'regression':
                self.regression()
        else:
            raise ValueError("Plot type not found")

    def histogram(self):

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

        # save to file or show in window
        if self.filename is not None:
            plt.savefig(self.filename)
        else:
            plt.show()

    def boxwhisker(self):

        # get file name from dictionary
        filename = self.dict['filename']

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

        f, (ax1) = plt.subplots(1, 1, figsize=(8, 4))
        f.suptitle(self.dict['title'], fontsize=14)

        sns.boxplot(x=xvar, y=yvar,
                    data=df, ax=ax1, color=self.dict['color'], showfliers=False)
        ax1.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
        ax1.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

        plt.savefig(self.filename)

    def regression(self):
        # Under construction
        pass
