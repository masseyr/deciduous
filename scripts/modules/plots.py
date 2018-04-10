from common import Handler, Sublist
import numpy as np
import pandas as pd
from numpy.random import randn
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["patch.force_edgecolor"] = True
plt.interactive(False)

# this is under construction


plot_dictionary = {
    'type': None,  # plot types: histogram, surface, relative, regression
    'data': None,  # list or 1d array
    'bins': None,  # list or 1d array
    'range': None,  # tuple of two elements
    'xlim': None,  # list of two elements
    'ylim': None,  # list of two elements
    'xlabel': None,  # x axis labels
    'ylabel': None,  # y axis labels
    'xtitle': None,  # title of x axis
    'ytitle': None,  # title of y axis
    'title': None,  # plot title
    'text': None,  # text to display in plot
    'text_loc': None,  # location of text on plot
    'filename': None,  # output file name
    'color': None,  # plot color
}


class Plot:
    """Class to plot data"""
    def __init__(self, dict):
        self.dict = dict
        if 'filename' in self.dict:
            self.filename = Handler(self.dict['filename']).file_remove_check()
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

    def histogram(self):

        # data
        if 'data' not in self.dict:
            raise ValueError("Data not found")
        else:
            dataset = self.dict['data']

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


if __name__ == '__main__':

    dataset1 = randn(1000)*0.0015 + 0.0025
    file1 = 'c:/temp/plot3.png'

    plot_d = {
        'type': 'histogram',
        'data': dataset1,
        'title': 'Random plot',
        'filename': file1,
        'xtitle': 'Independent variable (units)',
        'ytitle': 'Dependent variable (units)',

    }

    kk = Plot(plot_d)
    kk.draw()






