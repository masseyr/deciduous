from modules import *
import pandas as pd


if __name__ == '__main__':
    # **************************************************

    # plotting box whisker plot

    datafile = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/TEST_RUNS/10k_saved_samp_ak.csv"
    plotfile = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/production/uncert_fig3.png"
    names, data = Handler(datafile).read_csv_as_array()
    bins = Sublist.custom_list(0, 100, step=10)
    group_names = list(str(bins[i]) + '-' + str(bins[i + 1]) for i in range(0, len(bins) - 1))

    plot_boxwhisker = {
        'type': 'boxwhisker',
        'xvar': 'decid_fraction',
        'yvar': 'uncertainty',
        'data': data,
        'bins': bins,
        'names': names,
        'xlabel': 'Deciduous Fraction',
        'ylabel': 'Uncertainty',
        'title': 'Deciduous fraction and uncertainty (n = 50,000)',
        'plotfile': plotfile,
        'color': '#0C92CA',
        'group_names': group_names
    }

    boxwhisker = Plot(plot_boxwhisker)
    boxwhisker.draw()
