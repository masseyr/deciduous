import numpy as np
from modules import *

if __name__ == '__main__':
    # **************************************************
    # plotting histogram plot - rmse, rsq
    """
    datafile = "D:\\projects\\NAU\\landsat_deciduous\\data\\rf_info_val.csv"
    names, data = Handler(datafile).read_csv_as_array()

    # rmse plot
    plotfile_rmse = "D:\\Shared\\Dropbox\\projects\\NAU\\\landsat_deciduous\\data\\rmse_plot.png"
    plot_rmse = {
        'type': 'histogram',  # plot types: histogram, surface, relative, regression
        'names': names,
        'data': data,  # list or 1d array
        'xvar': 'rmse',
        'xtitle': 'Random Forest model RMSE',  # title of x axis
        'ytitle': 'Frequency',  # title of y axis
        'title': 'Model RMSE plot (n = 2000)',  # plot title
        'plotfile': plotfile_rmse,  # output file name
    }
    rmse = Plot(plot_rmse)
    rmse.draw()

    # rsq plot
    plotfile_rsq = "D:\\projects\\NAU\\\landsat_deciduous\\data\\rsq_plot.png"
    plot_rsq = {
        'type': 'histogram',  # plot types: histogram, surface, relative, regression
        'names': names,
        'data': data,  # list or 1d array
        'xvar': 'rsq',
        'xtitle': 'Random Forest model Pearson R-squared',  # title of x axis
        'ytitle': 'Frequency',  # title of y axis
        'title': 'Model R-squared plot (n = 2000)',  # plot title
        'plotfile': plotfile_rsq,  # output file name
        'color': 'purple'
    }
    rsq = Plot(plot_rsq)
    rsq.draw()

    # **************************************************
    # plotting box whisker plot

    datafile = "D:\\projects\\NAU\\landsat_deciduous\\data\\saved_samp_ak.csv"
    plotfile = "D:\\projects\\NAU\\landsat_deciduous\\data\\fig1.png"
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
        'xtitle': 'Deciduous Fraction',
        'ytitle': 'Uncertainty',
        'title': 'Deciduous fraction and uncertainty (n = 50,000)',
        'datafile': datafile,
        'plotfile': plotfile,
        'color': 'orange',
        'group_names': group_names
    }

    boxwhisker = Plot(plot_boxwhisker)
    boxwhisker.draw()

    # **************************************************
    # plotting matrix heatmap

    infile = "D:\\Shared\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\ABoVE_all_2010_sampV1_clean.csv"
    plotfile = "D:\\Shared\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\heatmap2.png"

    trn_samp = Samples(csv_file=infile, label_colname='Decid_AVG')
    corr_dict = trn_samp.correlation_matrix()

    corr_mat = corr_dict['data']
    xlabel = corr_dict['names']

    plot_heatmap = {
        'type': 'mheatmap',
        'data': corr_mat,
        'xlabel': xlabel,
        'title': 'Correlation among variables',
        'plotfile': plotfile,
        'show_values': False,
        'heat_range': [0.0, 1.0],
        'color_str': "YlGnBu"
    }

    heatmap = Plot(plot_heatmap)
    heatmap.draw()
    """
    # **************************************************
    # plotting regression heatmap

    yhb_arr = Handler("C:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/RF_bootstrap_2000_y_hat_bar_v1.csv")\
        .read_array_from_csv(array_1d=True, nodataval=-99.0)
    vary_arr = Handler("C:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/RF_bootstrap_2000_var_y_v1.csv") \
        .read_array_from_csv(array_1d=True, nodataval=-99.0)
    yf_arr = Handler("C:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/RF_bootstrap_2000_y_v1.csv") \
        .read_array_from_csv(array_1d=True, nodataval=-99.0)

    y = (np.abs(yf_arr-yhb_arr))**2
    x = vary_arr
    print(np.mean(y))
    print(np.mean(x))

    pts = [(np.sqrt(x[i]), np.sqrt(y[i])) for i in range(0, len(x))]
    # pts = [(x[i], y[i]) for i in range(0, len(x))]

    plotfile = 'C:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/test_plot12_.png'

    plot_heatmap = {
        'type': 'rheatmap',
        'points': pts,
        'xlabel': 'SD in tree predictors',
        'ylabel': 'Abs. difference in observed and predicted',
        'color_bar_label': 'Data-points per bin',
        'plotfile': None,
        'xlim': (0, 0.5),
        'ylim': (0, 1),
        'line': False,
        'xbins': 100,
        'ybins': 100,
    }

    heatmap = Plot(plot_heatmap)
    heatmap.draw()

    # **************************************************
