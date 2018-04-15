from modules import *

if __name__ == '__main__':
    # **************************************************
    # plotting histogram plot - rmse, rsq

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

    datafile = "D:\\projects\\NAU\\landsat_deciduous\\data\\10k_saved_samp_ak.csv"
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
