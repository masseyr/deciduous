from modules.plots import Plot
from modules import Handler, Samples
from modules.classification import RFRegressor, MRegressor


if __name__ == '__main__':

    """
    # **************************************************
    # plotting histogram plot - rmse, rsq

    datafile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/results_summary_v13i_show.csv"
    names, data = Handler(datafile).read_csv_as_array()

    # rmse plot
    plotfile_rmse = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/rmse_plot_all_v13i.png"
    plot_rmse = {
        'type': 'histogram',  # plot types: histogram, surface, relative, regression
        'names': names,
        'data': data,  # list or 1d array
        'nbins': 50,
        'xvar': 'rmse',
        'xtitle': 'Random Forest model RMSE',  # title of x axis
        'ytitle': 'Frequency',  # title of y axis
        'title': 'Model RMSE plot',  # plot title
        'plotfile': plotfile_rmse,  # output file name
        'xlim': [18, 23],
        'ylim': [0, 2500]
    }
    rmse = Plot(plot_rmse)
    rmse.draw()

    # rsq plot
    plotfile_rsq = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/rsq_plot_all_v13i.png"
    plot_rsq = {
        'type': 'histogram',  # plot types: histogram, surface, relative, regression
        'names': names,
        'data': data,  # list or 1d array
        'xvar': 'rsq',
        'nbins': 50,
        'xtitle': 'Random Forest model Pearson R-squared',  # title of x axis
        'ytitle': 'Frequency',  # title of y axis
        'title': 'Model R-squared plot',  # plot title
        'plotfile': plotfile_rsq,  # output file name
        'color': 'purple',
        'xlim': [60, 75],
        'ylim': [0, 2500]
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

    infile = "C:/Users/rm885/" \
             "Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/" + \
             "gee_data_cleaning_v27.csv"
    plotfile = "C:/Users/rm885/" \
               "Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/" + \
               "gee_data_cleaning_v27_heatmap2.png"

    trn_samp = Samples(csv_file=infile, label_colname='decid_frac')
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
    """
    # plotting regression heatmap2
    val_log = "C:/temp/tree_cover/out_tc_2010_samp_v1_RF_2.txt"
    '''
    pickle_file = "C:/temp/tree_cover/out_tc_2010_samp_v1_RF_2.pickle"
    classifier = RFRegressor.load_from_pickle(pickle_file)

    data_dict = classifier.sample_predictions(classifier.data)
    x = data_dict['obs_y']
    y = data_dict['pred_y']

    '''
    val_lines = Handler(val_log).read_text_by_line()
    x = list(float(elem.strip()) for elem in val_lines[0].split(',')[1:])
    y = list(float(elem.strip()) for elem in val_lines[1].split(',')[1:])

    pts = [(x[i], y[i]) for i in range(0, len(x))]

    print(pts[0:10])

    plotfile = 'C:/temp/tree_cover/out_tc_2010_samp_v1_RF_2_val.png'

    plot_heatmap = {
        'type': 'rheatmap2',
        'points': pts,
        'xlabel': 'Observed Tree Cover',
        'ylabel': 'Predicted Tree Cover',
        'color_bar_label': 'Data-points per bin',
        'plotfile': plotfile,
        'xlim': (0, 100),
        'ylim': (0, 100),
        'line': True,
        'legend': True
    }

    heatmap = Plot(plot_heatmap)
    heatmap.draw()
