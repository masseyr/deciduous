from modules import *
import pandas as pd

if __name__ == '__main__':
    # **************************************************
    file1 = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/TEST_RUNS/ak_decid_means_v2.csv"
    data1 = pd.read_csv(file1)
    data_dicts = data1.T.to_dict().values()

    fire_list = list()
    for data_dict in data_dicts:
        if 1950 <= data_dict['FireYear'] <= 1980:
            print(data_dict)
            temp_dict = dict()
            temp_dict['year'] = data_dict['FireYear']



    exit()

    # plotting box whisker plot

    datafile = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/TEST_RUNS/10k_saved_samp_ak.csv"
    plotfile = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/production/uncert_fig2.png"
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
        'datafile': datafile,
        'plotfile': plotfile,
        'color': '#0C92CA',
        'group_names': group_names
    }

    boxwhisker = Plot(plot_boxwhisker)
    boxwhisker.draw()

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