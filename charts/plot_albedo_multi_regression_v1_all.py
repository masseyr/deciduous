from modules import Handler, Opt, Sublist, RFRegressor, MRegressor, Samples
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import spline_filter
from matplotlib.font_manager import FontProperties as fp
import matplotlib.patches as patches
from scipy.interpolate import interpn
# import sphviewer as sph

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000


def heatplot(x,
             y,
             sigma=4,
             val_range=None,
             density=False,
             bins=100):

    if val_range is None:
        val_range = [[max(x), min(x)], [max(y), min(y)]]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, density=density, range=val_range)
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    extent = val_range[0] + val_range[1]

    return heatmap.T, extent

'''
def sph_plot(x, y, nb=16, xsize=1000, ysize=1000):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    x0 = (xmin + xmax) / 2.
    y0 = (ymin + ymax) / 2.

    pos = np.zeros([len(x), 3])
    pos[:, 0] = x
    pos[:, 1] = y
    w = np.ones(len(x))

    P = sph.Particles(pos, w, nb=nb)
    S = sph.Scene(P)
    S.update_camera(r='infinity', x=x0, y=y0, z=0,
                    xsize=xsize, ysize=ysize)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()
    extent = R.get_extent()
    for i, j in zip(xrange(4), [x0, x0, y0, y0]):
        extent[i] += j

    return img, extent
'''

def sc_plot(x, y, bins):
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=fill_value)

    # Sort the points by density, so that the densest points are plotted last
    idx = (np.array(z).argsort()).tolist()

    x = list(x[i] for i in idx)
    y = list(y[i] for i in idx)
    z = list(z[i] for i in idx)
    return x, y, z


if __name__ == '__main__':

    '''
    We are predicting y variable using x and z variables in a random forest model
    '''

    in_dir = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/"

    # indir = "/scratch/rm885/gdrive/sync/decid/excel/"

    csv_file = in_dir + "albedo_data_2000_2010_full_by_tc_loc_boreal_conn_val18.csv"

    plt.rcParams.update({'font.size': 16, 'font.family': 'Calibri'})
    plt.rcParams['axes.labelweight'] = 'regular'

    bin_limit = 50000

    deg = 1
    xlim = (0, 1)
    ylim = (0, 1)

    nodata = -9999.0

    density_bins = (1000, 1000)

    qtls = [0.1, 0.9]

    fill_value = 0

    z_thresh = 0.05

    eq_bins = 25

    eq_pctl = 40

    scaledown_albedo = 1000.0
    scaledown_treecover = 100.0
    scaledown_decid = 1.0

    band_list = ['pctl_50_2000_001_090','pctl_50_2005_150_240', 'tc2005','decid2010u', 'pctl_90_2000_240_300',
                 'tc2000', 'pctl_90_2000_180_240', 'land_extent', 'pctl_90_2010_120_180','connected_mask_val18',
                 'pctl_90_2005_180_240', 'pctl_90_2010_240_300', 'pctl_90_2010_180_240', 'pctl_90_2005_240_300',
                 'pctl_50_2005_090_150', 'tc2010', 'decid2005u', 'pctl_50_2000_090_150', 'pctl_50_2010_150_240',
                 'pctl_50_2005_001_090', 'decid2010', 'pctl_50_2010_090_150', 'decid2005', 'decid2000',
                 'pctl_50_2010_001_090', 'pctl_90_2005_120_180', 'decid2000u', 'pctl_50_2000_150_240',
                 'pctl_90_2000_120_180']

    xvar = 'decid'
    yvar = 'albedo'
    zvar = 'treecover'

    xlabel = 'deciduous_fraction'
    ylabel = 'albedo'
    zlabel = 'treecover'

    xvar_bands = ['decid2010', 'decid2005', 'decid2000']
    zvar_bands = ['tc2010', 'tc2005', 'tc2000']
    oz_bands = ['connected_mask_val18', 'land_extent']

    print('Reading file : {}'.format(csv_file))

    val_dicts = Handler(csv_file).read_from_csv(return_dicts=True,
                                                read_random=True,
                                                line_limit=None, )

    basename = Handler(csv_file).basename.split('.csv')[0]

    plot_file = in_dir + "RF{}_{}_{}_{}_cutoff_{}_deg{}_{}.png".format(ylabel, xlabel, zlabel, str(bin_limit),
                                                                       str(int(z_thresh * scaledown_treecover)), deg,
                                                                       datetime.now().isoformat().split('.')[0]
                                                                       .replace('-', '').replace(':', ''))
    print('Plot file: {}'.format(plot_file))

    # -------------------- spring -----------------------------------------------------------------------

    yvar_bands = ['pctl_50_2010_090_150', 'pctl_50_2005_090_150', 'pctl_50_2000_090_150']

    elem_list = list()
    for val_dict in val_dicts:
        for i in range(3):
            x, y, z = val_dict[xvar_bands[i]], val_dict[yvar_bands[i]], val_dict[zvar_bands[i]]

            oz1, oz2 = val_dict[oz_bands[0]], val_dict[oz_bands[1]]

            if (type(x) in (int, float)) and (type(y) in (int, float)) and (type(z) in (int, float)):
                if (x != nodata) and (y != nodata) and (y != nodata):
                    if (oz1 == 1) and (oz2 == 1):
                        elem_list.append([float(x)/scaledown_decid,
                                          float(y)/scaledown_albedo,
                                          float(z)/scaledown_treecover])

    plot_arr = np.array(elem_list)

    plot_arr = plot_arr[np.where(plot_arr[:, 2] > z_thresh)]

    x_range = range(0, 101)
    x_dict = dict()

    for x_val in x_range:
        x_dict[x_val] = list()

        list_vals = plot_arr[np.where((plot_arr[:, 0]*100.0).astype(np.int16) == x_val)].tolist()

        if len(list_vals) > bin_limit:
            temp_list = list(range(len(list_vals)))
            list_locs = Sublist(temp_list).random_selection(bin_limit)
        else:
            list_locs = list(range(len(list_vals)))

        x_dict[x_val] += list(list_vals[ii] for ii in list_locs)

    x_counts = dict()
    plot_dicts_ = list()
    for k, v in x_dict.items():
        x_counts[k] = len(v)
        plot_dicts_ += list({xvar: elem[0], yvar: elem[1], zvar: elem[2]} for elem in v)

    plot_dicts = Sublist.hist_equalize(plot_dicts_,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=xvar,
                                       minmax=(0, 1))

    plot_dicts = Sublist.hist_equalize(plot_dicts,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=zvar,
                                       minmax=(0, 1))

    X = np.array(list([elem['decid'], elem['treecover']] for elem in plot_dicts))
    Y = np.array(list(elem['albedo'] for elem in plot_dicts))

    absurd_loc = np.where((Y >= 0.12) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=500)
    Y_ = Y[ran_loc].copy()
    X_ = X[ran_loc, :].copy()

    absurd_loc = np.where((Y >= 0.14) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=500)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.13) & (Y <= 0.18))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=1000)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.15) & (Y <= 0.17))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=1000)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.16) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=500)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.18) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=1500)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    normal_loc = np.where((Y < 0) | (Y > 0.2))[0]
    absurd_loc = np.where((Y >= 0) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=2000)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    normal_loc = np.where((Y < 0.2) | (Y > 0.24))[0]
    absurd_loc = np.where((Y >= 0.2) & (Y <= 0.24))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=4500)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    Y = np.concatenate([Y, Y_])
    X = np.concatenate([X, X_])

    samp = Samples(x=X, y=Y)    # , x_name=['decid','treecover'], y_name='albedo')

    print(samp.head)

    param = {"samp_split": 10, "max_feat": 2, "trees": 100, "samp_leaf": 10}
    model = RFRegressor(**param)

    model.fit_data(samp.format_data())
    model.adjustment = {'bias': -0.4, 'lower_limit': 0.0, 'gain': 2, 'upper_limit': 1}

    pred = model.sample_predictions(samp.format_data())

    model = RFRegressor(**param)
    model.fit_data(samp.format_data())
    model.get_adjustment_param(clip=0.0, over_adjust=1.0)

    pred_ = model.sample_predictions(samp.format_data())

    pred_sd = model.sample_predictions(samp.format_data())

    var_imp1 = model.var_importance()

    x1 = pred['obs_y']
    y1 = pred['pred_y']

    intercept1 = pred_['intercept']
    slope1 = pred_['slope']
    rsq1 = pred_['rsq']

    picklefile = plot_file.replace('.png', '_spring.pickle')
    model.pickle_it(picklefile)

    print(model.features)
    print(pred_)

    # ------------ summer -------------------------------------------------------------------------------------------


    yvar_bands = ['pctl_50_2010_150_240', 'pctl_50_2005_150_240', 'pctl_50_2000_150_240']

    elem_list = list()
    for val_dict in val_dicts:
        for i in range(3):
            x, y, z = val_dict[xvar_bands[i]], val_dict[yvar_bands[i]], val_dict[zvar_bands[i]]

            oz1, oz2 = val_dict[oz_bands[0]], val_dict[oz_bands[1]]

            if (type(x) in (int, float)) and (type(y) in (int, float)) and (type(z) in (int, float)):
                if (x != nodata) and (y != nodata) and (y != nodata):
                    if (oz1 == 1) and (oz2 == 1):
                        elem_list.append([float(x)/scaledown_decid,
                                          float(y)/scaledown_albedo,
                                          float(z)/scaledown_treecover])

    plot_arr = np.array(elem_list)

    plot_arr = plot_arr[np.where(plot_arr[:, 2] > z_thresh)]

    x_range = range(0, 101)
    x_dict = dict()

    for x_val in x_range:
        x_dict[x_val] = list()

        list_vals = plot_arr[np.where((plot_arr[:, 0]*100.0).astype(np.int16) == x_val)].tolist()

        if len(list_vals) > bin_limit:
            temp_list = list(range(len(list_vals)))
            list_locs = Sublist(temp_list).random_selection(bin_limit)
        else:
            list_locs = list(range(len(list_vals)))

        x_dict[x_val] += list(list_vals[ii] for ii in list_locs)

    x_counts = dict()
    plot_dicts_ = list()
    for k, v in x_dict.items():
        x_counts[k] = len(v)
        plot_dicts_ += list({xvar: elem[0], yvar: elem[1], zvar: elem[2]} for elem in v)
    '''
    plot_dicts = Sublist.hist_equalize(plot_dicts_,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=xvar,
                                       minmax=(0, 1))

    plot_dicts = Sublist.hist_equalize(plot_dicts,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=zvar,
                                       minmax=(0, 1))
    '''
    plot_dicts = plot_dicts_
    X = np.array(list([elem['decid'], elem['treecover']] for elem in plot_dicts))
    Y = np.array(list(elem['albedo'] for elem in plot_dicts))

    freq_list, bin_edges = np.histogram(Y, bins=25, range=(0, 1))

    samp = Samples(x=X, y=Y, x_name=['decid','treecover'], y_name='albedo')

    trn_samp, val_samp = samp.random_partition(30)

    param = {"samp_split": 10, "max_feat": 2, "trees": 100, "samp_leaf": 10}
    model = RFRegressor(**param)

    model.fit_data(samp.format_data())

    pred = model.sample_predictions(samp.format_data())

    model = RFRegressor(**param)
    model.fit_data(samp.format_data())
    model.get_adjustment_param(clip=0.0, over_adjust=1.01)

    pred_ = model.sample_predictions(trn_samp.format_data(), sd_y=True)

    var_imp2 = model.var_importance()

    x2 = pred_['obs_y']
    y2 = pred_['pred_y']

    intercept2 = pred_['intercept']
    slope2 = pred_['slope']
    rsq2 = pred_['rsq']

    picklefile = plot_file.replace('.png', '_summer.pickle')
    model.pickle_it(picklefile)

    print(pred_)

    # ------------------fall ----------------------------------------------------------------------------------------

    yvar_bands = ['pctl_90_2010_240_300', 'pctl_90_2005_240_300', 'pctl_90_2000_240_300']

    elem_list = list()
    for val_dict in val_dicts:
        for i in range(3):
            x, y, z = val_dict[xvar_bands[i]], val_dict[yvar_bands[i]], val_dict[zvar_bands[i]]

            oz1, oz2 = val_dict[oz_bands[0]], val_dict[oz_bands[1]]

            if (type(x) in (int, float)) and (type(y) in (int, float)) and (type(z) in (int, float)):
                if (x != nodata) and (y != nodata) and (y != nodata):
                    if (oz1 == 1) and (oz2 == 1):
                        elem_list.append([float(x) / scaledown_decid,
                                          float(y) / scaledown_albedo,
                                          float(z) / scaledown_treecover])

    plot_arr = np.array(elem_list)

    plot_arr = plot_arr[np.where(plot_arr[:, 2] > z_thresh)]

    x_range = range(0, 101)
    x_dict = dict()

    for x_val in x_range:
        x_dict[x_val] = list()

        list_vals = plot_arr[np.where((plot_arr[:, 0] * 100.0).astype(np.int16) == x_val)].tolist()

        if len(list_vals) > bin_limit:
            temp_list = list(range(len(list_vals)))
            list_locs = Sublist(temp_list).random_selection(bin_limit)
        else:
            list_locs = list(range(len(list_vals)))

        x_dict[x_val] += list(list_vals[ii] for ii in list_locs)

    x_counts = dict()
    plot_dicts_ = list()
    for k, v in x_dict.items():
        x_counts[k] = len(v)
        plot_dicts_ += list({xvar: elem[0], yvar: elem[1], zvar: elem[2]} for elem in v)

    plot_dicts = Sublist.hist_equalize(plot_dicts_,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=xvar,
                                       minmax=(0, 1))

    plot_dicts = Sublist.hist_equalize(plot_dicts,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=zvar,
                                       minmax=(0, 1))

    X = np.array(list([elem['decid'], elem['treecover']] for elem in plot_dicts))
    Y = np.array(list(elem['albedo'] for elem in plot_dicts))

    absurd_loc = np.where((Y >= 0.12) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=500)
    Y_ = Y[ran_loc].copy()
    X_ = X[ran_loc, :].copy()

    absurd_loc = np.where((Y >= 0.14) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=500)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.13) & (Y <= 0.18))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=1000)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.15) & (Y <= 0.17))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=1000)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.16) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=500)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    absurd_loc = np.where((Y >= 0.18) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=1500)
    Y_ = np.concatenate([Y[ran_loc], Y_])
    X_ = np.concatenate([X[ran_loc, :], X_])

    normal_loc = np.where((Y < 0) | (Y > 0.2))[0]
    absurd_loc = np.where((Y >= 0) & (Y <= 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=2000)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    normal_loc = np.where((Y < 0.2) | (Y > 0.24))[0]
    absurd_loc = np.where((Y >= 0.2) & (Y <= 0.24))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=9000)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    Y = np.concatenate([Y, Y_])
    X = np.concatenate([X, X_])

    freq_list, bin_edges = np.histogram(Y, bins=25, range=(0, 1))

    samp = Samples(x=X, y=Y, x_name=['decid', 'treecover'], y_name='albedo')

    trn_samp, val_samp = samp.random_partition(60)

    param = {"samp_split": 10, "max_feat": 2, "trees": 100, "samp_leaf": 10}
    model = RFRegressor(**param)

    model.fit_data(samp.format_data())
    model.adjustment = {'bias': -0.35, 'lower_limit': 0.0, 'gain': 2, 'upper_limit': 1}

    pred = model.sample_predictions(samp.format_data())

    model = RFRegressor(**param)
    model.fit_data(samp.format_data())
    model.get_adjustment_param(clip=0.0, over_adjust=1.01)

    pred_ = model.sample_predictions(trn_samp.format_data(), sd_y=True)

    var_imp3 = model.var_importance()

    x3 = pred['obs_y']
    y3 = pred['pred_y']

    intercept3 = pred_['intercept']
    slope3 = pred_['slope']
    rsq3 = pred_['rsq']

    picklefile = plot_file.replace('.png', '_fall.pickle')
    model.pickle_it(picklefile)

    print(pred_)

    # ---------------------------------------------------------------------------------------------------------------

    fig = plt.figure(1, figsize=(10, 10))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    '''
    heatmap_8, extent_8 = myplot(x, y, nb=6)
    heatmap_16, extent_16 = myplot(x, y, nb=8)
    heatmap_32, extent_32 = myplot(x, y, nb=12)
    '''

    heatmap_1, extent_1 = heatplot(x1, y1, val_range=[[0,1], [0,1]], sigma=2, density=True)
    heatmap_2, extent_2 = heatplot(x2, y2, val_range=[[0,1], [0,1]], sigma=2, density=True)
    heatmap_3, extent_3 = heatplot(x3, y3, val_range=[[0,1], [0,1]], sigma=2, density=True)

    n1 = len(x1)
    n2 = len(x2)
    n3 = len(x3)

    out_str1 = '$R^2$ = {:3.2f}'.format(rsq1) + '\nN = {:,}'.format(n1)
    out_str2 = '$R^2$ = {:3.2f}'.format(rsq2) + '\nN = {:,}'.format(n2)
    out_str3 = '$R^2$ = {:3.2f}'.format(rsq3) + '\nN = {:,}'.format(n3)

    # x1, y1, z1 = sc_plot(x1, y1, density_bins)
    # x2, y2, z2 = sc_plot(x2, y2, density_bins)
    # x3, y3, z3 = sc_plot(x3, y3, density_bins)

    end_p = np.array([0, 1])

    font = fp()
    font.set_size('small')

    rect = patches.Rectangle((0.65 * xlim[1], 0.98 * ylim[1]), 0.33 * xlim[1], -0.18 * ylim[1],
                             linewidth=1, fill=True, alpha=0.9,
                             edgecolor='black', facecolor='whitesmoke', zorder=2)

    ax1.imshow(heatmap_1, extent=extent_1, origin='lower', aspect='auto', cmap=cm.hot_r, zorder=0)
    # ax1.scatter(x1, y1, c=z1, alpha=0.2, s=9, zorder=0, cmap=cm.gnuplot2_r)
    ax1.plot(end_p, end_p, 'k', ls='dashed', lw=1, zorder=1)
    ax1.text(0.68 * xlim[1], 0.95 * ylim[1], out_str1, horizontalalignment='left',verticalalignment='top', zorder=3)
    ax1.add_patch(rect)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    rect = patches.Rectangle((0.65 * xlim[1], 0.98 * ylim[1]), 0.33 * xlim[1], -0.18 * ylim[1],
                             linewidth=1, fill=True, alpha=0.9,
                             edgecolor='black', facecolor='whitesmoke', zorder=2)

    ax2.imshow(heatmap_2, extent=extent_2, origin='lower', aspect='auto', cmap=cm.hot_r, zorder=0)
    # ax2.scatter(x2, y2, c=z2, alpha=0.2, s=9, zorder=0, cmap=cm.gnuplot2_r)
    ax2.plot(end_p, end_p, 'k', ls='dashed', lw=1, zorder=1)
    ax2.text(0.68 * xlim[1], 0.95 * ylim[1], out_str2, horizontalalignment='left', verticalalignment='top', zorder=3)
    ax2.add_patch(rect)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')

    rect = patches.Rectangle((0.65 * xlim[1], 0.98 * ylim[1]), 0.33 * xlim[1], -0.18 * ylim[1],
                             linewidth=1, fill=True, alpha=0.9,
                             edgecolor='black', facecolor='whitesmoke', zorder=2)

    ax3.imshow(heatmap_3, extent=extent_3, origin='lower', aspect='auto', cmap=cm.hot_r, zorder=0)
    # ax3.scatter(x3, y3, c=z3, alpha=0.2, s=9, zorder=0, cmap=cm.gnuplot2_r)
    ax3.plot(end_p, end_p, 'k', ls='dashed', lw=1, zorder=1)
    ax3.text(0.68 * xlim[1], 0.95 * ylim[1], out_str3, horizontalalignment='left', verticalalignment='top', zorder=3)
    ax3.add_patch(rect)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')

    df = (var_imp3[1][1], var_imp2[1][1], var_imp1[1][1])
    tc = (var_imp3[0][1], var_imp2[0][1], var_imp1[0][1])

    print(var_imp1)
    print(var_imp2)
    print(var_imp3)

    print(tc)
    print(df)

    ind = np.arange(3)  # the x locations for the groups
    width = 0.35  # the width of the bars

    ax4.barh(ind + width, tc, width, edgecolor='k', color='#FFC300')
    ax4.barh(ind, df, width, edgecolor='k')

    ax4.set_yticks(ind + width / 2)
    ax4.set_yticklabels(('Fall', 'Summer', 'Spring'),  va='center', rotation='vertical')

    # font = fp()
    # font.set_size('small')
    ax4.legend(('DF', 'TC'), edgecolor='k')  # prop=font,
    ax4.set_xlim(0, 1)

    ax4.spines['right'].set_color('none')
    ax4.spines['top'].set_color('none')

    plt.savefig(plot_file, dpi=600)
