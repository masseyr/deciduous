from modules import Handler, Opt, Sublist, RFRegressor, MRegressor, Samples
from sys import argv
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.regression.quantile_regression as qr
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
import logging as logger
import statsmodels.formula.api as smf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interpn
from scipy.ndimage.filters import gaussian_filter
from matplotlib import gridspec
import matplotlib.cm as cm

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000


def myplot(x,
           y,
           sigma=1,
           bins=1000):
    """
    Method to generate heatmap image
    :param x: numpy ndarray
    :param y: numpy ndarray
    :param sigma: smoothing factor
    :param bins:
    :return:
    """

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def quantile_regress(x,
                     y,
                     q=None,
                     degree=1,
                     xlim=None,
                     ylim=None,
                     print_summary=False):
    if q is None:
        q = [0.1, 0.5, 0.9]
    elif type(q).__name__ in ('int', 'float', 'str'):
        try:
            q = [float(q)]
        except Exception as e:
            print(e)
    elif type(q).__name__ in ('list', 'tuple'):
        q = Sublist(q)
    else:
        raise ValueError("Data type not understood: q")

    if type(x).__name__ in ('list', 'tuple', 'NoneType'):
        x_ = np.array(x)
    else:
        x_ = x.copy()

    if type(y).__name__ in ('list', 'tuple', 'NoneType'):
        y_ = np.array(y)
    else:
        y_ = y.copy()

    if xlim is not None:
        y_ = y_[np.where((x_ >= xlim[0]) & (x_ <= xlim[1]))]
        x_ = x_[np.where((x_ >= xlim[0]) & (x_ <= xlim[1]))]

    if ylim is not None:
        x_ = x_[np.where((y_ >= ylim[0]) & (y_ <= ylim[1]))]
        y_ = y_[np.where((y_ >= ylim[0]) & (y_ <= ylim[1]))]

    '''
    y_ = y_[:, np.newaxis]
    const_ = (x_ * 0.0) + 1.0
    x_ = np.hstack([x_[:, np.newaxis], const_[:, np.newaxis]])
    '''

    df = pd.DataFrame({'y': y_, 'x': x_})

    if degree == 0:
        raise RuntimeError('Degree of polynomial should be > 0')

    rel_str = 'y ~ '
    for ii in range(degree):
        if ii == 0:
            rel_str += 'x'
        else:
            rel_str += ' + I(x ** {})'.format(str(float(ii + 1)))

    mod = smf.quantreg(rel_str, df)
    mod.initialize()

    q_res = list()
    for q_ in q:
        res_ = mod.fit(q=q_)

        if print_summary:
            print(res_.summary())
            print('\n')

        q_res.append({
            'q': q_,
            'rsq': res_.rsquared,
            'adj_rsq': res_.rsquared_adj,
            'coeffs': list(reversed(list(res_.params))),
            'pvals': res_.pvalues.tolist(),
            'stderrs': res_.bse.tolist()
        })

    return q_res


def poly_regress(x,
                 y,
                 degree=2,
                 xlim=None,
                 ylim=None):
    if type(x).__name__ in ('list', 'tuple', 'NoneType'):
        x_ = np.array(x)
    else:
        x_ = x.copy()

    if type(y).__name__ in ('list', 'tuple', 'NoneType'):
        y_ = np.array(y)
    else:
        y_ = y.copy()

    if xlim is not None:
        y_ = y_[np.where((x_ >= xlim[0]) & (x_ <= xlim[1]))]
        x_ = x_[np.where((x_ >= xlim[0]) & (x_ <= xlim[1]))]

    if ylim is not None:
        x_ = x_[np.where((y_ >= ylim[0]) & (y_ <= ylim[1]))]
        y_ = y_[np.where((y_ >= ylim[0]) & (y_ <= ylim[1]))]

    results = dict()
    results['degree'] = degree

    coeffs = np.polyfit(x_, y_, degree)

    # Polynomial Coefficients
    results['coeffs'] = coeffs.tolist()

    p_ = np.poly1d(coeffs)

    # fit values, and mean
    yhat = p_(x_)  # or [p(z) for z in x]
    ybar = np.sum(y_) / len(y_)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y_ - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['rsq'] = ssreg / sstot

    return results


if __name__ == '__main__':

    # script, in_dir = argv

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

    xvar_bands = ['decid2010', 'decid2005', 'decid2000']
    yvar_bands = ['pctl_50_2010_090_150', 'pctl_50_2005_090_150', 'pctl_50_2000_090_150']
    zvar_bands = ['tc2010', 'tc2005', 'tc2000']
    oz_bands = ['connected_mask_val18', 'land_extent']

    xvar = 'decid'
    yvar = 'albedo'
    zvar = 'treecover'

    xlabel = 'deciduous_fraction'
    ylabel = 'albedo_pctl_50_090_150'
    zlabel = 'treecover'

    '''
    We are predicting y variable using x and z variables in a random forest model
    '''

    in_dir = "c:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/"

    csv_file = "c:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/" \
               "albedo_data_2000_2010_full_by_tc_loc_boreal_conn_val18.csv"

    plot_file = in_dir + "RF{}_{}_{}_{}_cutoff_{}_deg{}_{}.png".format(ylabel, xlabel, zlabel, str(bin_limit),
                                                                  str(int(z_thresh*scaledown_treecover)), deg,
                                                                  datetime.now().isoformat().split('.')[0]
                                                                  .replace('-', '').replace(':', ''))

    print('Reading file : {}'.format(csv_file))
    print('Plot file: {}'.format(plot_file))

    val_dicts = Handler(csv_file).read_from_csv(return_dicts=True,
                                                read_random=True,
                                                line_limit=None, )

    for val_dict in val_dicts[0:10]:
        print(val_dict)

    basename = Handler(csv_file).basename.split('.csv')[0]

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
    print(plot_arr.shape)

    print('Min, max of x var: {}, {}'.format(str(np.min(plot_arr[:, 0])), str(np.max(plot_arr[:, 0]))))
    print('Min, max of y var: {}, {}'.format(str(np.min(plot_arr[:, 1])), str(np.max(plot_arr[:, 1]))))
    print('Min, max of z var: {}, {}'.format(str(np.min(plot_arr[:, 2])), str(np.max(plot_arr[:, 2]))))

    plot_arr = plot_arr[np.where(plot_arr[:, 2] > z_thresh)]

    print(plot_arr.shape)

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

    print(x_counts)
    print(len(plot_dicts_))

    freq_list, bin_edges = np.histogram(list(float(pt[xvar]) for pt in plot_dicts_), bins=eq_bins, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    X = np.array(list([elem['decid'], elem['treecover']] for elem in plot_dicts_))
    Y = np.array(list(elem['albedo'] for elem in plot_dicts_))

    # plot histogram
    plt.hist(Y,
             alpha=0.8,
             color='blue',
             ec='k',
             bins=eq_bins)

    tmp_file_y = in_dir + "tmp_pre_{}_{}_cutoff_{}_deg{}_{}.png".format(ylabel, str(bin_limit),
                                                                    str(int(z_thresh * scaledown_treecover)), deg,
                                                                    datetime.now().isoformat().split('.')[0]
                                                                    .replace('-', '').replace(':', ''))
    plt.savefig(tmp_file_y, dpi=600)
    plt.close()

    plt.hist(X[:, 0],
             alpha=0.8,
             color='red',
             ec='k',
             bins=eq_bins)

    tmp_file_x = in_dir + "tmp_pre_{}_{}_cutoff_{}_deg{}_{}.png".format(xlabel, str(bin_limit),
                                                                    str(int(z_thresh * scaledown_treecover)), deg,
                                                                    datetime.now().isoformat().split('.')[0]
                                                                    .replace('-', '').replace(':', ''))
    plt.savefig(tmp_file_x, dpi=600)
    plt.close()

    plt.hist(X[:, 1],
             alpha=0.8,
             color='green',
             ec='k',
             bins=eq_bins)

    tmp_file_z = in_dir + "tmp_pre_{}_{}_cutoff_{}_deg{}_{}.png".format(ylabel, str(bin_limit),
                                                                    str(int(z_thresh * scaledown_treecover)), deg,
                                                                    datetime.now().isoformat().split('.')[0]
                                                                    .replace('-', '').replace(':', ''))
    plt.savefig(tmp_file_z, dpi=600)
    plt.close()
    '''
    plot_dicts = Sublist.hist_equalize(plot_dicts_,
                                       num=12000,
                                       pctl=eq_pctl,
                                       nbins=eq_bins,
                                       var=yvar,
                                       minmax=(0, 1))
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



    pts = list()

    print(len(plot_dicts))

    for plot_dict in plot_dicts[0:10]:
        print(plot_dict)

    freq_list, bin_edges = np.histogram(list(float(pt[xvar]) for pt in plot_dicts), bins=eq_bins, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    # ---------------------------------------------------------------------------------------------------------
    # regression BEGIN

    param = {"samp_split": 10, "max_feat": 2, "trees": 100, "samp_leaf": 10}
    model = RFRegressor(**param)

    X = np.array(list([elem['decid'], elem['treecover']] for elem in plot_dicts))
    Y = np.array(list(elem['albedo'] for elem in plot_dicts))

    normal_loc = np.where((Y <= 0.08) | (Y >= 0.12))[0]
    absurd_loc = np.where((Y > 0.08) & (Y < 0.12))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=3000)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    normal_loc = np.where((Y <= 0.12) | (Y >= 0.16))[0]
    absurd_loc = np.where((Y > 0.12) & (Y < 0.16))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=4000)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    normal_loc = np.where((Y <= 0.16) | (Y >= 0.2))[0]
    absurd_loc = np.where((Y > 0.16) & (Y < 0.2))[0]
    ran_loc = np.random.choice(absurd_loc, replace=False, size=5000)
    Y = np.concatenate([Y[ran_loc], Y[normal_loc]])
    X = np.concatenate([X[ran_loc, :], X[normal_loc, :]])

    print('**********************************')
    print(len(Y))
    freq_list, bin_edges = np.histogram(Y, bins=25, range=(0, 1))
    print(freq_list)
    print(bin_edges)
    print(np.min(Y), np.max(Y))
    print('**********************************')

    # plot histogram
    plt.hist(Y,
             alpha=0.8,
             color='blue',
             ec='k',
             bins=bin_edges)

    tmp_file_y = in_dir + "tmp_{}_{}_cutoff_{}_deg{}_{}.png".format(ylabel, str(bin_limit),
                                                                    str(int(z_thresh * scaledown_treecover)), deg,
                                                                    datetime.now().isoformat().split('.')[0]
                                                                    .replace('-', '').replace(':', ''))
    plt.savefig(tmp_file_y, dpi=600)
    plt.close()

    plt.hist(X[:, 0],
             alpha=0.8,
             color='red',
             ec='k',
             bins=bin_edges)

    tmp_file_x = in_dir + "tmp_{}_{}_cutoff_{}_deg{}_{}.png".format(xlabel, str(bin_limit),
                                                                    str(int(z_thresh * scaledown_treecover)), deg,
                                                                    datetime.now().isoformat().split('.')[0]
                                                                    .replace('-', '').replace(':', ''))
    plt.savefig(tmp_file_x, dpi=600)
    plt.close()

    plt.hist(X[:, 1],
             alpha=0.8,
             color='green',
             ec='k',
             bins=bin_edges)

    tmp_file_z = in_dir + "tmp_{}_{}_cutoff_{}_deg{}_{}.png".format(zlabel, str(bin_limit),
                                                                    str(int(z_thresh * scaledown_treecover)), deg,
                                                                    datetime.now().isoformat().split('.')[0]
                                                                    .replace('-', '').replace(':', ''))
    plt.savefig(tmp_file_z, dpi=600)
    plt.close()

    exit()

    # Y = 0.9*(np.log10(Y*1000+1) - 1.8)
    # Y = np.abs((np.log10((Y**2)*1000+1) - 0.21) / 2.8)
    # Y = np.abs((np.log2(np.log10((Y ** 2) * 1000 + 1) + 1) - 0.8) / 1.2)
    '''
    Y = np.abs(0.9 * (np.log10(Y * 1000 + 1) - 1.8))
    Y = np.abs(np.log10(Y * 1000 + 1) - 0.8) / 3.5
    _loc = np.where(Y>0.3)
    Y = Y[_loc]
    X = X[_loc]
    '''

    samp = Samples(x=X, y=Y, x_name=['treecover','decid'], y_name='albedo')

    print('------------------------')
    print(len(X[:,0]))
    freq_list, bin_edges = np.histogram(X[:,0], bins=25, range=(0, 1))
    print(freq_list)
    print(bin_edges)
    print('====')
    print(len(Y))
    freq_list, bin_edges = np.histogram(Y, bins=25, range=(0, 1))
    print(freq_list)
    print(bin_edges)
    print(np.min(Y), np.max(Y))
    print('------------------------')

    # trn_samp, val_samp = samp.random_partition(50)

    print(samp)
    # print(trn_samp)
    # print(val_samp)

    model.fit_data(samp.format_data())

    # model.get_adjustment_param(clip=0.05,
    #                           over_adjust=1.0)

    pred = model.sample_predictions(samp.format_data())

    print(pred)

    print(len(pred['obs_y']))
    print(len(pred['pred_y']))

    print('\n')

    freq_list, bin_edges = np.histogram(pred['obs_y'], bins=eq_bins, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    freq_list, bin_edges = np.histogram(pred['pred_y'], bins=eq_bins, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    var_imp = model.var_importance()

    print(var_imp)

    x = pred['obs_y']
    y = pred['pred_y']
    intercept = pred['intercept']
    slope = pred['slope']
    rsq = pred['rsq']

    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
    plt.rcParams['axes.labelweight'] = 'bold'

    data, x_e, y_e = np.histogram2d(x, y, bins=density_bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=fill_value)

    # Sort the points by density, so that the densest points are plotted last
    idx = (np.array(z).argsort()).tolist()

    x = list(x[i] for i in idx)
    y = list(y[i] for i in idx)
    z = list(z[i] for i in idx)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, c=z, alpha=0.2, s=6)
    plt.plot(x, [slope*(x[i]) + intercept for i in range(0, len(x))], 'r-', lw=2)
    out_str = 'Y = {:3.2f} X + {:3.2f}'.format(slope, intercept)

    out_str += '\n$R^2$ = {:3.2f}'.format(rsq) + '\nN = {:,}'.format(len(x))

    rect = patches.Rectangle((0.42 * xlim[1], 0.98 * ylim[1]), 0.94 * xlim[1], -0.2 * ylim[1],
                             linewidth=1, fill=True, alpha=0.8,
                             edgecolor='black', facecolor='whitesmoke')

    # Add the patch to the Axes
    plt.gca().add_patch(rect)

    plt.text(0.45 * xlim[1], 0.95 * ylim[1], out_str,
             horizontalalignment='left',
             verticalalignment='top', )

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.savefig(plot_file, dpi=1200)

    plt.close()
    Opt.cprint('Plot file : {}'.format(plot_file))


