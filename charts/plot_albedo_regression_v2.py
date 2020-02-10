from modules.plots import Plot
from modules import Handler, Opt, Sublist, _Regressor
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

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000


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

    deg = 2
    xlim = (0, 1)
    ylim = (0, 1)

    nodata = -9999.0

    density_bins = (500, 500)

    qtls = [0.1, 0.9]

    fill_value = 0

    z_thresh = 0.25

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
    yvar_bands = ['pctl_90_2010_240_300', 'pctl_90_2005_240_300', 'pctl_90_2000_240_300']
    zvar_bands = ['tc2010', 'tc2005', 'tc2000']
    oz_bands = ['connected_mask_val18', 'land_extent']

    xvar = 'decid'
    yvar = 'albedo'

    xlabel = 'deciduous_fraction'
    ylabel = 'albedo_pctl_90_240_300'

    in_dir = "c:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/"

    csv_file = "c:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/" \
               "albedo_data_2000_2010_full_by_tc_loc_boreal_conn_val18.csv"
    plot_file = in_dir + "{}_{}_{}_cutoff_{}_deg{}_{}.png".format(ylabel, xlabel, str(bin_limit),
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
                                          (np.log10(float(y)) -1)/2.0,
                                          float(z)/scaledown_treecover])

    plot_arr = np.array(elem_list)
    print(np.min(plot_arr[:, 1]), np.max(plot_arr[:, 1]))
    print(plot_arr.shape)

    plot_arr = plot_arr[np.where(plot_arr[:, 2] > z_thresh)]

    print(plot_arr.shape)

    x_range = range(0, 101)
    x_dict = dict()

    for x_val in x_range:
        x_dict[x_val] = list()

        list_vals = plot_arr[np.where(plot_arr[:, 0] == float(x_val)/100.0)].tolist()

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
        plot_dicts_ += list({xvar: elem[0], yvar: elem[1]} for elem in v)

    print(x_counts)
    print(len(plot_dicts_))

    freq_list, bin_edges = np.histogram(list(float(pt[xvar]) for pt in plot_dicts_), bins=25, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    plot_dicts = Sublist.hist_equalize(plot_dicts_,
                                       pctl=25,
                                       nbins=50,
                                       var=xvar,
                                       minmax=(0, 1.01))

    pts = list()

    print(len(plot_dicts))

    freq_list, bin_edges = np.histogram(list(float(pt[xvar]) for pt in plot_dicts), bins=25, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    x__ = np.array(list(plot_dict[xvar] for plot_dict in plot_dicts))
    y__ = np.array(list(plot_dict[yvar] for plot_dict in plot_dicts))

    loc = np.where((x__ >= xlim[0]) & (x__ <= xlim[1]) & (y__ >= ylim[0]) & (y__ <= ylim[1]))

    x_ = x__[loc]
    y_ = y__[loc]

    freq_list, bin_edges = np.histogram(x_, bins=25, range=(0, 1))
    print(freq_list)
    print(bin_edges)

    print(len(plot_dicts))
    print('x_ :{}'.format(str(x_.shape[0])))
    print('y_ :{}'.format(str(y_.shape[0])))

    poly = poly_regress(x_, y_, xlim=xlim, ylim=ylim, degree=deg)
    print(poly)

    quant_reg = quantile_regress(x_, y_, q=qtls, xlim=xlim, ylim=ylim,
                                 degree=deg, print_summary=True)

    for res in quant_reg:
        print(res)

    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
    plt.rcParams['axes.labelweight'] = 'bold'

    data, x_e, y_e = np.histogram2d(x_, y_, bins=density_bins)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x_, y_]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=fill_value)

    # Sort the points by density, so that the densest points are plotted last
    idx = (np.array(z).argsort()).tolist()

    x = list(x_[i] for i in idx)
    y = list(y_[i] for i in idx)
    z = list(z[i] for i in idx)

    plt.figure(figsize=(6, 5))

    zorder = 0

    plt.scatter(x, y, c=z, alpha=0.2, s=6, zorder=zorder)

    zorder += 1

    p = np.poly1d(poly['coeffs'])

    print('Fitting coefficients: ' + ' '.join([str(i) for i in poly['coeffs']]))
    plt.plot(x_, [p(x_[i]) for i in range(0, len(x_))], 'r-', lw=2)

    for reg in quant_reg:
        p = np.poly1d(reg['coeffs'])
        print('Fitting coefficients: ' + ' '.join([str(i) for i in reg['coeffs']]))
        plt.plot(x_, [p(x_[i]) for i in range(0, len(x_))], 'r-', lw=1, linestyle='dashed')

    rhs = list()
    for i, coeff in enumerate(poly['coeffs']):
        if i < (len(poly['coeffs']) - 2):
            rhs.append('{:3.3f} $x^{}$'.format(coeff, (len(poly['coeffs']) - i - 1)))
        elif i == (len(poly['coeffs']) - 2):
            rhs.append('{:3.3f} x'.format(coeff, (len(poly['coeffs']) - i - 1)))
        else:
            rhs.append('{:3.2f}'.format(coeff))

    out_str = 'Y = {}'.format(' + '.join(rhs))

    out_str += '\n$R^2$ = {:3.2f}'.format(poly['rsq']) + '\nN = {:,}'.format(len(x_))

    rect = patches.Rectangle((0.42 * xlim[1], 0.98 * ylim[1]), 0.94 * xlim[1], -0.2 * ylim[1],
                             linewidth=1, fill=True, alpha=0.8,
                             edgecolor='black', facecolor='whitesmoke')

    # Add the patch to the Axes
    plt.gca().add_patch(rect)

    plt.text(0.45 * xlim[1], 0.95 * ylim[1], out_str,
             horizontalalignment='left',
             verticalalignment='top', )
    '''
    rect = patches.Rectangle((0.02 * xlim[1], 0.98 * ylim[1]), 0.56 * xlim[1], -0.2 * ylim[1],
                             linewidth=1, fill=True, alpha=0.8,
                             edgecolor='black', facecolor='whitesmoke')

    # Add the patch to the Axes
    plt.gca().add_patch(rect)

    plt.text(0.05 * xlim[1], 0.95 * ylim[1], out_str,
             horizontalalignment='left',
             verticalalignment='top', )

    '''

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.savefig(plot_file, dpi=1200)
    plt.close()
    Opt.cprint('Plot file : {}'.format(plot_file))

    '''

    hist_plot_file = in_dir + "{}_{}_{}_hist_cutoff_{}_deg{}_{}.png".format(bandname, xvar, str(bin_limit),
                                                                     str(cutoff), deg,
                                                               datetime.now().isoformat().split('.')[0]
                                                               .replace('-', '').replace(':', ''))

    Opt.cprint('Plot file : {}'.format(hist_plot_file))

    plot_histogram = {
        'type': 'histogram',
        'data': plot_dicts,
        'xtitle': xvar,
        'xvar': xvar,
        'names': list(plot_dicts[0]),
        'plotfile': hist_plot_file,
        'xlim': (0, 1),
        'binwidth': 0.04,

    }
    histogram = Plot(plot_histogram)
    histogram.draw()

    hist_plot_file = in_dir + "{}_{}_{}_hist_cutoff_{}_deg{}_{}.png".format(bandname, yvar, str(bin_limit),
                                                                     str(cutoff), deg,
                                                               datetime.now().isoformat().split('.')[0]
                                                               .replace('-', '').replace(':', ''))

    Opt.cprint('Plot file : {}'.format(hist_plot_file))

    plot_histogram = {
        'type': 'histogram',
        'data': plot_dicts,
        'xtitle': yvar,
        'xvar': yvar,
        'names': list(plot_dicts[0]),
        'plotfile': hist_plot_file,
        'xlim': (0, 1),
        'binwidth': 0.04,

    }
    histogram = Plot(plot_histogram)
    histogram.draw()

    '''
