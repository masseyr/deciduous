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

"""
this script is to read the reformatted decid and tc files and 
plot them on regression plots
I might also add plotting all the best fits to the same plot here
"""

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

    in_dir = "c:/users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/fires/burn_samp_250/"

    decid_bands = ['decid1992', 'decid2000', 'decid2005', 'decid2010', 'decid2015']
    tc_bands = ['tc1992', 'tc2000', 'tc2005', 'tc2010', 'tc2015']

    decid_uncertainty_bands = ['decid1992u', 'decid2000u', 'decid2005u', 'decid2010u', 'decid2015u']
    tc_uncertainty_bands = ['tc1992u', 'tc2000u', 'tc2005u', 'tc2010u', 'tc2015u']

    fire_cols = ['FIREID', 'SIZE_HA', 'longitude', 'latitude,']
    burn_cols = list('burnyear_{}'.format(str(i+1)) for i in range(20))
    year_edges = [(1950, 1960), (1960, 1970), (1970, 1980),
                  (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2018)]
    year_names = ['year50_60', 'year60_70', 'year70_80',
                  'year80_90', 'year90_00', 'year00_10', 'year10_18']

    fire_types = ['single', 'multiple']

    xvar = 'years'
    cutoff_var = 'decid'

    xlabel = xvar.upper()

    tc_thresh = 25
    weight = False

    density_bins = (100, 100)

    qtls = [0.25, 0.75]

    bin_limit = 10000

    deg = 2
    xlim = (0, 1)
    ylim = (0, 1)

    filelist = list(in_dir + 'year{}_{}_{}_fire.csv'.format(str(year_edge[0])[2:],
                                                            str(year_edge[1])[2:],
                                                            fire_type) for year_edge in year_edges
                    for fire_type in fire_types)

    for filename in filelist:
        if 'single' in filename:
            print(filename)
            val_dicts = Handler(filename).read_from_csv(return_dicts=True)

            print(len(val_dicts))

            if weight:
                outfile = filename.split('.csv')[0] + '_tc_wghtd_tc_thresh_{}.csv'.format(str(tc_thresh))

                plot_dicts = list()
                for val_dict in val_dicts:
                    temp_dict = dict()
                    for i, decid_band in enumerate(decid_bands):
                        temp_dict[decid_band] = float(val_dict[decid_band]) * float(val_dict[tc_bands[i]])




