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

    in_dir = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/"

    csv_file = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/albedo_data_2000_2010_full_by_tc.csv"

    Opt.cprint('Reading file : {}'.format(csv_file))

    val_dicts = Handler(csv_file).read_from_csv(return_dicts=True,
                                                read_random=True,
                                                line_limit=None, )

    bandname = Handler(csv_file).basename.split('.csv')[0]

    colors = ['darkred', 'darkgreen', 'orange', 'royalblue']

    albedo_list = ['albedo_1', 'albedo_2', 'albedo_3', 'albedo_4']
    cutoff_list = [(0, .25), (.25, .50), (.50, .75), (.75, 1.00)]

    xvar = 'treecover'
    cutoff_var = 'decid'

    xlabel = xvar.upper()

    bin_limit = 10000

    deg = 2
    xlim = (0, 1)
    ylim = (0, 1)

    density_bins = (100, 100)

    qtls = [0.25, 0.75]

    fill_value = 0

    for albedo_var in albedo_list:
        yvar = albedo_var
        ylabel = albedo_var.upper()

        plot_file = in_dir + "{}_{}_{}_{}_deg{}_{}.png".format(bandname, yvar, xvar, str(bin_limit), deg,
                                                                         datetime.now().isoformat().split('.')[0]
                                                                         .replace('-', '').replace(':', ''))

        x_range = range(0, 101)
        x_dict = dict()

        for x_val in x_range:
            x_dict[x_val] = list()

            list_vals = list()
            for dict_ in val_dicts:
                if dict_[xvar] == x_val and dict_[yvar] > 0 \
                        and cutoff_list[0][0]*100.0 <= dict_[cutoff_var] <= cutoff_list[-1][1]*100.0:

                    temp = {xvar: float(dict_[xvar]) / 100.0,
                            yvar: float(dict_[yvar]) / 1000.0,
                            cutoff_var: float(dict_[cutoff_var] / 100.0)}
                    list_vals.append(temp)

            if len(list_vals) > bin_limit:
                list_vals = Sublist(list_vals).random_selection(bin_limit)

            x_dict[x_val] += list_vals

        x_counts = dict()
        plot_dicts_ = list()
        for k, v in x_dict.items():
            x_counts[k] = len(v)
            plot_dicts_ += v

        plot_dicts = Sublist.hist_equalize(plot_dicts_,
                                           pctl=5,
                                           nbins=25,
                                           var=xvar,
                                           minmax=(0, 0.95))

        pts = list()

        plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.figure(figsize=(6, 5))

        for jj, cutoff in enumerate(cutoff_list):

            print('Cutoff: {}'.format(cutoff))

            cutoff_plot_dicts = list(plot_dict for plot_dict in plot_dicts
                                     if cutoff[0] <= plot_dict[cutoff_var] <= cutoff[1])

            print(len(cutoff_plot_dicts))

            x__ = np.array(list(plot_dict[xvar] for plot_dict in cutoff_plot_dicts))
            y__ = np.array(list(plot_dict[yvar] for plot_dict in cutoff_plot_dicts))

            loc = np.where((x__ >= xlim[0]) & (x__ <= xlim[1]) & (y__ >= ylim[0]) & (y__ <= ylim[1]))

            x_ = x__[loc]
            y_ = y__[loc]

            poly = poly_regress(x_, y_, xlim=xlim, ylim=ylim, degree=deg)
            print(poly)

            p = np.poly1d(poly['coeffs'])

            plox_x_ = [xlim[0]] + x_.tolist() + [xlim[1]]

            print('Fitting coefficients: ' + ' '.join([str(i) for i in poly['coeffs']]))
            plt.plot(plox_x_, [p(plox_x_[i]) for i in range(0, len(plox_x_))], 'r-', c=colors[jj], lw=2)

            quant_reg = quantile_regress(x_, y_, q=qtls, xlim=xlim, ylim=ylim,
                                         degree=deg, print_summary=False)

            l_quant = quant_reg[0]
            u_quant = quant_reg[1]

            plt.fill_between(plox_x_, np.poly1d(u_quant['coeffs'])(plox_x_), np.poly1d(l_quant['coeffs'])(plox_x_),
                             color=colors[jj], interpolate=True, alpha=0.2)



            '''
            for res in quant_reg:
                print(res)
            for reg in quant_reg:
                p = np.poly1d(reg['coeffs'])
                print('Fitting coefficients: ' + ' '.join([str(i) for i in reg['coeffs']]))
                plt.plot(x_, [p(x_[i]) for i in range(0, len(x_))], 'r-', lw=1, linestyle='dashed')
            '''

            rhs = list()
            for i, coeff in enumerate(poly['coeffs']):
                if i < (len(poly['coeffs']) - 2):
                    rhs.append('{:3.3f} $x^{}$'.format(coeff, (len(poly['coeffs']) - i - 1)))
                elif i == (len(poly['coeffs']) - 2):
                    rhs.append('{:3.3f} x'.format(coeff, (len(poly['coeffs']) - i - 1)))
                else:
                    rhs.append('{:3.2f}'.format(coeff))

            print(' '.join(rhs))

            out_str = '$R^2$ = {:3.2f}'.format(poly['rsq']) + '\nN = {:,}'.format(len(x_))

            print(out_str)

        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.savefig(plot_file, dpi=1200)
        plt.close()
        Opt.cprint('Plot file : {}'.format(plot_file))

