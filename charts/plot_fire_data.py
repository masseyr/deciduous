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
plt.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})
plt.rcParams['axes.labelweight'] = 'bold'
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000


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

    in_dir = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/fires/burn_samp_250_by_5yr/"

    decid_bands = ['decid1992', 'decid2000', 'decid2005', 'decid2010', 'decid2015']
    tc_bands = ['tc1992', 'tc2000', 'tc2005', 'tc2010', 'tc2015']

    decid_uncertainty_bands = ['decid1992u', 'decid2000u', 'decid2005u', 'decid2010u', 'decid2015u']
    tc_uncertainty_bands = ['tc1992u', 'tc2000u', 'tc2005u', 'tc2010u', 'tc2015u']

    fire_cols = ['FIREID', 'SIZE_HA', 'longitude', 'latitude,']
    burn_cols = list('burnyear_{}'.format(str(i+1)) for i in range(20))
    # year_edges = [(1950, 1960), (1960, 1970), (1970, 1980),
    #               (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2018)]

    year_edges = [(1950, 1955), (1955, 1960), (1960, 1965), (1965, 1970), (1970, 1975), (1975, 1980),
                  (1980, 1985), (1985, 1990), (1990, 1995), (1995, 2000), (2000, 2005),  (2005, 2010),
                  (2010, 2015)]

    year_names = list('year{}_{}'.format(str(year_edge[0])[2:], str(year_edge[1])[2:]) for year_edge in year_edges)

    fire_types = ['single', 'multiple']

    xvar = 'years'
    cutoff_var = 'decid'

    xlabel = xvar.upper()

    tc_thresh = 25
    weight = True

    density_bins = (100, 100)

    qtls = [0.25, 0.75]

    bin_limit = 10000

    deg = 2
    xlim = (1985, 2020)
    ylim = (0, 1)

    x__ = [1992, 2000, 2005, 2010, 2015]
    y__ = [0.1, 0.3, 0.5, 0.7, 0.9]

    version = 7

    f_ = lambda x_: 0.5*(x_ ** 2) + 0.5

    filelist = list(in_dir + 'year_{}_{}_{}_fire.csv'.format(str(year_edge[0])[2:],
                                                            str(year_edge[1])[2:],
                                                            fire_type) for year_edge in year_edges
                    for fire_type in fire_types)

    for filename in filelist:
        '''
        if 'single' in filename:
            print(filename)
            val_dicts = Handler(filename).read_from_csv(return_dicts=True)

            print(len(val_dicts))

            # weight:
            plotfile = filename.split('.csv')[0] + '_tc_wghtd_tc_thresh_{}_v{}.png'.format(str(tc_thresh),
                                                                                           str(version))

            x = list()
            y = list()
            for val_dict in val_dicts:
                temp_dict = dict()
                for i, decid_band in enumerate(decid_bands):

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_bands[i]]).__name__ in ('int', 'float')):

                        if val_dict[tc_bands[i]] >= tc_thresh:
                            y.append((float(val_dict[decid_band]) * float(val_dict[tc_bands[i]]))/10000.0)
                            x.append(int(decid_band.split('decid')[1]))

            f, (ax) = plt.subplots(1, 1, figsize=(8, 4))
            f.suptitle(Handler(plotfile).basename, fontsize=14)

            sns.boxplot(x, y, ax=ax, color='orange', showfliers=False)
            # ax.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
            # ax.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

            plt.savefig(plotfile)
            plt.close()

            print('Plotted: {}'.format(plotfile))

            # thresh only
            plotfile = filename.split('.csv')[0] + '_tc_thresh_{}_v{}.png'.format(str(tc_thresh),
                                                                                  str(version))

            x = list()
            y = list()
            for val_dict in val_dicts:
                temp_dict = dict()
                for i, decid_band in enumerate(decid_bands):

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_bands[i]]).__name__ in ('int', 'float')):

                        if val_dict[tc_bands[i]] >= tc_thresh:
                            y.append(float(val_dict[decid_band]) / 100.0)
                            x.append(int(decid_band.split('decid')[1]))

            f, (ax) = plt.subplots(1, 1, figsize=(8, 4))
            f.suptitle(Handler(plotfile).basename, fontsize=14)

            sns.boxplot(x, y, ax=ax, color='orange', showfliers=False)
            # ax.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
            # ax.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

            # plt.ylim(ylim)

            plt.savefig(plotfile)
            plt.close()

            print('Plotted: {}'.format(plotfile))
        

        if 'single' in filename:
            print(filename)
            val_dicts = Handler(filename).read_from_csv(return_dicts=True)

            print(len(val_dicts))

            # weight:
            plotfile = filename.split('.csv')[0] + '_tc_wghtd_tc_thresh_{}_v{}.png'.format(str(tc_thresh),
                                                                                           str(version))

            x = list()
            y = list()
            z = list()
            medians = list()
            for val_dict in val_dicts:
                temp_dict = dict()
                for i, decid_band in enumerate(decid_bands):

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_bands[i]]).__name__ in ('int', 'float')):

                        if float(val_dict['longitude']) > -142.0:

                            if val_dict[tc_bands[i]] >= 0:
                                y.append((float(val_dict[decid_band]) * float(val_dict[tc_bands[i]])) / 10000.0)
                                x.append(int(decid_band.split('decid')[1]))
                                z.append('Deciduous_fraction')

                            y.append(float(val_dict[tc_bands[i]]) / 100.0)
                            x.append(int(tc_bands[i].split('tc')[1]))
                            z.append('Tree_cover')

            f, (ax) = plt.subplots(1, 1, figsize=(8, 6))
            # f.suptitle(Handler(plotfile).basename, fontsize=14)

            sns.set(font_scale=2)
            sns.boxplot(x, y, ax=ax, hue=z, showfliers=False)
            # ax.set_xlabel('Years', size=12, alpha=0.8)
            # ax.set_ylabel('Deciduous Fraction', size=12, alpha=0.8)
            ax.legend_.remove()

            plt.savefig(plotfile)
            plt.close()

            print('Plotted: {}'.format(plotfile))

            # thresh only
            plotfile = filename.split('.csv')[0] + '_tc_thresh_{}_v{}.png'.format(str(tc_thresh),
                                                                                  str(version))

            x = list()
            y = list()
            z = list()
            for val_dict in val_dicts:
                temp_dict = dict()
                for i, decid_band in enumerate(decid_bands):

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_bands[i]]).__name__ in ('int', 'float')):

                        if float(val_dict['longitude']) > -142.0:

                            if val_dict[tc_bands[i]] >= tc_thresh:
                                y.append(float(val_dict[decid_band]) / 100.0)
                                x.append(int(decid_band.split('decid')[1]))
                                z.append('Deciduous_fraction')

                            y.append(float(val_dict[tc_bands[i]]) / 100.0)
                            x.append(int(tc_bands[i].split('tc')[1]))
                            z.append('Tree_cover')

            f, (ax) = plt.subplots(1, 1, figsize=(8, 6))
            # f.suptitle(Handler(plotfile).basename, fontsize=14)

            sns.set(font_scale=2)
            sns.boxplot(x, y, ax=ax, hue=z, showfliers=False)
            # ax.set_xlabel('Years', size=12, alpha=0.8)
            # ax.set_ylabel('Deciduous Fraction', size=12, alpha=0.8)
            ax.legend_.remove()

            plt.savefig(plotfile)
            plt.close()

            print('Plotted: {}'.format(plotfile))
        
        if 'single' in filename:

            tc_thresh = 0

            print(filename)
            val_dicts = Handler(filename).read_from_csv(return_dicts=True)

            print(len(val_dicts))

            # weight:
            plotfile = filename.split('.csv')[0] + '_tc_wghtd_tc_thresh_{}_v{}.png'.format(str(tc_thresh),
                                                                                           str(version))

            x = list()
            y = list()
            z = list()
            for val_dict in val_dicts:
                temp_dict = dict()
                for i, decid_band in enumerate(decid_bands):

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_bands[i]]).__name__ in ('int', 'float')):

                        if val_dict[tc_bands[i]] >= tc_thresh:
                            y.append((float(val_dict[decid_band]) * float(val_dict[tc_bands[i]])) / 10000.0)
                            x.append(int(decid_band.split('decid')[1]))
                            z.append('Deciduous_fraction')

                            y.append(float(val_dict[tc_bands[i]]) / 100.0)
                            x.append(int(tc_bands[i].split('tc')[1]))
                            z.append('Tree_cover')

            f, (ax) = plt.subplots(1, 1, figsize=(8, 4))
            f.suptitle(Handler(plotfile).basename, fontsize=14)

            sns.violinplot(x, y, ax=ax, hue=z, showfliers=False)
            # ax.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
            # ax.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

            plt.savefig(plotfile)
            plt.close()

            print('Plotted: {}'.format(plotfile))

            # thresh only
            plotfile = filename.split('.csv')[0] + '_tc_thresh_{}_v{}.png'.format(str(tc_thresh),
                                                                                  str(version))

            tc_thresh = 0

            x = list()
            y = list()
            z = list()
            for val_dict in val_dicts:
                temp_dict = dict()
                for i, decid_band in enumerate(decid_bands):

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_bands[i]]).__name__ in ('int', 'float')):

                        if val_dict[tc_bands[i]] >= tc_thresh:
                            y.append(float(val_dict[decid_band]) / 100.0)
                            x.append(int(decid_band.split('decid')[1]))
                            z.append('Deciduous_fraction')

                            y.append(float(val_dict[tc_bands[i]]) / 100.0)
                            x.append(int(tc_bands[i].split('tc')[1]))
                            z.append('Tree_cover')

            f, (ax) = plt.subplots(1, 1, figsize=(8, 4))
            f.suptitle(Handler(plotfile).basename, fontsize=14)

            sns.violinplot(x, y, ax=ax, hue=z, showfliers=False)
            # ax.set_xlabel(self.dict['xlabel'], size=12, alpha=0.8)
            # ax.set_ylabel(self.dict['ylabel'], size=12, alpha=0.8)

            # plt.ylim(ylim)

            plt.savefig(plotfile)
            plt.close()

            print('Plotted: {}'.format(plotfile))
        '''