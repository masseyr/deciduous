import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from modules import *
from scipy.interpolate import interpn
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.formula.api as smf
from matplotlib import gridspec


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
            rel_str += ' + I(x ** {})'.format(str(float(ii+1)))

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

    main_dir = "d:/shared/"

    forc_col = 'forc_sep'
    ver = 'wtc_v2'
    '''
    file1 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1950_1978_extract_wtc.csv"
    file2 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1978_1998_extract_wtc.csv"
    file3 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1998_2018_extract_wtc.csv"

    '''
    file1 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1950_1978_extract_wtc_v2_.csv"
    file2 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1978_1998_extract_wtc_v2_.csv"
    file3 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1998_2018_extract_wtc_v2_.csv"
    '''
    file1 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1950_1978_extract_f_full.csv"
    file2 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1978_1998_extract_f_full.csv"
    file3 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1998_2018_extract_f_full4.csv"
    '''

    data1 = Handler(file1).read_from_csv(return_dicts=True)
    data2 = Handler(file2).read_from_csv(return_dicts=True)
    data3 = Handler(file3).read_from_csv(return_dicts=True)

    '''
    ofile1 = "C:/Users/Richard/Downloads/fire_1958_1978_extract_f.png"

    forc1 = list(float(elem['forcing'])/1000.0 for elem in data1 if type(elem['forcing']) is not str)

    # matplotlib histogram
    res = plt.hist(forc1, color='#0C92CA', edgecolor='black',
                   bins=int(30),range=(-400, 400))

    # Add labels
    plt.savefig(ofile1,dpi=300)
    plt.close()

    ofile2 = "C:/Users/Richard/Downloads/fire_1978_1998_extract_f.png"

    forc2 = list(float(elem['forc_tc25'])/1000.0 for elem in data2 if float(elem['forc_tc25']) is not str)

    # matplotlib histogram
    res = plt.hist(forc2, color='#EE9B48', edgecolor='black',
                   bins=int(30),range=(-400, 400))

    # Add labels
    plt.savefig(ofile2,dpi=300)
    plt.close()

    ofile3 = "C:/Users/Richard/Downloads/fire_1998_2018_extract_f.png"

    forc3 = list(float(elem['forc_tc25'])/1000.0 for elem in data3 if float(elem['forc_tc25']) is not str)

    # matplotlib histogram
    res = plt.hist(forc3, color='#AD2A0D', edgecolor='black',
                   bins=int(30),range=(-400, 400))

    # Add labels
    plt.savefig(ofile3, dpi=300)
    plt.close()
    '''

    ofile1 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1950_1978_extract_f_norm_area_{}_{}.png".format(ver, forc_col)

    forc1 = list(float(elem[forc_col])/float(elem['area']) for elem in data1 if (type(elem[forc_col]) is not str) and \
                 (type(elem['area']) is not str))

    # matplotlib histogram
    res = plt.hist(forc1, color='#0C92CA', edgecolor='black',
                   bins=int(30),range=(-0.01, 0.01))

    plt.ylim((0, 1800))
    plt.savefig(ofile1,dpi=300)
    plt.close()

    ofile2 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1978_1998_extract_f_norm_area_{}_{}.png".format(ver, forc_col)

    forc2 = list(float(elem[forc_col])/float(elem['area']) for elem in data2 if (type(elem[forc_col]) is not str) and \
                 (type(elem['area']) is not str))

    # matplotlib histogram
    res = plt.hist(forc2, color='#EE9B48', edgecolor='black',
                   bins=int(30),range=(-0.01, 0.01))

    plt.ylim((0, 1800))
    plt.savefig(ofile2,dpi=300)
    plt.close()

    ofile3 = main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_1998_2018_extract_f_norm_area_{}_{}.png".format(ver, forc_col)

    forc3 = list(float(elem[forc_col])/float(elem['area']) for elem in data3 if (type(elem[forc_col]) is not str) and \
                 (type(elem['area']) is not str))

    # matplotlib histogram
    res = plt.hist(forc3, color='#AD2A0D', edgecolor='black',
                   bins=int(30),range=(-0.01, 0.01))

    plt.ylim((0, 1800))
    plt.savefig(ofile3, dpi=300)
    plt.close()


    '''
    all_data = data1 + data2 + data3
    all_forc = list((float(elem['forc_tc25']) / 1000.0, elem['YEAR'])
                    for elem in all_data if type(elem['forc_tc25']) is not str)

    x_ = list(elem[1] for elem in all_forc)
    y_ = list(elem[0] for elem in all_forc)

    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
    plt.rcParams['axes.labelweight'] = 'bold'

    data, x_e, y_e = np.histogram2d(x_, y_, bins=(100,100))
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x_, y_]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0)

    # Sort the points by density, so that the densest points are plotted last
    idx = (np.array(z).argsort()).tolist()

    x = list(x_[i] for i in idx)
    y = list(y_[i] for i in idx)
    z = list(z[i] for i in idx)

    plt.figure(figsize=(12, 6))

    plt.scatter(x, y, c='#F14F17', marker='o', alpha=0.5, s=10)
    plt.hlines(0, 1955,2020, lw=1)
    plt.xlim((1955,2020))
    plt.ylim((-4000,4000))

    plt.savefig("C:/Users/Richard/Downloads/fire_extract_f.png", dpi=1200)
    plt.close()

    '''

    all_data = data1 + data2  + data3
    all_forc = list((float(elem[forc_col]), int(elem['YEAR']), float(elem['area']), float(elem['decid_ch_00_15']))
                    for elem in all_data if (type(elem[forc_col]) is not str) and \
                 (type(elem['area']) is not str))

    x_ = list(elem[1] for elem in all_forc)
    y_ = list(elem[0] for elem in all_forc)

    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
    plt.rcParams['axes.labelweight'] = 'bold'

    data, x_e, y_e = np.histogram2d(x_, y_, bins=(1000,1000))
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x_, y_]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0)

    # Sort the points by density, so that the densest points are plotted last
    idx = (np.array(z).argsort()).tolist()

    x = list(x_[i] for i in idx)
    y = list(y_[i] for i in idx)
    z = list(z[i] for i in idx)

    plt.figure(figsize=(12, 6))

    plt.scatter(x, y, c='#F14F17', marker='o',  alpha=1, s=10)  # '#F14F17'
    plt.hlines(0, 1948,2020, lw=1)
    plt.xlim((1948,2020))
    plt.ylim((-0.025, 0.025))

    plt.savefig(main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_extract_f_norm_area2__{}_{}.png".format(ver, forc_col), dpi=1200)
    plt.close()

    all_list = list()
    year_list = list(range(1950, 2020))

    neg_area_list = list(0 for _ in range(1950, 2020))
    pos_area_list = list(0 for _ in range(1950, 2020))

    pos_decid_list = list(0 for _ in range(1950, 2020))
    neg_decid_list = list(0 for _ in range(1950, 2020))

    negative_list = list(0 for _ in range(1950, 2020))
    positive_list = list(0 for _ in range(1950, 2020))

    counts = list(0 for _ in range(1950, 2020))
    for j, i in enumerate(range(1950, 2020)):
        temp_list = list()
        for elem in all_forc:
            if elem[1] == i:
                temp_list.append(elem[0]/elem[2])

                counts[j] += 1

                if elem[0] >= 0:
                    positive_list[j] += elem[0]
                    pos_area_list[j] += elem[2]
                    pos_decid_list[j] += elem[3]
                else:
                    negative_list[j] += elem[0]
                    neg_area_list[j] += elem[2]
                    neg_decid_list[j] += elem[3]

        positive_list[j] = (float(positive_list[j]) / float(pos_area_list[j])) if pos_area_list[j] > 0 else 0
        negative_list[j] = (float(negative_list[j]) / float(neg_area_list[j])) if neg_area_list[j] > 0 else 0

        pos_decid_list[j] = (float(pos_decid_list[j]) / float(pos_area_list[j])) if pos_area_list[j] > 0 else 0
        neg_decid_list[j] = (float(neg_decid_list[j]) / float(neg_area_list[j])) if neg_area_list[j] > 0 else 0

        print((i, counts[j]))

        all_list.append(np.array(temp_list))

    print(counts)
    counts = [None] + counts[:-1]
    print(len(counts))
    print(counts)
    print(min(counts))
    print(max(counts))

    print(positive_list)
    print(negative_list)

    print(pos_decid_list)
    print(neg_decid_list)

    # labels = list(i if (i%5==0) else '' for i in range(1950, 2020))
    labels = list((5*i+1950) for i in range(0, len(range(1950, 2020))))
    # labels = list(i for i in range(1950, 2020) if (i%5==0))

    '''
    
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hlines(0, 0,len(all_list), lw=0.5,colors='k')

    box = ax.boxplot(all_list, patch_artist=True, vert=True, whis=1.75,labels=labels,showfliers=False,
                                flierprops=dict(marker='.', markerfacecolor='#F14F17',fillstyle='full',
                                markeredgecolor='#F14F17', markeredgewidth=0.0, markersize=4, linestyle='none'))

    colors = ['#205CD5' for _ in box['boxes']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlim((0,len(all_list)))

    ax.xaxis.set_major_locator(ticker.IndexLocator(5,1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.tick_params(which='major', length=5, width=2)
    ax.tick_params(which='minor', length=3)

    ax.set_ylim((-0.013, 0.013))

    ax.plot(range(len(counts)), counts, lw=0.5, color='#EA8F19')
    fig.savefig(main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_extract_f_norm_area4_.png", dpi=1200)
    plt.close()

    fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(4, 1)  # rows, cols

    labels2 = list('' for i in range(0, len(range(1950, 2020))))


    ax1 = fig.add_subplot(gs[1:4, 0])
    ax1.hlines(0, 0,len(all_list), lw=0.5,colors='k')

    box = ax1.boxplot(all_list, patch_artist=True, vert=True, whis=1.75,labels=labels,showfliers=False,
                                flierprops=dict(marker='.', markerfacecolor='#F14F17',fillstyle='full',
                                markeredgecolor='#F14F17', markeredgewidth=0.0, markersize=4, linestyle='none'))

    colors = ['#205CD5' for _ in box['boxes']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlim(0,len(all_list))

    ax1.xaxis.set_major_locator(ticker.IndexLocator(5,1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax1.tick_params(which='major', length=5, width=2)
    ax1.tick_params(which='minor', length=3)

    ax1.set_ylim(-0.013, 0.011)
    ax1.grid(True, linestyle='dotted')

    ax2 = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2.plot(range(len(counts)), counts, lw=1, color='#EA8F19')
    ax2.tick_params(labelbottom=False, )

    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax2.tick_params(which='minor', length=2)
    ax2.set_xlim(0, len(all_list))
    ax2.set_ylim(-1, 600)
    ax2.grid(True, linestyle='dotted')

    fig.savefig(main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_extract_f_norm_area6_2.png", dpi=1200)
    plt.close()

    fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(4, 1)  # rows, cols

    labels2 = list('' for i in range(0, len(range(1950, 2020))))


    ax1 = fig.add_subplot(gs[1:4, 0])
    ax1.hlines(0, 0,len(all_list), lw=0.5,colors='k')

    box = ax1.boxplot(all_list, patch_artist=True, vert=True, whis=1.75,labels=labels,showfliers=False,
                                flierprops=dict(marker='.', markerfacecolor='#F14F17',fillstyle='full',
                                markeredgecolor='#F14F17', markeredgewidth=0.0, markersize=4, linestyle='none'))

    colors = ['#205CD5' for _ in box['boxes']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlim(0,len(all_list))

    ax1.xaxis.set_major_locator(ticker.IndexLocator(5,1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax1.tick_params(which='major', length=5, width=2)
    ax1.tick_params(which='minor', length=3)

    ax1.set_ylim(-0.013, 0.011)
    ax1.grid(True, linestyle='dotted')

    ax2 = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2.plot(range(len(counts)), counts, lw=1, color='#EA8F19')
    ax2.tick_params(labelbottom=False, )

    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax2.tick_params(which='minor', length=2)
    ax2.set_xlim(0, len(all_list))
    ax2.set_ylim(-1, 600)
    ax2.grid(True, linestyle='dotted')

    fig.savefig(main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_extract_f_norm_area6_2.png", dpi=1200)
    plt.close()

    fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(4, 1)  # rows, cols

    labels2 = list('' for i in range(0, len(range(1950, 2020))))


    ax1 = fig.add_subplot(gs[2:4, 0])
    ax1.hlines(0, 0,len(all_list), lw=0.75,colors='k')

    box = ax1.boxplot(all_list, patch_artist=True, vert=True, whis=1.75,labels=labels,showfliers=False,
                                flierprops=dict(marker='.', markerfacecolor='#F14F17',fillstyle='full',
                                markeredgecolor='#F14F17', markeredgewidth=0.0, markersize=4, linestyle='none'))

    colors = ['#205CD5' for _ in box['boxes']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlim(0,len(all_list))

    ax1.xaxis.set_major_locator(ticker.IndexLocator(5,1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(which='major', axis='x', length=5, width=2)
    ax1.tick_params(which='minor', axis='x', length=3)

    ax1.set_ylim(-0.013, 0.013)
    ax1.grid(True, linestyle='dotted', lw=0.75, zorder=0)

    ax2 = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax2.grid(True, linestyle='dotted', lw=0.75, zorder=0)
    ax2.plot(range(len(counts)), counts, lw=1, color='#EA8F19')
    ax2.tick_params(labelbottom=False, )

    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax2.tick_params(which='minor', length=2)
    ax2.set_xlim(0, len(all_list))
    ax2.set_ylim(-1, 600)


    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3.grid(True, linestyle='dotted', lw=0.75, zorder=0)
    ax3.hlines(0, 0, len(year_list), lw=0.75, colors='k')
    ax3.bar(range(len(year_list)), positive_list, width=0.8, color='#EF5C1D',edgecolor='black',zorder=3)
    ax3.bar(range(len(year_list)), negative_list, width=0.8, color='#2C82DA',edgecolor='black', zorder=3)
    ax3.tick_params(labelbottom=False, )

    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax2.tick_params(which='minor', length=2)
    ax3.set_xlim(0, len(all_list))
    ax3.set_ylim(-1, 1)

    fig.savefig(main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_extract_f_norm_area7_0.png", dpi=1200)
    plt.close()
    '''

    fig = plt.figure(figsize=(12, 12))

    gs = gridspec.GridSpec(6, 1)  # rows, cols

    labels2 = list('' for i in range(0, len(range(1950, 2020))))

    # boxplot for "forcing" distribution
    ax1 = fig.add_subplot(gs[4:6, 0])
    ax1.hlines(0, 0,len(all_list), lw=0.75,colors='k')

    box = ax1.boxplot(all_list, patch_artist=True, vert=True, whis=1.75,labels=labels,showfliers=False,
                                flierprops=dict(marker='.', markerfacecolor='#F14F17',fillstyle='full',
                                markeredgecolor='#F14F17', markeredgewidth=0.0, markersize=4, linestyle='none'),
                      medianprops=dict(linestyle='-.', linewidth=1, color='#FADE4F'),
                      zorder=5)

    colors = ['#656565' for _ in box['boxes']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlim(0,len(all_list))

    ax1.xaxis.set_major_locator(ticker.IndexLocator(5,1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.tick_params(which='major', axis='x', length=5, width=2)
    ax1.tick_params(which='minor', axis='x', length=3)

    ax1.set_ylim(-0.1025, 0.1025)
    ax1.grid(True, linestyle='dotted', lw=0.75, zorder=0)


    ax2 = ax1.twinx()

    ax2.plot(range(len(counts)), counts, lw=1, marker='o', markersize=3, color='#FA732F',zorder=2)
    ax2.tick_params(labelbottom=False, )

    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax2.tick_params(which='minor', length=2)
    ax2.set_xlim(0, len(all_list))
    ax2.set_ylim(-1, 600)

    ax3 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
    ax3.grid(True, linestyle='dotted', lw=0.75, zorder=0)
    ax3.hlines(0, 0, len(year_list), lw=0.75, colors='k')
    ax3.bar(range(len(year_list)), [0] + positive_list[:-1], width=0.8, color='#EF5C1D',edgecolor='black', zorder=4)
    ax3.bar(range(len(year_list)), [0] + negative_list[:-1], width=0.8, color='#2C82DA',edgecolor='black', zorder=4)
    ax3.tick_params(labelbottom=False, )
    ax3.set_xlim(0, len(all_list))
    ax3.set_ylim(-0.041, 0.041)

    ax4 = fig.add_subplot(gs[0:2, 0], sharex=ax1)
    ax4.grid(True, linestyle='dotted', lw=0.75, zorder=0)
    ax4.hlines(0, 0, len(year_list), lw=0.75, colors='k')
    ax4.bar(range(len(year_list)), [0] + pos_decid_list[:-1], width=0.8, color='#056302',edgecolor='black', zorder=4)
    ax4.bar(range(len(year_list)), [0] + neg_decid_list[:-1], width=0.8, color='#FFC100',edgecolor='black', zorder=4)
    ax4.tick_params(labelbottom=False, )
    ax4.set_xlim(0, len(all_list))
    ax4.set_ylim(-0.205, 0.205)

    # ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax2.tick_params(which='minor', length=2)


    fig.savefig(main_dir + "Dropbox/projects/NAU/landsat_deciduous/data/fire_extract_f_norm_area8_4_{}_{}.png".format(ver, forc_col), dpi=1200)
    plt.close()
