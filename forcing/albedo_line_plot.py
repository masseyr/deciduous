from modules import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch


def moving_average(arr, n=3):
    if n < 1:
        n = 1
    ret = np.cumsum(arr, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    tail = (n-1)/2 if (n % 2 == 1) else n/2

    if tail == 0:
        return arr
    else:
        return np.concatenate([np.array(arr)[0:tail], ret[n - 1:] / n, np.array(arr[-tail:])])


if __name__ == '__main__':

    infolder = 'd:/shared/Dropbox/projects/NAU/landsat_deciduous/data/forcings/'

    # plotfile1 = infolder + 'temp_plot1.png'
    plotfile2 = infolder + 'line_plot1_v3_3.png'
    # plotfile3 = infolder + 'line_plot2.png'

    infile_arr = ['forcing_samples1.csv',
                  'forcing_samples2.csv',
                  'forcing_samples3.csv',
                  'forcing_samples4.csv',
                  'forcing_samples5.csv',
                  'forcing_samples6.csv',
                  'forcing_samples7.csv']

    start_year = 1950
    end_year = 2019

    avg_window = 5

    xlim = (start_year, end_year-1)
    ylim = (-0.65, 0.55)

    plt.rcParams.update({'font.size': 20, 'font.family': 'Calibri'})
    plt.rcParams['axes.labelweight'] = 'regular'

    dict_list = list()

    for infile in infile_arr:
        infile = infolder + infile
        if Handler(infile).file_exists():
            dict_list += Handler(infile).read_from_csv(return_dicts=True)

    for dict_ in dict_list[0:10]:
        print(dict_)

    print(len(dict_list))

    full_dicts = list(elem for elem in dict_list if elem['area_calc'] > 0.0)

    print(len(full_dicts))

    years = list(elem['year'] for elem in full_dicts)

    hist, _ = np.histogram(years, bins=range(start_year, end_year))

    print(hist)

    '''
    plt.hist(years, bins=range(1915,2025), edgecolor='k')
    
    plt.savefig(plotfile1,dpi=300)
    plt.close()
    '''

    years = list(elem['year'] for elem in full_dicts)

    binned_dicts = list(list() for _ in range(start_year, end_year))

    sum_mean = list()
    sum_sd = list()

    spr_mean = list()
    spr_sd = list()

    fall_mean = list()
    fall_sd = list()

    decid_diff = list()
    decid_sd = list()
    decid_udiff = list()

    tc_diff = list()
    tc_sd = list()
    tc_udiff = list()

    for j, i in enumerate(range(start_year, end_year)):
        temp_list = list()
        for elem in full_dicts:
            if elem['year'] == i:
                binned_dicts[j].append(elem)

        sum_mean.append(np.mean(list(elem['sum_forc'] for elem in binned_dicts[j])))
        sum_sd.append(np.std(list(elem['sum_forc'] for elem in binned_dicts[j])))

        spr_mean.append(np.mean(list(elem['spr_forc'] for elem in binned_dicts[j])))
        spr_sd.append(np.std(list(elem['spr_forc'] for elem in binned_dicts[j])))

        fall_mean.append(np.mean(list(elem['fall_forc'] for elem in binned_dicts[j])))
        fall_sd.append(np.std(list(elem['fall_forc'] for elem in binned_dicts[j])))

        decid_diff.append(np.mean(list(elem['decid_diff'] for elem in binned_dicts[j])))
        decid_sd.append(np.std(list(elem['decid_diff'] for elem in binned_dicts[j])))
        decid_udiff.append(np.mean(list(elem['decid_udiff'] for elem in binned_dicts[j])))

        tc_diff.append(np.mean(list(elem['tc_diff'] for elem in binned_dicts[j])))
        tc_sd.append(np.std(list(elem['tc_diff'] for elem in binned_dicts[j])))
        # tc_udiff.append(np.mean(list(elem['tc_udiff'] for elem in binned_dicts[j])))

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(6, 1)  # rows, cols

    ax1 = fig.add_subplot(gs[3:6, 0])
    # ax1.set_ylabel('Change in shortwave\n albedo radiative forcing (W/$\mathregular{m^2}$)')
    ax1.set_ylabel('$\Delta$ Radiative $\mathregular{forcing_{\ SW\ albedo}}$ (W/$\mathregular{m^2}$)')
    ax1.set_xlabel('Fire occurrence date')
    ax1.hlines(0, start_year-2, end_year+1, color='k', lw=1.5)

    ax1.plot(range(start_year, end_year), moving_average(spr_mean, avg_window), color='#066001', ls='-.',lw=1.5)
    ax1.fill_between(range(start_year, end_year),
                     moving_average(np.array(spr_mean) - np.array(spr_sd), avg_window),
                     moving_average(np.array(spr_mean) + np.array(spr_sd), avg_window),
                     color='#066001',
                     lw=0,
                     alpha=.2)

    ax1.plot(range(start_year, end_year), moving_average(fall_mean, avg_window), color='#0350D8', ls='--',lw=1.5)
    ax1.fill_between(range(start_year, end_year),
                     moving_average(np.array(fall_mean) - np.array(fall_sd), avg_window),
                     moving_average(np.array(fall_mean) + np.array(fall_sd), avg_window),
                     color='#0350D8',
                     lw=0,
                     alpha=.3)

    ax1.plot(range(start_year, end_year), moving_average(sum_mean, avg_window), color='#DE3503', lw=1.5)
    ax1.fill_between(range(start_year, end_year),
                     moving_average(np.array(sum_mean) - np.array(sum_sd), avg_window),
                     moving_average(np.array(sum_mean) + np.array(sum_sd), avg_window),
                     color='#EA6A21',
                     lw=0,
                     alpha=.4)

    ax1.set_ylim(*ylim)
    ax1.set_xlim(xlim[0]-2, xlim[1]+2)

    ax1.locator_params(axis='y', nbins=4)
    ax1.locator_params(axis='x', nbins=8)
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_linewidth(2.0)
    ax1.spines['bottom'].set_linewidth(2.5)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)

    # ax1.vlines([2000, 2015], ylim[0], ylim[1], colors='k',linestyles='dotted', lw=2)

    ax1.add_patch(patches.Rectangle((1950, 0.5), 5, -0.05,
                             fill=True, alpha=0.2,
                             edgecolor='none', facecolor='#066001'))
    ax1.plot([1950, 1955], [0.5-(0.05/2),0.5-(0.05/2)], color='#066001', ls='-.',lw=1.5)
    ax1.text(1956, 0.5, 'Spring',
             horizontalalignment='left',
             verticalalignment='top', fontsize=16 )

    ax1.add_patch(patches.Rectangle((1950, 0.43), 5, -0.05,
                                    fill=True, alpha=0.3,
                                    edgecolor='none', facecolor='#DE3503'))
    ax1.plot([1950, 1955], [0.43 - (0.05/2), 0.43 - (0.05/2)], color='#EA6A21', ls='-',lw=1.5)
    ax1.text(1956, 0.43, 'Summer',
             horizontalalignment='left',
             verticalalignment='top',fontsize=16  )

    ax1.add_patch(patches.Rectangle((1950, 0.36), 5, -0.05,
                                    fill=True, alpha=0.4,
                                    edgecolor='none', facecolor='#0350D8'))
    ax1.plot([1950, 1955], [0.36 - (0.05/2), 0.36 - (0.05/2)], color='#0350D8', ls='--',lw=1.5)
    ax1.text(1956, 0.36, 'Fall',
             horizontalalignment='left',
             verticalalignment='top',fontsize=16  )

    ax2 = fig.add_subplot(gs[0:3, 0], sharex=ax1)
    #ax2.set_ylabel('Fractional change')
    ax2.set_ylabel('$\Delta$ Forest composition')

    ax2.hlines(0, start_year-2, end_year+1, color='k', lw=1.5)

    # ax2.vlines([2000, 2015], ylim[0], ylim[1], colors='k', linestyles='dotted', lw=2)

    ax2.plot(range(start_year, end_year), moving_average(tc_diff, avg_window), color='#0C7287', lw=1.5)
    ax2.fill_between(range(start_year, end_year),
                     moving_average(np.array(tc_diff) - np.array(tc_sd), avg_window),
                     moving_average(np.array(tc_diff) + np.array(tc_sd), avg_window),
                     color='#0C7287',
                     lw=0,
                     alpha=.4)

    ax2.plot(range(start_year, end_year), moving_average(decid_diff, avg_window), color='#CC6C0A', ls='-.',lw=1.5)
    ax2.fill_between(range(start_year, end_year),
                     moving_average(np.array(decid_diff) - np.array(decid_sd), avg_window),
                     moving_average(np.array(decid_diff) + np.array(decid_sd), avg_window),
                     color='#CC6C0A',
                     lw=0,
                     alpha=.3)

    ax2.add_patch(patches.Rectangle((1950, 0.3), 5, -0.03,
                             fill=True, alpha=0.2,
                             edgecolor='none', facecolor='#CC6C0A'))
    ax2.plot([1950, 1955], [0.3-(0.03/2),0.3-(0.03/2)], color='#CC6C0A', ls='-.',lw=1.5)
    ax2.text(1956, 0.30, 'Deciduous fraction',
             horizontalalignment='left',
             verticalalignment='top', fontsize=16 )

    ax2.add_patch(patches.Rectangle((1950, 0.26), 5, -0.03,
                                    fill=True, alpha=0.3,
                                    edgecolor='none', facecolor='#0C7287'))
    ax2.plot([1950, 1955], [0.26 - (0.03/2), 0.26 - (0.03/2)], color='#0C7287', ls='-',lw=1.5)
    ax2.text(1956, 0.26, 'Fractional tree cover',
             horizontalalignment='left',
             verticalalignment='top', fontsize=16 )

    ax2.set_ylim(-0.35,0.35)
    ax2.yaxis.set_tick_params(width=2)
    ax2.set_xlim(xlim[0]-2, xlim[1]+2)
    ax2.spines['left'].set_linewidth(2.0)
    ax2.locator_params(axis='y', nbins=4)

    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.xaxis.set_ticks_position('none')
    ax2.tick_params(labelbottom=False, )

    con = ConnectionPatch(xyA=[2000, 0.35], xyB=[2000, ylim[0]], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="gray", lw=2, ls='dotted')
    ax2.add_artist(con)

    con = ConnectionPatch(xyA=[2015.5, 0.35], xyB=[2015.5, ylim[0]], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="gray", lw=2, ls='dotted')
    ax2.add_artist(con)
    '''
    plt.fill_between([1998, 2002],
                     [ylim[0], ylim[0]],
                     [ylim[1], ylim[1]],
                     color='#737272',
                     lw=0,
                     alpha=.2)

    plt.fill_between([2013, 2017],
                     [ylim[0], ylim[0]],
                     [ylim[1], ylim[1]],
                     color='#737272',
                     lw=0,
                     alpha=.2)
    '''

    rct = plt.Rectangle((1998, 1.0), 4, height=-2.06,
                     transform=ax2.get_xaxis_transform(), clip_on=False,
                     edgecolor="none", facecolor="#686968", alpha=.15)
    ax2.add_patch(rct)

    rct2 = plt.Rectangle((2013, 1.0), 5, height=-2.06,
                     transform=ax2.get_xaxis_transform(), clip_on=False,
                     edgecolor="none", facecolor="#686968", alpha=.15)
    ax2.add_patch(rct2)

    fig.savefig(plotfile2,dpi=300)

    plt.close()









