from geosoup import Sublist, Handler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import json




# draw_dodge(ax.errorbar, X, y, yerr =y/4., ax=ax, dodge=d, marker="d" )
def draw_dodge(*args, **kwargs):
    """
    Method to displace error bars by a small amount
    """
    func = args[0]
    dodge = kwargs.pop("dodge", 0)
    _ax = kwargs.pop("ax", plt.gca())
    _trans = _ax.transData + transforms.ScaledTranslation(dodge/125.,
                                                          0,
                                                          _ax.figure.dpi_scale_trans)
    _artist = func(*args[1:], **kwargs)

    def iterate(_artist):
        if hasattr(_artist, '__iter__'):
            for obj in _artist:
                iterate(obj)
        else:
            _artist.set_transform(_trans)
    iterate(_artist)
    return _artist


if __name__ == '__main__':

    plt.rcParams['legend.handlelength'] = 0
    plt.rcParams['legend.handleheight'] = 0
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams.update({'font.size': 38, 'font.family': 'Calibri'})
    plt.rcParams['axes.labelweight'] = 'regular'

    dodge = 10

    plotdir = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/"

    outfile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/ecozones_area_summary_plot_v2.png"
    plotfiles = ["ecozones_area_summary_rec_2000_2015_v3_90.csv",  #  recent
                 "ecozones_area_summary_intr_2000_2015_v3_90.csv",  #  intermediate
                 "ecozones_area_summary_old_2000_2015_v3_90.csv",]   #  old

    dict_lists = dict()
    # zone_names = list()

    time_periods = ['Recent', 'Intermediate', 'Old']

    zones = [("Boreal Plain", 3), ("Boreal Shield East", 9), ("Boreal Shield West", 6),
             ("Hudson Plain", 7), ("Boreal Cordillera", 4), ("Taiga Shield East", 8), ("Taiga Shield West", 5),
             ("Taiga Cordillera", 1), ("Taiga Plain", 2)]

    zone_abbr = [("Boreal Plain",  "BP"), ("Boreal Shield East",  "BSE"), ("Boreal Shield West", "BSW"),
             ("Hudson Plain",  "HP"), ("Boreal Cordillera",  "BC"), ("Taiga Shield East",  "TSE"),
             ("Taiga Shield West",  "TSW"), ("Taiga Cordillera", "TC"), ("Taiga Plain",  "TP")]

    markers = ["s", "o", "o", "s", "p", "^", "D", "p", "^"]

    decid_diff_names = ['decid_diff_pos', 'decid_udiff_pos', 'decid_diff_pos_area', 'decid_diff_neg_area']
    tc_diff_names = ['tc_diff_pos', 'tc_udiff_pos', 'tc_diff_pos_area', 'tc_diff_neg_area']

    colors = ['#6C327A', '#62E162', '#2F8066', '#F56D94', '#3E9FEF', '#A0B48C', '#764728', '#EF963E', '#A45827']

    area = 'area'

    decid_arr = np.zeros((9, 3), dtype=np.float64)
    decid_uarr = decid_arr.copy()

    tc_arr = decid_arr.copy()
    tc_uarr = decid_arr.copy()

    area_arr = decid_arr.copy()

    for i, plotfile in enumerate(plotfiles):
        file_dicts = Handler(plotdir + plotfile).read_from_csv(return_dicts=True)
        for file_dict in file_dicts:
            for j, zone in enumerate(zones):
                if file_dict['ZONE_NAME'] == zone[0]:

                    # decid
                    decid_perc = file_dict[decid_diff_names[2]]/(file_dict[area]) * 100.0
                    # decid perc
                    decid_uperc = file_dict[decid_diff_names[1]]/(file_dict[area]) * 100.0
                    # tc
                    tc_perc = file_dict[tc_diff_names[2]] / (file_dict[area])
                    # tc perc
                    tc_uperc = file_dict[tc_diff_names[1]] / (file_dict[area])

                    decid_arr[j, i] = decid_perc
                    decid_uarr[j, i] = decid_uperc

                    tc_arr[j, i] = tc_perc
                    tc_uarr[j, i] = tc_uperc

        #     print(file_dict)
        #     zone_names.append(file_dict['ZONE_NAME'])

        dict_lists[time_periods[i]] = file_dicts

    # zone_names = sorted(list(set(zone_names)))

    fig = plt.figure(figsize=(32, 12))
    gs = gridspec.GridSpec(12, 32)  # rows, cols
    ax1 = fig.add_subplot(gs[1:10, 1:15])

    nbars = decid_arr.shape[0]
    nplots = decid_arr.shape[1]

    # set width of bar
    bar_offset = 0.1
    bar_sep = 0.01
    bar_width = 0.35
    plot_sep = 4

    bar_loc_arr = np.arange(nbars) * (bar_width + bar_sep) + bar_offset
    plot_loc_arr = np.arange(nplots) * plot_sep

    bar_x, bar_y = np.meshgrid(bar_loc_arr, plot_loc_arr)

    bar_locs = bar_x + bar_y

    for i in range(nplots):
        ax1.bar(bar_locs[i, :], decid_arr.T[i, :],
                color=colors, width=bar_width, yerr=decid_uarr.T[i, :],
                error_kw=dict(lw=2, capsize=3, capthick=2),
                edgecolor='black', linewidth=2.5,)
                #label=[abbr[1] for abbr in zone_abbr])

    # Add xticks on the middle of the group bars
    ax1.set_xlabel('Time-periods', fontweight='bold')

    ax1.margins(x=0.4)
    ax1.set_ylim(0.0, 100)
    ax1.set_xlim(-0.5, 12)
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_linewidth(2.0)
    ax1.spines['bottom'].set_linewidth(2.0)

    ax1.xaxis.set_ticklabels(time_periods)
    ax1.set_xticks(bar_locs.mean(1))
    ax1.set_yticks([0, 50, 100])
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)

    #ax1.legend((rects1[0], rects2[0]), ('Men', 'Women'))

    # Create legend & Show graphic
    # plt.legend()
    # plt.show()

    plt.savefig(outfile, dpi=300)
