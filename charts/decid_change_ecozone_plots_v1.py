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
    plt.rcParams.update({'font.size': 26, 'font.family': 'Calibri'})
    plt.rcParams['axes.labelweight'] = 'regular'

    dodge = 10

    plotdir = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/"
    outfile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/ecozones_area_summary__a_plot_v3.png"

    forc_file = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/" \
                "ecozones_area_summary_all_2000_2015_v4_30.csv"

    forcing_file = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/" \
                   "ecozones_area_summary_intr_2000_2015_v4_30.csv"

    plotfiles = ["ecozones_area_summary_rec_2000_2015_v3_90.csv",  # recent
                 "ecozones_area_summary_intr_2000_2015_v3_90.csv",  # intermediate
                 "ecozones_area_summary_old_2000_2015_v3_90.csv"]   # old

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

    area_arr = decid_arr.copy()

    for i, plotfile in enumerate(plotfiles):
        file_dicts = Handler(plotdir + plotfile).read_from_csv(return_dicts=True)
        for file_dict in file_dicts:
            for j, zone in enumerate(zones):
                if file_dict['ZONE_NAME'] == zone[0]:
                    print(file_dict)

                    decid_perc = file_dict[decid_diff_names[2]]/(file_dict[area]) * 100.0
                    decid_uperc = file_dict[decid_diff_names[1]]/(file_dict[area]) * 100.0

                    decid_arr[j, i] = decid_perc
                    decid_uarr[j, i] = decid_uperc

                    # tc_arr[j, i] = tc_perc
                    # tc_uarr[j, i] = tc_uperc

        dict_lists[time_periods[i]] = file_dicts

    # zone_names = sorted(list(set(zone_names)))

    fig = plt.figure(figsize=(30, 18))
    gs = gridspec.GridSpec(18, 30)  # rows, cols

    ax1 = fig.add_subplot(gs[1:8, 1:11])

    X = time_periods
    Y = decid_arr
    yerr = decid_uarr

    Dodge = np.arange(len(Y), dtype=float) * dodge
    Dodge -= Dodge.mean()

    print(Dodge)

    ii = 0
    for y, d in zip(Y, Dodge):

        draw_dodge(ax1.plot, X, y,
                   ax=ax1, dodge=d,
                   color=(colors[zones[ii][1] - 1]),
                   linewidth=2.5,
                   zorder=-1,
                   alpha=1)
        ii += 1

    ii = 0
    for y, d in zip(Y, Dodge):

        draw_dodge(ax1.errorbar, X, y, yerr=yerr[ii, :],
                   ax=ax1, dodge=d, marker=markers[zones[ii][1] - 1],
                   label=zones[ii][0],
                   markerfacecolor=colors[zones[ii][1] - 1],
                   markeredgecolor='black',
                   markersize=15,
                   markeredgewidth=2,
                   fmt='.',
                   color=(colors[zones[ii][1] - 1]),
                   capsize=3,
                   zorder=3,
                   alpha=1)
        ii += 1

    ax1.margins(x=0.4)
    ax1.set_ylim(0.0, 100.0)
    ax1.set_xlim(-0.5, 2.5)
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_linewidth(2.0)
    ax1.spines['bottom'].set_linewidth(2.0)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)

    # get handles
    handles, labels = ax1.get_legend_handles_labels()
    # remove the errorbars
    indx = [dict(zones)[label] for label in labels]
    sorted_indx = [indx.index(i+1) for i in range(len(indx))]

    print(indx)
    print(sorted_indx)

    handles = [handles[i][0] for i in sorted_indx]
    # labels = [ labels[i] for i in sorted_indx]   #str(dict(zones)[labels[i]]) + '. ' +

    labels = [labels[i] + " ({})".format(dict(zone_abbr)[labels[i]]) for i in sorted_indx]

    box = ax1.get_position()

    #ax1.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])

    leg = ax1.legend(handles, labels,
                     loc='center left', numpoints=1,
                     markerscale=1.25,
                     shadow=False,
                     fancybox=False,
                     bbox_to_anchor=(2.25, 0.75))

    leg.get_frame().set_linewidth(0.0)

    # ----------------------------------------------------------------------------------

    spr_forc_parr = np.zeros((9,)).copy()
    spr_forc_narr = np.zeros((9,)).copy()
    spr_forc_uparr = spr_forc_parr.copy()
    spr_forc_unarr = spr_forc_narr.copy()

    forc_names = ['spr_pforc', 'spr_nforc', 'spr_upforc', 'spr_unforc']

    file_dicts = Handler(forc_file).read_from_csv(return_dicts=True)
    for file_dict in file_dicts:
        for j, zone in enumerate(zones):
            if file_dict['ZONE_NAME'] == zone[0]:
                print(file_dict)

                spr_forc_parr[j] = file_dict[forc_names[0]] / (file_dict[area])
                spr_forc_uparr[j] = file_dict[forc_names[2]] / (file_dict[area])

                spr_forc_narr[j] = file_dict[forc_names[1]] / (file_dict[area])
                spr_forc_unarr[j] = file_dict[forc_names[3]] / (file_dict[area])

    ax2 = fig.add_subplot(gs[1:8, 12:23])
    nbars = spr_forc_parr.shape[0]

    # set width of bar
    bar_offset = 0.1
    bar_sep = 0.01
    bar_width = 0.35

    bar_loc_arr = np.arange(nbars) * (bar_width + bar_sep) + bar_offset

    bartop = ax2.bar(bar_loc_arr, spr_forc_parr, color=colors, width=bar_width, alpha=0.85, edgecolor='black',
                     linewidth=2.5, error_kw=dict(lw=2, capsize=3, capthick=2), yerr=spr_forc_uparr)
    barbot = ax2.bar(bar_loc_arr, spr_forc_narr, color=colors, width=bar_width, alpha=0.85, edgecolor='black',
                     linewidth=2.5, error_kw=dict(lw=2, capsize=3, capthick=2), yerr=spr_forc_unarr)

    plt.savefig(outfile, dpi=300)






