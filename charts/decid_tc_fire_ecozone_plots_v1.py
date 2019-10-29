from geosoup import Handler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches


if __name__ == '__main__':

    fontsize = 32

    plt.rcParams['legend.handlelength'] = 0
    plt.rcParams['legend.handleheight'] = 0
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams.update({'font.size': fontsize, 'font.family': 'Calibri'})
    plt.rcParams['axes.labelweight'] = 'regular'

    outfile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/ecozones_area_summary__b_plot_v3.png"

    forc_file = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/" \
                "ecozones_area_summary_all_decid_tc_only_2000_2015_v8_30.csv"

    zones = [("Boreal Plain", 3), ("Boreal Shield East", 9), ("Boreal Shield West", 6),
             ("Hudson Plain", 7), ("Boreal Cordillera", 4), ("Taiga Shield East", 8), ("Taiga Shield West", 5),
             ("Taiga Cordillera", 1), ("Taiga Plain", 2)]

    zone_abbr = [("Boreal Plain",  "BP"), ("Boreal Shield East",  "BSE"), ("Boreal Shield West", "BSW"),
             ("Hudson Plain",  "HP"), ("Boreal Cordillera",  "BC"), ("Taiga Shield East",  "TSE"),
             ("Taiga Shield West",  "TSW"), ("Taiga Cordillera", "TC"), ("Taiga Plain",  "TP")]

    hatch_pattern = ('', '//', 'XX')

    decid_names = ['decidcover2000', 'evergreencover2000']
    fire_names = ['old_fire_mask', 'interm_fire_mask', 'recent_fire_mask']

    colors = ['#6C327A', '#62E162', '#2F8066', '#F56D94', '#3E9FEF', '#A0B48C', '#764728', '#EF963E', '#A45827']

    area = 'area'

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(10, 22)  # rows, cols

    # -----------------------------------------------------------------------------------------------------------

    scale = 1e11

    ylim = (-5.0, 3.0)

    decid_cparr = np.zeros((9,), dtype=np.float64)
    decid_cnarr = np.zeros((9,), dtype=np.float64)

    decid_ucparr = decid_cparr.copy()
    decid_ucnarr = decid_cnarr.copy()

    file_dicts = Handler(forc_file).read_from_csv(return_dicts=True)

    for file_dict in file_dicts:
        for j, zone in enumerate(zones):
            if file_dict['ZONE_NAME'] == zone[0]:
                # print(file_dict)
                decid_cparr[j] = file_dict[decid_names[0]] / (scale * 1e4)
                decid_cnarr[j] = file_dict[decid_names[1]] / (scale * 1e4)

                decid_ucparr[j] = (file_dict[decid_names[0]] / (scale * 1e4)) * 0.33
                decid_ucnarr[j] = (file_dict[decid_names[1]] / (scale * 1e4)) * 0.25

    # zone_names = sorted(list(set(zone_names)))
    ax1 = fig.add_subplot(gs[1:9, 1:10])

    nbars = decid_cparr.shape[0]

    # set width of bar
    bar_offset = 0.2
    bar_sep = 0.125
    bar_width = 0.3
    bar_close_sep = 0.05

    bar_loc_decid_arr = np.arange(nbars) * (bar_width + bar_sep) + bar_offset

    colors1 = ['#EEC10B', '#5A2D09']
    colors2 = ['#046E03', '#7C388D']

    barbot = ax1.bar(bar_loc_decid_arr, -decid_cnarr, color=colors2[0], width=bar_width, alpha=1, edgecolor='none',
                     linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=decid_ucnarr)

    bartop = ax1.bar(bar_loc_decid_arr, decid_cparr, color=colors1[0], width=bar_width, alpha=1, edgecolor='none',
                     linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=decid_ucparr)

    # ax3.spines['left'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    ax1.xaxis.set_ticks_position('none')
    # ax3.yaxis.set_ticks_position('none')
    ax1.tick_params(labelbottom=False, labeltop=False, labelright=False)
    ax1.hlines(0, -0.1, 4, color='k', lw=2.0)

    ax1.set_ylim(*ylim)

    ax1.yaxis.set_tick_params(width=2)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['left'].set_position(('data', -0.1))
    ax1.locator_params(axis='y', nbins=4)
    ax1.ticklabel_format(axis='y')

    ax1.set_ylabel('Area ($10^{7}$ Hectares)', fontsize=fontsize,labelpad=20)  # $\Delta$
    labels = list(str(elem) for elem in [0, 4, 2, 0, 2])
    ax1.set_yticklabels(labels)

    minx = 2
    maxy = ylim[1]

    patch_hght = 0.25 * (maxy/2)
    patch_wdth = 0.4
    pathc_dist = 0.1

    # for j, zone in enumerate(zones):
    #    ax1.text(bar_loc_decid_arr[j], ylim[0] * 1.025, zone_abbr[j][1], rotation=90,
    #             horizontalalignment='center', verticalalignment='top', fontsize=fontsize)

    for j, zone in enumerate(zones):
        ax1.text(bar_loc_decid_arr[j], ylim[0] * 1.025, zone_abbr[j][1], rotation=90,
                 horizontalalignment='center', verticalalignment='top', fontsize=fontsize)

    ax1.add_patch(patches.Rectangle((2.1, maxy * 0.99), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#EEC10B'))

    ax1.add_patch(patches.Rectangle((2.1, maxy - (maxy/2.5) * (patch_hght + pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#046E03'))

    ax1.text(2.6, maxy * 0.99, 'Deciduous', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax1.text(2.6, maxy - (maxy/2.5)*(patch_hght + pathc_dist), 'Evergreen', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax1.text(0.25, maxy * 1.25, 'a', horizontalalignment='left',
             verticalalignment='top', fontsize=54, fontweight='bold')

    # ----------------------------------------------------------------------------------

    scale = 1e11
    ylim = (-0.025, 2.9)

    fire_old = np.zeros((9,))
    fire_intr = np.zeros((9,))
    fire_rec = np.zeros((9,))

    colors = {'old': '#8F8F8F',
              'intr': '#C65D2B',
              'rec': '#456EDE'}

    file_dicts = Handler(forc_file).read_from_csv(return_dicts=True)

    for file_dict in file_dicts:
        for j, zone in enumerate(zones):
            if file_dict['ZONE_NAME'] == zone[0]:
                fire_old[j] = file_dict[fire_names[0]]/scale
                fire_intr[j] = file_dict[fire_names[1]]/scale
                fire_rec[j] = file_dict[fire_names[2]]/scale

    ax2 = fig.add_subplot(gs[1:9, 11:20])
    nbars = fire_old.shape[0]

    # set width of bar
    bar_offset = 0.2
    bar_sep = 0.125
    bar_width = 0.3
    bar_close_sep = 0.05

    bar_loc_arr = np.arange(nbars) * (bar_width + bar_sep) + bar_offset

    # print(bar_loc_sum_arr)
    # print(bar_loc_spr_arr)

    barbot = ax2.bar(bar_loc_arr, fire_old, color=colors['old'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0)
    for bar in barbot:
        bar.set_hatch(hatch_pattern[0])

    barmid = ax2.bar(bar_loc_arr, fire_intr, bottom=fire_old, color=colors['intr'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0)
    for bar in barmid:
        bar.set_hatch(hatch_pattern[1])

    bartop = ax2.bar(bar_loc_arr, fire_rec, bottom=np.add(fire_intr, fire_old), color=colors['rec'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0)
    for bar in bartop:
        bar.set_hatch(hatch_pattern[2])

    print(bar_loc_arr)
    print(fire_old)
    print(fire_intr)
    print(fire_rec)

    # ax3.spines['left'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.xaxis.set_ticks_position('none')
    # ax3.yaxis.set_ticks_position('none')
    ax2.tick_params(labelbottom=False, labeltop=False, labelright=False)
    ax2.hlines(0, -0.1, 4, color='k', lw=2.0)

    ax2.set_ylim(*ylim)

    ax2.yaxis.set_tick_params(width=2)
    ax2.spines['left'].set_linewidth(2.5)
    ax2.spines['left'].set_position(('data', -0.1))
    ax2.locator_params(axis='y', nbins=4)
    ax2.ticklabel_format(axis='y')

    ax2.set_ylabel('Fire area ($10^{7}$ Hectares)', fontsize=fontsize, labelpad=20)

    minx = 2
    maxy = ylim[1]

    patch_hght = 0.25 * (maxy/4)
    patch_wdth = 0.4
    pathc_dist = 0.1

    for j, zone in enumerate(zones):
        ax2.text(bar_loc_arr[j], ylim[0] * 1.025, zone_abbr[j][1], rotation=90,
                 horizontalalignment='center', verticalalignment='top', fontsize=fontsize)

    # for j, zone in enumerate(zones):
    #    ax1.text(bar_loc_decid_arr[j], ylim[0] * 1.025, zone_abbr[j][1], rotation=90,
    #             horizontalalignment='center', verticalalignment='top', fontsize=fontsize)

    pt1 = ax2.add_patch(patches.Rectangle((2.1, maxy * 0.99), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor=colors['rec']))
    pt1.set_hatch(hatch_pattern[2])

    pt2 = ax2.add_patch(patches.Rectangle((2.1, maxy - 1.1 * (maxy/3.5) * (patch_hght + pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor=colors['intr']))
    pt2.set_hatch(hatch_pattern[1])

    pt3 = ax2.add_patch(patches.Rectangle((2.1, maxy - 2.1 * (maxy/3.5) * (patch_hght + pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor=colors['old']))
    pt3.set_hatch(hatch_pattern[0])

    ax2.text(2.6, maxy * 0.99, 'Recent', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax2.text(2.6, maxy - 1.1 * (maxy/3.5)*(patch_hght + pathc_dist), 'Intermediate', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax2.text(2.6, maxy - 2.1 * (maxy/3.5)*(patch_hght + pathc_dist), 'Old', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax2.text(0.25, maxy * 1.1, 'b', horizontalalignment='left',
             verticalalignment='top', fontsize=54, fontweight='bold')
    # ----------------------------------------------------------------------------------------------------

    plt.savefig(outfile, dpi=300)





