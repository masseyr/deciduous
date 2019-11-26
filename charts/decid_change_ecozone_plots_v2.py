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

    outfile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/ecozones_area_summary__a_plot_v7.png"

    forc_file = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/decid/" \
                "ecozones_area_summary_all_2000_2015_v4_30.csv"

    zones = [("Boreal Plain", 3), ("Boreal Shield East", 9), ("Boreal Shield West", 6),
             ("Hudson Plain", 7), ("Boreal Cordillera", 4), ("Taiga Shield East", 8), ("Taiga Shield West", 5),
             ("Taiga Cordillera", 1), ("Taiga Plain", 2)]

    zone_abbr = [("Boreal Plain",  "BP"), ("Boreal Shield East",  "BSE"), ("Boreal Shield West", "BSW"),
             ("Hudson Plain",  "HP"), ("Boreal Cordillera",  "BC"), ("Taiga Shield East",  "TSE"),
             ("Taiga Shield West",  "TSW"), ("Taiga Cordillera", "TC"), ("Taiga Plain",  "TP")]

    markers = ["s", "o", "o", "s", "p", "^", "D", "p", "^"]

    decid_diff_names = ['decid_cdiff_pos_area', 'decid_cdiff_neg_area', 'decid_diff_pos_area', 'decid_diff_neg_area']

    colors = ['#6C327A', '#62E162', '#2F8066', '#F56D94', '#3E9FEF', '#A0B48C', '#764728', '#EF963E', '#A45827']

    area = 'area'

    fig = plt.figure(figsize=(30, 28))
    gs = gridspec.GridSpec(28, 30)  # rows, cols

    # -----------------------------------------------------------------------------------------------------------

    scale = 1e11

    ylim = (-2.5, 2.5)

    decid_cparr = np.zeros((9,), dtype=np.float64)
    decid_cnarr = np.zeros((9,), dtype=np.float64)

    decid_ucparr = decid_cparr.copy()
    decid_ucnarr = decid_cnarr.copy()

    file_dicts = Handler(forc_file).read_from_csv(return_dicts=True)

    for file_dict in file_dicts:
        for j, zone in enumerate(zones):
            if file_dict['ZONE_NAME'] == zone[0]:
                # print(file_dict)
                decid_cparr[j] = file_dict[decid_diff_names[0]] / scale
                decid_cnarr[j] = file_dict[decid_diff_names[1]] / scale

                decid_ucparr[j] = (file_dict[decid_diff_names[0]] / scale) * 0.33
                decid_ucnarr[j] = (file_dict[decid_diff_names[1]] / scale) * 0.33

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

    bartop = ax1.bar(bar_loc_decid_arr, decid_cparr, color=colors1[0], width=bar_width, alpha=1, edgecolor='none',
                     linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=decid_ucparr)

    barbot = ax1.bar(bar_loc_decid_arr, -decid_cnarr, color=colors2[0], width=bar_width, alpha=1, edgecolor='none',
                     linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=decid_ucnarr)

    # ax3.spines['left'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    ax1.xaxis.set_ticks_position('none')
    # ax3.yaxis.set_ticks_position('none')
    ax1.tick_params(labelbottom=False, labeltop=False, labelright=False)
    ax1.hlines(0, -0.1, 4, color='k', lw=2.0)

    ax1.set_ylim(-2.5, 2.5)

    ax1.yaxis.set_tick_params(width=2)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['left'].set_position(('data', -0.1))
    ax1.locator_params(axis='y', nbins=4)
    ax1.ticklabel_format(axis='y')

    ax1.set_ylabel('Area ($\mathregular{10^7}$ Hectares)', fontsize=fontsize, labelpad=45)  # $\Delta$
    labels = list(str(elem) for elem in [0, 2, 0, 2])
    ax1.set_yticklabels(labels)

    patch_hght = 0.25
    patch_wdth = 0.4
    pathc_dist = 0.1

    minx = 2
    maxy = ylim[1]

    # for j, zone in enumerate(zones):
    #    ax1.text(bar_loc_decid_arr[j], ylim[0] * 1.025, zone_abbr[j][1], rotation=90,
    #             horizontalalignment='center', verticalalignment='top', fontsize=fontsize)

    ax1.add_patch(patches.Rectangle((2.1, maxy * 0.99), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#EEC10B'))

    ax1.add_patch(patches.Rectangle((2.1, maxy - 1 * (patch_hght + pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#046E03'))

    ax1.text(2.6, maxy * 0.99, 'Increase', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax1.text(2.6, maxy - (patch_hght + pathc_dist), 'Decrease', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    # ----------------------------------------------------------------------------------

    sum_forc_parr = np.zeros((9,)).copy()
    sum_forc_narr = np.zeros((9,)).copy()
    sum_forc_uparr = sum_forc_parr.copy()
    sum_forc_unarr = sum_forc_narr.copy()

    colors = {'pos': '#E8924A',
              'neg': '#0567AB',
              }

    forc_names = ['sum_pforc', 'sum_nforc', 'sum_upforc', 'sum_unforc']

    file_dicts = Handler(forc_file).read_from_csv(return_dicts=True)

    for file_dict in file_dicts:
        for j, zone in enumerate(zones):
            if file_dict['ZONE_NAME'] == zone[0]:
                sum_forc_parr[j] = file_dict[forc_names[0]] / (file_dict[area])
                sum_forc_uparr[j] = file_dict[forc_names[2]] / (file_dict[area]) * 2

                sum_forc_narr[j] = file_dict[forc_names[1]] / (file_dict[area])
                sum_forc_unarr[j] = file_dict[forc_names[3]] / (file_dict[area]) * 1.5

    ax2 = fig.add_subplot(gs[20:29, 1:10])
    nbars = sum_forc_parr.shape[0]

    # set width of bar
    bar_offset = 0.2
    bar_sep = 0.125
    bar_width = 0.3
    bar_close_sep = 0.05

    bar_loc_spr_arr = np.arange(nbars) * (bar_width + bar_sep) + bar_offset

    # print(bar_loc_sum_arr)
    # print(bar_loc_spr_arr)

    bartop = ax2.bar(bar_loc_spr_arr, sum_forc_parr, color=colors['pos'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=sum_forc_uparr)

    barbot = ax2.bar(bar_loc_spr_arr, sum_forc_narr, color=colors['neg'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=sum_forc_unarr)

    # ax3.spines['left'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.xaxis.set_ticks_position('none')
    # ax3.yaxis.set_ticks_position('none')
    ax2.tick_params(labelbottom=False, labeltop=False, labelright=False)
    ax2.hlines(0, -0.1, 4, color='k', lw=2.0)

    ylim_new = (ylim[0] * 0.25, ylim[1] * 0.25)
    ax2.set_ylim(*ylim_new)

    ax2.yaxis.set_tick_params(width=2)
    ax2.spines['left'].set_linewidth(2.5)
    ax2.spines['left'].set_position(('data', -0.1))
    ax2.locator_params(axis='y', nbins=4)
    ax2.ticklabel_format(axis='y')

    ax2.set_ylabel('Summer rad. forcing (W/$\mathregular{m^2}$)', fontsize=fontsize)

    patch_hght = ylim_new[1] / 10.
    patch_wdth = 0.4
    pathc_dist = ylim_new[1] / 25.

    minx = 2
    maxy = ylim_new[1]

    for j, zone in enumerate(zones):
        ax2.text(bar_loc_spr_arr[j], ylim_new[0] * 1.025, zone_abbr[j][1], rotation=90,
                 horizontalalignment='center', verticalalignment='top', fontsize=fontsize)

    ax2.add_patch(patches.Rectangle((2.1, maxy * 0.99), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#E8924A'))

    ax2.add_patch(patches.Rectangle((2.1, maxy - (patch_hght + pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#0567AB'))

    ax2.text(2.6, maxy * 0.99, 'Warming', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax2.text(2.6, maxy - (patch_hght + pathc_dist), 'Cooling', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    # ----------------------------------------------------------------------------------------------------

    spr_forc_parr = np.zeros((9,)).copy()
    spr_forc_narr = np.zeros((9,)).copy()
    spr_forc_uparr = spr_forc_parr.copy()
    spr_forc_unarr = spr_forc_narr.copy()

    colors = {'pos': '#E8924A',
              'neg': '#0567AB',
              }

    forc_names = ['spr_pforc', 'spr_nforc', 'spr_upforc', 'spr_unforc']

    file_dicts = Handler(forc_file).read_from_csv(return_dicts=True)

    for file_dict in file_dicts:
        for j, zone in enumerate(zones):
            if file_dict['ZONE_NAME'] == zone[0]:

                spr_forc_parr[j] = file_dict[forc_names[0]] / (file_dict[area])
                spr_forc_uparr[j] = file_dict[forc_names[2]] / (file_dict[area]) * 12

                spr_forc_narr[j] = file_dict[forc_names[1]] / (file_dict[area])
                spr_forc_unarr[j] = file_dict[forc_names[3]] / (file_dict[area]) * 6

    ax3 = fig.add_subplot(gs[10:19, 1:10])
    nbars = spr_forc_parr.shape[0]

    # set width of bar
    bar_offset = 0.2
    bar_sep = 0.125
    bar_width = 0.3
    bar_close_sep = 0.05

    bar_loc_spr_arr = np.arange(nbars) * (bar_width + bar_sep) + bar_offset

    # print(bar_loc_sum_arr)
    # print(bar_loc_spr_arr)

    bartop = ax3.bar(bar_loc_spr_arr, spr_forc_parr, color=colors['pos'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=spr_forc_uparr)

    barbot = ax3.bar(bar_loc_spr_arr, spr_forc_narr, color=colors['neg'], width=bar_width, alpha=1,
                     edgecolor='none', linewidth=2.0, error_kw=dict(lw=2, capsize=5, capthick=2), yerr=spr_forc_unarr)

    # ax3.spines['left'].set_color('none')
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.spines['bottom'].set_color('none')
    ax3.xaxis.set_ticks_position('none')
    # ax3.yaxis.set_ticks_position('none')
    ax3.tick_params(labelbottom=False, labeltop=False, labelright=False)
    ax3.hlines(0, -0.1, 4, color='k', lw=2.0)

    ax3.set_ylim(*ylim)

    ax3.yaxis.set_tick_params(width=2)
    ax3.spines['left'].set_linewidth(2.5)
    ax3.spines['left'].set_position(('data', -0.1))
    ax3.locator_params(axis='y', nbins=4)
    ax3.ticklabel_format(axis='y')

    ax3.set_ylabel('Spring rad. forcing (W/$\mathregular{m^2}$)', fontsize=fontsize, labelpad=30)

    patch_hght = ylim[1] / 10.
    patch_wdth = 0.4
    pathc_dist = ylim[1] / 25.

    minx = 2
    maxy = ylim[1]

    '''
    for j, zone in enumerate(zones):
        ax3.text(bar_loc_spr_arr[j], ylim[0] * 1.025, zone_abbr[j][1], rotation=90,
                 horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
    '''

    ax3.add_patch(patches.Rectangle((2.1, maxy * 0.99), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#E8924A'))

    ax3.add_patch(patches.Rectangle((2.1, maxy - (patch_hght + pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=1, linewidth=2,
                                    edgecolor='black', facecolor='#0567AB'))

    ax3.text(2.6, maxy * 0.99, 'Warming', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    ax3.text(2.6, maxy - (patch_hght + pathc_dist), 'Cooling', horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize)

    # -----------------------------------------------------------------------------------------------------------


    plt.savefig(outfile, dpi=300)





