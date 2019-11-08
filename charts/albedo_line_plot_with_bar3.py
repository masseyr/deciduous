from geosoup import Sublist, Handler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import json


moving_average = Sublist.moving_average


data_dict = {
    # total area boreal m-sq
    'area': 6255893852434.762,

    # change in decid area overall
    'pos_decid_area': 1580082013207.2942,
    'neg_decid_area': 2636083247853.748,
    'pos_tc_area': 1926498130680.639,
    'neg_tc_area': 2307520492626.951,

    # total area spring
    'pos_spr_area': 2234235288351.595,
    'neg_spr_area': 1855789275157.6775,

    # total area summer
    'pos_sum_area': 1966018147982.6433,
    'neg_sum_area': 1072543112058.4097,

    # total area fall
    'pos_fall_area': 2179395143382.9397,
    'neg_fall_area': 1868060312004.0847,

    'thresh': 0.1,

    'area_tc_gt_25': 3978564035924.3096,  # 397.8 Mha
    'area_tc_gt_25_fire': 971807350730.3446,  # 97.2 Mha

    'recent_fire_area': 378818541818.7525,   # 37.9 Mha
    'intermediate_fire_area': 329868161414.7575,   # 32.9 Mha
    'old_fire_area': 200592087315.63504,   # 20.1 Mha
}


def get_lims(elem_list, multiplier=1.0):

    elem_min = np.min(elem_list)
    elem_max = np.max(elem_list)
    elem_mean = np.mean(elem_list)
    elem_sd = np.std(elem_list)

    u_sd = elem_mean + (elem_sd * multiplier)
    l_sd = elem_mean - (elem_sd * multiplier)

    if elem_max > u_sd:
        elem_max = u_sd

    if elem_min < l_sd:
        elem_min = l_sd

    return elem_max, elem_mean, elem_min


if __name__ == '__main__':

    plotfile = 'd:/shared/Dropbox/projects/NAU/landsat_deciduous/data/forcings/line_plot1_v5_test2.png'

    data_file = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/forcings/" \
            "chart_2_data_scale_250_lim_0_1_5_11_2020-Copy2.txt"
    # "chart_2_data_scale_250_lim_0_1_5_12_2020.txt"
    # "chart_2_data_scale_250_lim_0_1_5_11_2020.txt"
    # "chart_2_data_scale_500_lim_0_1_5_9_2020.txt"
    # "chart_2_data_scale_1000_lim_0_15_5_1_2020.txt"

    decfolder = 'd:/shared/Dropbox/projects/NAU/landsat_deciduous/data/forcings/'
    decfile_arr = ['forcing_samples1.csv',
                   'forcing_samples2.csv',
                   'forcing_samples3.csv',
                   'forcing_samples4.csv',
                   'forcing_samples5.csv',
                   'forcing_samples6.csv',
                   'forcing_samples7.csv']

    infolder = 'd:/shared/Dropbox/projects/NAU/landsat_deciduous/data/forcings/alb_samples3/'
    infile_arr = ['alb_samples1.csv',
                  'alb_samples2.csv',
                  'alb_samples3.csv',
                  'alb_samples4.csv',
                  'alb_samples5.csv',
                  'alb_samples6.csv',
                  'alb_samples7.csv']

    # ---------------------------------------------------------------------------------------
    fontsize1 = 30
    fontsize2 = 26
    fontsize3 = 26

    plt.rcParams.update({'font.size': 20, 'font.family': 'Calibri',
                         'hatch.color': 'k', 'hatch.linewidth': 0.5})
    plt.rcParams['axes.labelweight'] = 'regular'
    plt.rcParams['hatch.linewidth'] = 1.0

    hatch_pattern = ('', '//', 'xx')

    # ---------------------------------------------------------------------------------------

    start_year = 1950
    end_year = 2019

    avg_window = 5

    # ---------------------------------------------------------------------------------------

    data_lines = Handler(data_file).read_text_by_line()

    data_dict.update(json.loads(data_lines[0]))


    xlim = (start_year, end_year-1)
    ylim = (-4., 1.)
    ylim_new = [ylim[0]*1.25, ylim[1]*1.25]

    maxy = ylim[1]-0.05
    minx = xlim[0]

    decid_ylim = (-0.35, 0.35)
    decid_miny = decid_ylim[1] - 0.05

    years = list(range(start_year, end_year))
    n_years = len(years)

    # ---------------------------------------------------------------------------------------
    # albedo difference data

    # albedo difference arrays
    spr_arr = np.zeros((3, n_years))
    sum_arr = np.zeros((3, n_years))
    fall_arr = np.zeros((3, n_years))

    alb_dict = dict()
    for infile in infile_arr:
        infile = infolder + infile
        if Handler(infile).file_exists():
            tmp_dict_list = Handler(infile).read_from_csv(return_dicts=True)
            for tmp_dict in tmp_dict_list:
                if tmp_dict['area_calc'] > 0.0 and start_year <= int(tmp_dict['year']) <= end_year:
                    if int(tmp_dict['year']) in alb_dict:
                        alb_dict[int(tmp_dict['year'])].append(tmp_dict)
                    else:
                        alb_dict[int(tmp_dict['year'])] = [tmp_dict]

    for indx, year in enumerate(years):
        if year in alb_dict:
            spr_list = list(float(elem['spr_diff']) * 0.01 for elem in alb_dict[year])
            spr_arr[:, indx] = get_lims(spr_list)

            sum_list = list(elem['sum_diff'] * 0.01 for elem in alb_dict[year])
            sum_arr[:, indx] = get_lims(sum_list)

            fall_list = list(elem['fall_diff'] * 0.01 for elem in alb_dict[year])
            fall_arr[:, indx] = get_lims(fall_list)

    # ---------------------------------------------------------------------------------------
    # forest composition arrays

    # decid composition arrays
    decid_arr = np.zeros((3, n_years))
    tc_arr = np.zeros((3, n_years))

    dec_dict = dict()
    for infile in decfile_arr:
        infile = decfolder + infile
        if Handler(infile).file_exists():
            temp_dict_list = Handler(infile).read_from_csv(return_dicts=True)
            for tmp_dict in temp_dict_list:
                if tmp_dict['area_calc'] > 0.0 and start_year <= int(tmp_dict['year']) <= end_year:
                    if int(tmp_dict['year']) in dec_dict:
                        dec_dict[int(tmp_dict['year'])].append(tmp_dict)
                    else:
                        dec_dict[int(tmp_dict['year'])] = [tmp_dict]

    for indx, year in enumerate(years):
        if year in dec_dict:
            decid_list = list(float(elem['decid_diff']) for elem in dec_dict[year])
            decid_arr[:, indx] = get_lims(decid_list)

            tc_list = list(elem['tc_diff'] for elem in dec_dict[year])
            tc_arr[:, indx] = get_lims(tc_list)

    # ---------------------------------------------------------------------------------------
    # Figure begin

    fig = plt.figure(figsize=(28, 12))
    gs = gridspec.GridSpec(13, 44)  # rows, cols

    # ---------------------------------------------------------------------------------------
    # Albedo line plots
    alb_ylim = [-0.18, 0.18]

    y_scale = np.abs(alb_ylim[0]-alb_ylim[1])
    y_disp = -y_scale/10
    patch_hght = 0.075 * y_scale
    patch_wdth = 5
    pathc_dist = 0.02 * y_scale
    maxy = alb_ylim[1] + y_disp

    ax1 = fig.add_subplot(gs[7:13, 0:13])
    ax1.set_ylabel('Albedo', fontsize=fontsize1, labelpad=10)
    ax1.set_xlabel('Fire occurrence date', fontsize=fontsize1)
    ax1.hlines(0, start_year-2, end_year+1, color='k', lw=1.5)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

    ax1.plot(years, moving_average(spr_arr[1, :], avg_window, True), color='#054C01', ls='-.', lw=1.5)
    ax1.fill_between(years,
                     moving_average(spr_arr[0, :], avg_window, True),
                     moving_average(spr_arr[2, :], avg_window, True),
                     color='#066001',
                     lw=0,
                     alpha=.3)

    ax1.plot(years, moving_average(sum_arr[1, :], avg_window, True), color='#912303', lw=1.5)
    ax1.fill_between(years,
                     moving_average(sum_arr[0, :], avg_window, True),
                     moving_average(sum_arr[2, :], avg_window, True),
                     color='#EA6A21',
                     lw=0,
                     alpha=.4)

    '''
    ax1.plot(years, moving_average(fall_arr[1, :], avg_window, True), color='#02348D', ls='--', lw=1.5)
    ax1.fill_between(years,
                     moving_average(fall_arr[0, :], avg_window, True),
                     moving_average(fall_arr[2, :], avg_window, True),
                     color='#0350D8',
                     lw=0,
                     alpha=.3)
    '''

    ax1.set_ylim(*alb_ylim)
    ax1.set_xlim(xlim[0]-2, xlim[1]+2)

    ax1.locator_params(axis='y', nbins=4)
    ax1.locator_params(axis='x', nbins=8)
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_linewidth(2.0)
    ax1.spines['bottom'].set_linewidth(2.5)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)

    # ---------------------------------------------------------------------------------------
    # alb fig legend patches

    # spring patch
    ax1.add_patch(patches.Rectangle((minx, maxy), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.3,
                                    edgecolor='none', facecolor='#066001'))
    ax1.plot([minx, minx+patch_wdth], [maxy - (patch_hght / 2), maxy - (patch_hght / 2)],
             color='#054C01', ls='-.', lw=1.5)
    ax1.text(minx+patch_wdth+1, maxy, 'Spring',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    # summer patch
    ax1.add_patch(patches.Rectangle((1950, maxy-1*(patch_hght+pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.4,
                                    edgecolor='none', facecolor='#EA6A21'))
    ax1.plot([minx, minx+patch_wdth],
             [maxy - 1 * (patch_hght+pathc_dist) - (patch_hght / 2),
              maxy - 1 * (patch_hght+pathc_dist) - (patch_hght / 2)],
             color='#912303', ls='-', lw=1.5)
    ax1.text(minx+patch_wdth+1, maxy-1*(patch_hght+pathc_dist), 'Summer',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    # fall patch
    '''
    ax1.add_patch(patches.Rectangle((minx, maxy-2*(patch_hght+pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.4,
                                    edgecolor='none', facecolor='#0350D8'))
    ax1.plot([minx, minx+patch_wdth],
             [maxy - 2 * (patch_hght+pathc_dist) - (patch_hght / 2),
              maxy - 2 * (patch_hght+pathc_dist) - (patch_hght / 2)],
             color='#02348D', ls='--', lw=1.5)
    ax1.text(minx+patch_wdth+1, maxy-2*(patch_hght+pathc_dist), 'Fall',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)
    '''
    # ---------------------------------------------------------------------------------------
    # decid and tc line plots

    ax2 = fig.add_subplot(gs[0:6, 0:13], sharex=ax1)
    ax2.set_ylabel('$\Delta$ Forest composition', fontsize=fontsize1)

    ax2.hlines(0, start_year-2, end_year+1, color='k', lw=1.5)

    ax2.plot(years, moving_average(tc_arr[1, :], avg_window, True),
             color='#035263', lw=1.5)
    ax2.fill_between(years,
                     moving_average(tc_arr[0, :], avg_window, True),
                     moving_average(tc_arr[2, :], avg_window, True),
                     color='#0C7287',
                     lw=0,
                     alpha=.4)

    ax2.plot(years, moving_average(decid_arr[1, :], avg_window, True),
             color='#A55606', ls='-.', lw=1.5)
    ax2.fill_between(years,
                     moving_average(decid_arr[0, :], avg_window, True),
                     moving_average(decid_arr[2, :], avg_window, True),
                     color='#CC6C0A',
                     lw=0,
                     alpha=.3)

    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

    ax2.set_ylim(-0.35, 0.35)
    ax2.yaxis.set_tick_params(width=2)
    ax2.set_xlim(xlim[0]-2, xlim[1]+2)
    ax2.spines['left'].set_linewidth(2.0)
    ax2.locator_params(axis='y', nbins=4)

    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.xaxis.set_ticks_position('none')
    ax2.tick_params(labelbottom=False, )

    # ---------------------------------------------------------------------------------------
    # decid and tc line plot patches

    patch_hght = 0.05
    patch_wdth = 5
    pathc_dist = 0.01

    ax2.add_patch(patches.Rectangle((minx, decid_miny), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.2,
                                    edgecolor='none', facecolor='#CC6C0A'))
    ax2.plot([minx, minx+patch_wdth], [decid_miny-(patch_hght/2),
                                       decid_miny-(patch_hght/2)], color='#A55606', ls='-.', lw=1.5)
    ax2.text(minx+patch_wdth+1, decid_miny, 'Deciduous fraction',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    ax2.add_patch(patches.Rectangle((minx, decid_miny-1*(patch_hght+pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.3,
                                    edgecolor='none', facecolor='#0C7287'))
    ax2.plot([minx, minx+patch_wdth], [decid_miny-1*(patch_hght+pathc_dist) - (patch_hght/2),
                                       decid_miny - 1*(patch_hght+pathc_dist) - (patch_hght/2)],
             color='#035263', ls='-', lw=1.5)
    ax2.text(minx+patch_wdth+1, decid_miny-1*(patch_hght+pathc_dist), 'Tree canopy cover',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    # ---------------------------------------------------------------------------------------
    # decid and tc line plot connection patches with albedo plots

    rct = plt.Rectangle((1998, 1.0), 4, height=-2.2,
                        transform=ax2.get_xaxis_transform(), clip_on=False,
                        edgecolor="none", facecolor="#686968", alpha=.15)
    ax2.add_patch(rct)

    rct2 = plt.Rectangle((2013, 1.0), 5, height=-2.2,
                         transform=ax2.get_xaxis_transform(), clip_on=False,
                         edgecolor="none", facecolor="#686968", alpha=.15)
    ax2.add_patch(rct2)

    con = ConnectionPatch(xyA=[2000, 0.35], xyB=[2000, alb_ylim[0]], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="gray", lw=2, ls='dotted')
    ax2.add_artist(con)

    con = ConnectionPatch(xyA=[2015.5, 0.35], xyB=[2015.5, alb_ylim[0]], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="gray", lw=2, ls='dotted')
    ax2.add_artist(con)
    # ---------------------------------------------------------------------------------------
    # decid tc stacked plots

    ax8 = fig.add_subplot(gs[0:6, 24:28])

    scale = 1e11

    bar_x = [0.1, 0.9]

    print('-------------------------------------------------------------------')
    ax8_keys = ['decid_area_val_lim_rec', 'decid_area_val_0_lim_rec',
                'decid_area_val__lim_0_rec', 'decid_area_val__lim_rec',
                'tc_area_val_lim_rec', 'tc_area_val_0_lim_rec',
                'tc_area_val__lim_0_rec', 'tc_area_val__lim_rec']

    for key in ax8_keys:
        print(key + ': {0:.2f} Mha'.format(data_dict[key]/1e10))
    print('-------------------------------------------------------------------')

    top_band = np.array([data_dict['decid_area_val_lim_rec'], data_dict['tc_area_val_lim_rec']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_rec'], data_dict['tc_area_val_0_lim_rec']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_rec'], -data_dict['tc_area_val__lim_0_rec']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_rec'], -data_dict['tc_area_val__lim_rec']]) / scale

    b1 = ax8.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b2 = ax8.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1, edgecolor='none', linewidth=0, zorder=-1)

    b12 = ax8.bar(bar_x, third_band + last_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b12[0].set_hatch(hatch_pattern[0])
    b12[1].set_hatch(hatch_pattern[1])

    b3 = ax8.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'],  width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b4 = ax8.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1, edgecolor='none', linewidth=0,  zorder=-1)

    b34 = ax8.bar(bar_x, top_band + second_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b34[0].set_hatch(hatch_pattern[0])
    b34[1].set_hatch(hatch_pattern[1])


    ax8.spines['left'].set_color('none')
    ax8.spines['right'].set_color('none')
    ax8.spines['top'].set_color('none')
    ax8.spines['bottom'].set_color('none')
    ax8.xaxis.set_ticks_position('none')
    ax8.yaxis.set_ticks_position('none')
    ax8.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # ax8.hlines(0, -1.25, 1.5, color='None', lw=1.5)
    ax8.hlines(0, -0.5, 1.5, color='k', lw=1.5)

    lim = max([top_band[0] + second_band[0], top_band[1]+ second_band[1],
               -(third_band[0] + last_band[0]), -(third_band[1] + last_band[1])]) * 1.1

    ax8.set_ylim(-lim, lim)
    lim0=lim

    ax8.text(np.mean(bar_x), lim*1.02 , '(iii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    # decid tc stacked plots
    ax9 = fig.add_subplot(gs[0:6, 20:24])

    print('-------------------------------------------------------------------')
    ax9_keys = ['decid_area_val_lim_intr', 'decid_area_val_0_lim_intr',
                'decid_area_val__lim_0_intr', 'decid_area_val__lim_intr',
                'tc_area_val_lim_intr', 'tc_area_val_0_lim_intr',
                'tc_area_val__lim_0_intr', 'tc_area_val__lim_intr']

    for key in ax9_keys:
        print(key + ': {0:.2f} Mha'.format(data_dict[key]/1e10))
    print('-------------------------------------------------------------------')

    top_band = np.array([data_dict['decid_area_val_lim_intr'], data_dict['tc_area_val_lim_intr']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_intr'], data_dict['tc_area_val_0_lim_intr']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_intr'], -data_dict['tc_area_val__lim_0_intr']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_intr'], -data_dict['tc_area_val__lim_intr']]) / scale

    b1 = ax9.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b2 = ax9.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1, edgecolor='none', linewidth=0, zorder=-1)

    b12 = ax9.bar(bar_x, third_band + last_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b12[0].set_hatch(hatch_pattern[0])
    b12[1].set_hatch(hatch_pattern[1])

    b3 = ax9.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'],  width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b4 = ax9.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1, edgecolor='none', linewidth=0,  zorder=-1)

    b34 = ax9.bar(bar_x, top_band + second_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b34[0].set_hatch(hatch_pattern[0])
    b34[1].set_hatch(hatch_pattern[1])

    ax9.spines['left'].set_color('none')
    ax9.spines['right'].set_color('none')
    ax9.spines['top'].set_color('none')
    ax9.spines['bottom'].set_color('none')
    ax9.xaxis.set_ticks_position('none')
    ax9.yaxis.set_ticks_position('none')
    ax9.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    ax9.hlines(0, -0.5, 1.5, color='k', lw=1.5)

    ax9.set_ylim(-lim, lim)
    #ax9.text(np.mean(bar_x), lim*0.96, '1980 - 2000',
    #         horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    ax9.text(np.mean(bar_x), lim*1.02 , '(ii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    # ax9.set_ylabel('$\Delta$ Area ($\mathregular{10^8}$ Hectares)', fontsize=22)


    # decid tc stacked plots
    ax10 = fig.add_subplot(gs[0:6, 16:20])

    print('-------------------------------------------------------------------')
    ax10_keys = ['decid_area_val_lim_old', 'decid_area_val_0_lim_old',
                 'decid_area_val__lim_0_old', 'decid_area_val__lim_old',
                 'tc_area_val_lim_old', 'tc_area_val_0_lim_old',
                 'tc_area_val__lim_0_old', 'tc_area_val__lim_old']

    for key in ax10_keys:
        print(key + ': {0:.2f} Mha'.format(data_dict[key]/1e10))
    print('-------------------------------------------------------------------')

    top_band = np.array([data_dict['decid_area_val_lim_old'], data_dict['tc_area_val_lim_old']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_old'], data_dict['tc_area_val_0_lim_old']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_old'], -data_dict['tc_area_val__lim_0_old']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_old'], -data_dict['tc_area_val__lim_old']]) / scale

    b1 = ax10.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b2 = ax10.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1, edgecolor='none', linewidth=0, zorder=-1)

    b12 = ax10.bar(bar_x, third_band + last_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b12[0].set_hatch(hatch_pattern[0])
    b12[1].set_hatch(hatch_pattern[1])

    b3 = ax10.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'],  width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b4 = ax10.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1, edgecolor='none', linewidth=0,  zorder=-1)

    b34 = ax10.bar(bar_x, top_band + second_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b34[0].set_hatch(hatch_pattern[0])
    b34[1].set_hatch(hatch_pattern[1])

    # ax10.spines['left'].set_color('none')
    ax10.spines['right'].set_color('none')
    ax10.spines['top'].set_color('none')
    ax10.spines['bottom'].set_color('none')
    ax10.xaxis.set_ticks_position('none')
    # ax10.yaxis.set_ticks_position('none')
    ax10.tick_params(labelbottom=False, labeltop=False, labelright=False)

    ax10.hlines(0, -0.5, 1.5, color='k', lw=1.5)

    ax10.yaxis.set_tick_params(width=2)
    ax10.spines['left'].set_linewidth(2.0)
    ax10.spines['left'].set_position(('data', -0.5))
    ax10.locator_params(axis='y', nbins=4)

    ax10.ticklabel_format(axis='y')
    ax10.set_ylabel('Area ($\mathregular{10^7}$ Hectares)', fontsize=fontsize1, labelpad=16)  # $\Delta$
    ax10.set_ylim(-lim, lim)

    labels = list(str(elem) for elem in [0, 2, 0, 2])
    ax10.set_yticklabels(labels)

    for i, tick in enumerate(ax10.yaxis.get_major_ticks()):
        tick.label.set_fontsize(fontsize2)

    ax10.text(np.mean(bar_x), lim*1.02, '(i)',
              horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    # decid tc stacked plots: fire summary
    overall_area = fig.add_subplot(gs[0:6, 35:39])

    scale = 1e12

    bar_x = [0.1, 0.9]

    print('-------------------------------------------------------------------')
    overall_keys = ['decid_area_val_lim_all', 'decid_area_val_0_lim_all',
                    'decid_area_val__lim_0_all', 'decid_area_val__lim_all',
                    'tc_area_val_lim_all', 'tc_area_val_0_lim_all',
                    'tc_area_val__lim_0_all', 'tc_area_val__lim_all']

    for key in overall_keys:
        print(key + ': {0:.2f} Mha'.format(data_dict[key]/1e10))
    print('-------------------------------------------------------------------')

    top_band = np.array([data_dict['decid_area_val_lim_all'], data_dict['tc_area_val_lim_all']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_all'], data_dict['tc_area_val_0_lim_all']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_all'], -data_dict['tc_area_val__lim_0_all']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_all'], -data_dict['tc_area_val__lim_all']]) / scale

    b1 = overall_area.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b2 = overall_area.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1, edgecolor='none', linewidth=0, zorder=-1)

    b12 = overall_area.bar(bar_x, third_band + last_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b12[0].set_hatch(hatch_pattern[0])
    b12[1].set_hatch(hatch_pattern[1])

    b3 = overall_area.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'],  width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b4 = overall_area.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1, edgecolor='none', linewidth=0,  zorder=-1)

    b34 = overall_area.bar(bar_x, top_band + second_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b34[0].set_hatch(hatch_pattern[0])
    b34[1].set_hatch(hatch_pattern[1])

    overall_area.spines['left'].set_color('none')
    overall_area.spines['right'].set_color('none')
    overall_area.spines['top'].set_color('none')
    overall_area.spines['bottom'].set_color('none')
    overall_area.xaxis.set_ticks_position('none')
    overall_area.yaxis.set_ticks_position('none')
    overall_area.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # firesum1.hlines(0, -1.25, 1.5, color='None', lw=1.5)
    overall_area.hlines(0, -0.5, 1.5, color='k', lw=1.5)

    lim = max([top_band[0] + second_band[0], top_band[1]+ second_band[1],
               -(third_band[0] + last_band[0]), -(third_band[1] + last_band[1])]) * 1.1

    overall_area.set_ylim(-lim, lim)
    overall_area.text(np.mean(bar_x), lim*1.02 , '(ii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    firesum1 = fig.add_subplot(gs[0:6, 31:35])

    print('-------------------------------------------------------------------')
    firesum1_keys = ['decid_area_val_lim', 'decid_area_val_0_lim',
                     'decid_area_val__lim_0', 'decid_area_val__lim',
                     'tc_area_val_lim', 'tc_area_val_0_lim',
                     'tc_area_val__lim_0', 'tc_area_val__lim']

    for key in firesum1_keys:
        print(key + ': {0:.2f} Mha'.format(data_dict[key]/1e10))
    print('-------------------------------------------------------------------')

    top_band = np.array([data_dict['decid_area_val_lim'], data_dict['tc_area_val_lim']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim'], data_dict['tc_area_val_0_lim']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0'], -data_dict['tc_area_val__lim_0']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim'], -data_dict['tc_area_val__lim']]) / scale

    b1 = firesum1.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b2 = firesum1.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1, edgecolor='none', linewidth=0, zorder=-1)

    b12 = firesum1.bar(bar_x, third_band + last_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b12[0].set_hatch(hatch_pattern[0])
    b12[1].set_hatch(hatch_pattern[1])

    b3 = firesum1.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'],  width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b4 = firesum1.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1, edgecolor='none', linewidth=0,  zorder=-1)

    b34 = firesum1.bar(bar_x, top_band + second_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b34[0].set_hatch(hatch_pattern[0])
    b34[1].set_hatch(hatch_pattern[1])

    # ax10.spines['left'].set_color('none')
    firesum1.spines['right'].set_color('none')
    firesum1.spines['top'].set_color('none')
    firesum1.spines['bottom'].set_color('none')
    firesum1.xaxis.set_ticks_position('none')
    # firesum1.yaxis.set_ticks_position('none')
    firesum1.tick_params(labelbottom=False, labeltop=False, labelright=False)

    firesum1.hlines(0, -0.5, 1.5, color='k', lw=1.5)

    firesum1.yaxis.set_tick_params(width=2)
    firesum1.spines['left'].set_linewidth(2.0)
    firesum1.spines['left'].set_position(('data', -0.5))
    firesum1.locator_params(axis='y', nbins=4)

    firesum1.ticklabel_format(axis='y')
    firesum1.set_ylabel('Area ($\mathregular{10^8}$ Hectares)', fontsize=fontsize1, labelpad=16)  # $\Delta$
    firesum1.set_ylim(-lim, lim)

    labels = list(str(elem) for elem in [0, 2, 0, 2])
    firesum1.set_yticklabels(labels)
    for tick in firesum1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)
    #ax10.text(np.mean(bar_x), lim*0.96 , '1950 - 1980',
    #         horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    firesum1.text(np.mean(bar_x), lim*1.02, '(i)',
                  horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    # decid tc stacked plots
    legend1 = fig.add_subplot(gs[0:6, 40:44])

    top_band = np.array([1,1])
    second_band = np.array([1,1])
    third_band = np.array([-1, -1])
    last_band = np.array([-1,-1])

    b1 = legend1.bar([0,1], third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b2 = legend1.bar([0,1], last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1, edgecolor='none', linewidth=0, zorder=-1)

    b12 = legend1.bar([0,1], third_band + last_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b12[0].set_hatch(hatch_pattern[0])
    b12[1].set_hatch(hatch_pattern[1])

    b3 = legend1.bar([0,1], second_band, color=['#EEC10B', '#5A2D09'],  width=0.65, alpha=0.4, edgecolor='none', linewidth=0, zorder=-1)
    b4 = legend1.bar([0,1], top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1, edgecolor='none', linewidth=0,  zorder=-1)

    b34 = legend1.bar([0,1], top_band + second_band, facecolor='none', width=0.65, alpha=1,
                  edgecolor='black', linewidth=1,  zorder=1)
    b34[0].set_hatch(hatch_pattern[0])
    b34[1].set_hatch(hatch_pattern[1])

    legend1.spines['left'].set_color('none')
    legend1.spines['right'].set_color('none')
    legend1.spines['top'].set_color('none')
    legend1.spines['bottom'].set_color('none')
    legend1.xaxis.set_ticks_position('none')
    legend1.yaxis.set_ticks_position('none')
    legend1.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    legend1.hlines(0, -0.5, 3.5, color='none', lw=1.5)

    # legend1.spines['left'].set_linewidth(1)
    # legend1.spines['left'].set_color('gray')
    # legend1.spines['left'].set_position(('data', -1))

    legend1.set_ylim(-2.5, 10.5)
    legend1.text(0, 2.25 , 'Deciduous fraction',
             horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=fontsize3)
    legend1.text(1, 2.25, 'Tree canopy cover',
             horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=fontsize3)

    legend1.text(1.6, 1.5, '$\Delta \geq$ {0:.1f}'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)
    legend1.text(1.6, 0.5, '0 < $\Delta$ < {0:.1f}'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)

    legend1.text(1.6, -0.5, '$-${0:.1f} < $\Delta$ < 0'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)
    legend1.text(1.6, -1.5, '$\Delta \leq$ $-${0:.1f}'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center',fontsize=fontsize3)


    bar_x = [0,0.95, 1.9]

    # spring forcing bar plots
    ax5 = fig.add_subplot(gs[7:13, 24:28])

    print('-------------------------------------------------------------------')
    ax5_keys = ['pos_spr_area_val_rec', 'pos_spr_area_u_val_rec',
                'pos_sum_area_val_rec', 'pos_sum_area_u_val_rec',
                'pos_fall_area_val_rec', 'pos_fall_area_u_val_rec',
                'neg_spr_area_val_rec', 'neg_spr_area_u_val_rec',
                'neg_sum_area_val_rec', 'neg_sum_area_u_val_rec',
                'neg_fall_area_val_rec', 'neg_fall_area_u_val_rec']

    for key in ax5_keys:
        print(key + ': {0:.3f} '.format(data_dict[key]))
    print('-------------------------------------------------------------------')

    spr_y = [data_dict['pos_spr_area_val_rec'], data_dict['pos_sum_area_val_rec'], data_dict['pos_fall_area_val_rec']]
    spr_y_sd = [data_dict['pos_spr_area_u_val_rec']/ 2., data_dict['pos_sum_area_u_val_rec']/ 2., data_dict['pos_fall_area_u_val_rec']/ 2.]

    spr_y_ = [data_dict['neg_spr_area_val_rec'], data_dict['neg_sum_area_val_rec'], data_dict['neg_fall_area_val_rec']]
    spr_y__sd = [data_dict['neg_spr_area_u_val_rec']/ 2., data_dict['neg_sum_area_u_val_rec']/ 2., data_dict['neg_fall_area_u_val_rec']/ 2.]

    '''
    bartop = ax5.bar(bar_x, spr_y, facecolor='#E8924A', width=0.8, alpha=0.85,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=spr_y_sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax5.bar(bar_x, spr_y_, facecolor='#0567AB', width=0.8, alpha=0.85,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=spr_y__sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)
    '''

    lwr_y_sd = [spr_y_sd[i] if (spr_y_sd[i] < spr_y[i]) else spr_y[i] * 0.9 for i in range(len(spr_y))]
    lwr_y__sd = [spr_y__sd[i] if (spr_y__sd[i] < abs(spr_y_[i])) else abs(spr_y_[i]) * 0.9 for i in range(len(spr_y_))]

    bartop = ax5.bar(bar_x, spr_y, facecolor='#E8924A', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[spr_y_sd, lwr_y_sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax5.bar(bar_x, spr_y_, facecolor='#0567AB', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[spr_y__sd, lwr_y__sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)

    ax5.spines['left'].set_color('none')
    ax5.spines['right'].set_color('none')
    ax5.spines['top'].set_color('none')
    ax5.spines['bottom'].set_color('none')
    ax5.xaxis.set_ticks_position('none')
    ax5.yaxis.set_ticks_position('none')
    ax5.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # ax5.hlines(0, -1.5, 3, color='None', lw=1.5)
    ax5.hlines(0, -0.75, 2.5, color='k', lw=1.5)

    lim = max(spr_y+spr_y_) * 1.05

    ax5.set_ylim(*ylim_new)

    ax5.text(np.mean(bar_x), ylim_new[1], '(iii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)
    ax5.text(np.mean(bar_x), ylim_new[0]*1.05, '1998 - 2018',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)

    bar_x = [-0.15,1, 2.15]

    # summer forcing bar plots
    ax6 = fig.add_subplot(gs[7:13, 20:24])

    print('-------------------------------------------------------------------')
    ax6_keys = ['pos_spr_area_val_intr', 'pos_spr_area_u_val_intr',
                'pos_sum_area_val_intr', 'pos_sum_area_u_val_intr',
                'pos_fall_area_val_intr', 'pos_fall_area_u_val_intr',
                'neg_spr_area_val_intr', 'neg_spr_area_u_val_intr',
                'neg_sum_area_val_intr', 'neg_sum_area_u_val_intr',
                'neg_fall_area_val_intr', 'neg_fall_area_u_val_intr']

    for key in ax6_keys:
        print(key + ': {0:.3f} '.format(data_dict[key]))
    print('-------------------------------------------------------------------')

    sum_y = [data_dict['pos_spr_area_val_intr'], data_dict['pos_sum_area_val_intr'], data_dict['pos_fall_area_val_intr']]
    sum_y_sd = [data_dict['pos_spr_area_u_val_intr']/ 2., data_dict['pos_sum_area_u_val_intr']/ 2., data_dict['pos_fall_area_u_val_intr']/ 2.]

    sum_y_ = [data_dict['neg_spr_area_val_intr'], data_dict['neg_sum_area_val_intr'], data_dict['neg_fall_area_val_intr']]
    sum_y__sd = [data_dict['neg_spr_area_u_val_intr']/ 2., data_dict['neg_sum_area_u_val_intr']/ 2., data_dict['neg_fall_area_u_val_intr']/ 2.]

    '''
    bartop = ax6.bar(bar_x, sum_y, facecolor='#E8924A', width=0.95, alpha=1,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=sum_y_sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax6.bar(bar_x, sum_y_, facecolor='#0567AB', width=0.95, alpha=1,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=sum_y__sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)
    '''

    lwr_y_sd = [sum_y_sd[i] if (sum_y_sd[i] < sum_y[i]) else sum_y[i] * 0.9 for i in range(len(sum_y))]
    lwr_y__sd = [sum_y__sd[i] if (sum_y__sd[i] < abs(sum_y_[i])) else abs(sum_y_[i]) * 0.9 for i in
                 range(len(sum_y_))]

    bartop = ax6.bar(bar_x, sum_y, facecolor='#E8924A', width=0.8, alpha=1,
                             error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[sum_y_sd, lwr_y_sd], edgecolor='black',
                             linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax6.bar(bar_x, sum_y_, facecolor='#0567AB', width=0.8, alpha=1,
                             error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[sum_y__sd, lwr_y__sd], edgecolor='black',
                             linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)

    ax6.spines['left'].set_color('none')
    ax6.spines['right'].set_color('none')
    ax6.spines['top'].set_color('none')
    ax6.spines['bottom'].set_color('none')
    ax6.xaxis.set_ticks_position('none')
    ax6.yaxis.set_ticks_position('none')
    ax6.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax6.hlines(0, -1, 3, color='k', lw=1.5)
    ax6.set_ylim(*ylim_new)
    ax6.text(np.mean(bar_x), ylim_new[1], '(ii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)
    ax6.text(np.mean(bar_x), ylim_new[0]*1.05, '1978 - 1998',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)

    # fall forcing gar plots
    ax7 = fig.add_subplot(gs[7:13, 16:20])

    print('-------------------------------------------------------------------')
    ax7_keys = ['pos_spr_area_val_old', 'pos_spr_area_u_val_old',
                'pos_sum_area_val_old', 'pos_sum_area_u_val_old',
                'pos_fall_area_val_old', 'pos_fall_area_u_val_old',
                'neg_spr_area_val_old', 'neg_spr_area_u_val_old',
                'neg_sum_area_val_old', 'neg_sum_area_u_val_old',
                'neg_fall_area_val_old', 'neg_fall_area_u_val_old']

    for key in ax7_keys:
        print(key + ': {0:.3f} '.format(data_dict[key]))
    print('-------------------------------------------------------------------')

    fall_y = [data_dict['pos_spr_area_val_old'], data_dict['pos_sum_area_val_old'], data_dict['pos_fall_area_val_old']]
    fall_y_sd = [data_dict['pos_spr_area_u_val_old'] / 2., data_dict['pos_sum_area_u_val_old'] / 2., data_dict['pos_fall_area_u_val_old'] / 2.]

    fall_y_ = [data_dict['neg_spr_area_val_old'], data_dict['neg_sum_area_val_old'], data_dict['neg_fall_area_val_old']]
    fall_y__sd = [data_dict['neg_spr_area_u_val_old'] / 2., data_dict['neg_sum_area_u_val_old'] / 2., data_dict['neg_fall_area_u_val_old'] / 2.]
    '''
    bartop = ax7.bar(bar_x, fall_y, facecolor='#E8924A', width=0.95, alpha=1,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=fall_y_sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax7.bar(bar_x, fall_y_, facecolor='#0567AB', width=0.95, alpha=1,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=fall_y__sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)
    '''
    lwr_y_sd = [fall_y_sd[i] if (fall_y_sd[i] < fall_y[i]) else fall_y[i] * 0.9 for i in range(len(fall_y))]
    lwr_y__sd = [fall_y__sd[i] if (fall_y__sd[i] < abs(fall_y_[i])) else abs(fall_y_[i]) * 0.9 for i in range(len(fall_y_))]

    bartop = ax7.bar(bar_x, fall_y, facecolor='#E8924A', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[fall_y_sd, lwr_y_sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax7.bar(bar_x, fall_y_, facecolor='#0567AB', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[fall_y__sd, lwr_y__sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)

    # ax7.spines['left'].set_color('none')
    ax7.spines['right'].set_color('none')
    ax7.spines['top'].set_color('none')
    ax7.spines['bottom'].set_color('none')
    ax7.xaxis.set_ticks_position('none')
    # ax7.yaxis.set_ticks_position('none')
    ax7.tick_params(labelbottom=False, labeltop=False, labelright=False)
    ax7.hlines(0, -1, 3, color='k', lw=1.5)
    ax7.set_ylim(*ylim_new)

    ax7.yaxis.set_tick_params(width=2)
    ax7.spines['left'].set_linewidth(2.0)
    ax7.spines['left'].set_position(('data', -1))
    ax7.locator_params(axis='y', nbins=4)
    ax7.ticklabel_format(axis='y')
    for tick in ax7.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

    ax7.set_ylabel('Radiative $\mathregular{forcing_{\ SW}}$ (W/$\mathregular{m^2}$)', fontsize=fontsize1)

    ax7.text(np.mean(bar_x), ylim_new[1], '(i)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    ax7.text(np.mean(bar_x), ylim_new[0]*1.05, '1950 - 1978',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)



    # fall forcing gar plots
    fireforc1 = fig.add_subplot(gs[7:13, 31:35])

    print('-------------------------------------------------------------------')
    fireforc1_keys = ['pos_spr_area_val_fc', 'pos_spr_area_u_val_fc',
                      'pos_sum_area_val_fc', 'pos_sum_area_u_val_fc',
                      'pos_fall_area_val_fc', 'pos_fall_area_u_val_fc',
                      'neg_spr_area_val_fc', 'neg_spr_area_u_val_fc',
                      'neg_sum_area_val_fc', 'neg_sum_area_u_val_fc',
                      'neg_fall_area_val_fc', 'neg_fall_area_u_val_fc']

    for key in fireforc1_keys:
        print(key + ': {0:.3f} '.format(data_dict[key]))
    print('-------------------------------------------------------------------')

    fall_y = [data_dict['pos_spr_area_val_fc'],
              data_dict['pos_sum_area_val_fc'],
              data_dict['pos_fall_area_val_fc']]

    fall_y_sd = [data_dict['pos_spr_area_u_val_fc'] / 2.,
                 data_dict['pos_sum_area_u_val_fc'] / 2.,
                 data_dict['pos_fall_area_u_val_fc'] / 2.]

    fall_y_ = [data_dict['neg_spr_area_val_fc'],
               data_dict['neg_sum_area_val_fc'],
               data_dict['neg_fall_area_val_fc']]

    fall_y__sd = [data_dict['neg_spr_area_u_val_fc'] / 2.,
                  data_dict['neg_sum_area_u_val_fc'] / 2.,
                  data_dict['neg_fall_area_u_val_fc'] / 2.]
    '''
    bartop = fireforc1.bar(bar_x, fall_y, facecolor='#E8924A', width=0.95, alpha=1,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=fall_y_sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = fireforc1.bar(bar_x, fall_y_, facecolor='#0567AB', width=0.95, alpha=1,
                     error_kw=dict(lw=1, capsize=4, capthick=1), yerr=fall_y__sd, edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)
    '''

    lwr_y_sd = [fall_y_sd[i] if (fall_y_sd[i] < fall_y[i]) else fall_y[i] * 0.9 for i in range(len(fall_y))]
    lwr_y__sd = [fall_y__sd[i] if (fall_y__sd[i] < abs(fall_y_[i])) else abs(fall_y_[i]) * 0.9 for i in range(len(fall_y_))]

    bartop = fireforc1.bar(bar_x, fall_y, facecolor='#E8924A', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[fall_y_sd, lwr_y_sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = fireforc1.bar(bar_x, fall_y_, facecolor='#0567AB', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[fall_y__sd, lwr_y__sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)

    # ax7.spines['left'].set_color('none')
    fireforc1.spines['right'].set_color('none')
    fireforc1.spines['top'].set_color('none')
    fireforc1.spines['bottom'].set_color('none')
    fireforc1.xaxis.set_ticks_position('none')
    # ax7.yaxis.set_ticks_position('none')
    fireforc1.tick_params(labelbottom=False, labeltop=False, labelright=False)
    fireforc1.hlines(0, -1, 3, color='k', lw=1.5)

    fireforc1.set_ylim(*ylim_new)

    fireforc1.yaxis.set_tick_params(width=2)
    fireforc1.spines['left'].set_linewidth(2.0)
    fireforc1.spines['left'].set_position(('data', -1))
    fireforc1.locator_params(axis='y', nbins=4)
    fireforc1.ticklabel_format(axis='y')

    fireforc1.set_ylabel('Radiative $\mathregular{forcing_{\ SW}}$ (W/$\mathregular{m^2}$)', fontsize=fontsize1)

    fireforc1.text(np.mean(bar_x), ylim_new[1], '(i)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    fireforc1.text(np.mean(bar_x), ylim_new[0] *1.05, 'Fire scars',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)
    for tick in fireforc1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

    bar_x = [0, 0.95, 1.9]

    # spring forcing bar plots
    overallforc = fig.add_subplot(gs[7:13, 35:39])

    print('-------------------------------------------------------------------')
    overallforc_keys = ['pos_spr_area_val_all', 'pos_spr_area_u_val_all',
                        'pos_sum_area_val_all', 'pos_sum_area_u_val_all',
                        'pos_fall_area_val_all', 'pos_fall_area_u_val_all',
                        'neg_spr_area_val_all', 'neg_spr_area_u_val_all',
                        'neg_sum_area_val_all', 'neg_sum_area_u_val_all',
                        'neg_fall_area_val_all', 'neg_fall_area_u_val_all']

    for key in overallforc_keys:
        print(key + ': {0:.3f} '.format(data_dict[key]))
    print('-------------------------------------------------------------------')

    spr_y = [data_dict['pos_spr_area_val_all'],
             data_dict['pos_sum_area_val_all'],
             data_dict['pos_fall_area_val_all']]

    spr_y_sd = [data_dict['pos_spr_area_u_val_all'] / 2.,
                data_dict['pos_sum_area_u_val_all'] / 2.,
                data_dict['pos_fall_area_u_val_all'] / 2.]

    spr_y_ = [data_dict['neg_spr_area_val_all'],
              data_dict['neg_sum_area_val_all'],
              data_dict['neg_fall_area_val_all']]

    spr_y__sd = [data_dict['neg_spr_area_u_val_all'] / 2.,
                 data_dict['neg_sum_area_u_val_all'] / 2.,
                 data_dict['neg_fall_area_u_val_all'] / 2.]

    lwr_y_sd = [spr_y_sd[i] if (spr_y_sd[i] < spr_y[i]) else spr_y[i] * 0.9 for i in range(len(spr_y))]
    lwr_y__sd = [spr_y__sd[i] if (spr_y__sd[i] < abs(spr_y_[i])) else abs(spr_y_[i]) * 0.9 for i in range(len(spr_y_))]

    bartop = overallforc.bar(bar_x, spr_y, facecolor='#E8924A', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[spr_y_sd, lwr_y_sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = overallforc.bar(bar_x, spr_y_, facecolor='#0567AB', width=0.8, alpha=1,
                     error_kw=dict(lw=1.5, capsize=4, capthick=1.5), yerr=[spr_y__sd, lwr_y__sd], edgecolor='black', linewidth=1)
    for bar, patt in zip(barbot, hatch_pattern):
        bar.set_hatch(patt)

    overallforc.spines['left'].set_color('none')
    overallforc.spines['right'].set_color('none')
    overallforc.spines['top'].set_color('none')
    overallforc.spines['bottom'].set_color('none')
    overallforc.xaxis.set_ticks_position('none')
    overallforc.yaxis.set_ticks_position('none')
    overallforc.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # ax5.hlines(0, -1.5, 3, color='None', lw=1.5)
    overallforc.hlines(0, -0.75, 2.5, color='k', lw=1.5)

    #lim = max(spr_y+spr_y_) * 1.05

    overallforc.set_ylim(*ylim_new)  #  *.6    *.625

    overallforc.text(np.mean(bar_x), ylim_new[1], '(ii)',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=fontsize3)

    overallforc.text(np.mean(bar_x),
                     ylim_new[0] * 1.05,
                     'Overall',
                     horizontalalignment='center',
                     verticalalignment='top',
                     fontsize=fontsize3)

    # decid tc stacked plots
    legend2 = fig.add_subplot(gs[7:13, 40:44])

    top_band = np.array([1,1,1])
    second_band = np.array([1,1, 1])

    b1 = legend2.bar(bar_x, second_band, color='#0567AB', width=0.8, alpha=1, edgecolor='black', linewidth=1)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])
    b1[2].set_hatch(hatch_pattern[2])

    b2 = legend2.bar(bar_x, top_band, bottom=second_band, color='#E8924A', width=0.8, alpha=1, edgecolor='black', linewidth=1)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])
    b2[2].set_hatch(hatch_pattern[2])

    legend2.spines['left'].set_color('none')
    legend2.spines['right'].set_color('none')
    legend2.spines['top'].set_color('none')
    legend2.spines['bottom'].set_color('none')
    legend2.xaxis.set_ticks_position('none')
    legend2.yaxis.set_ticks_position('none')
    legend2.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    legend2.hlines(0, -0.5, 3.5, color='none', lw=1.5)

    # legend2.spines['left'].set_linewidth(1)
    # legend2.spines['left'].set_color('gray')
    # legend2.spines['left'].set_position(('data', -1))

    legend2.set_ylim(-5, 8)
    legend2.text(0, 2.25 , 'Spring',
             horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=fontsize3)
    legend2.text(1, 2.25, 'Summer',
             horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=fontsize3)
    legend2.text(2, 2.25, 'Fall',
                 horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=fontsize3)

    legend2.text(2.6, 1.5, 'Warming',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)
    legend2.text(2.6, 0.5, 'Cooling',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)

    ax1.text(1950, alb_ylim[1], 'd',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2,fontweight='bold')
    ax2.text(1950, 0.35*1.02, 'a',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2, fontweight='bold')

    ax10.text(-0.3, lim0*1.02, 'b',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2,fontweight='bold')
    ax7.text(-0.6, ylim_new[1], 'e',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2, fontweight='bold')

    firesum1.text(-0.3, lim0*0.86, 'c',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2,fontweight='bold')

    fireforc1.text(-0.6, ylim_new[1], 'f',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2, fontweight='bold')

    fig.savefig(plotfile, dpi=300)

    plt.close()

