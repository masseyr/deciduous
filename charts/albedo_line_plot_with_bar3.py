from geosoup import Sublist, Handler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

moving_average = Sublist.moving_average


if __name__ == '__main__':

    infolder = 'd:/shared/Dropbox/projects/NAU/landsat_deciduous/data/forcings/'

    # plotfile1 = infolder + 'temp_plot1.png'
    plotfile2 = infolder + 'line_plot1_v4.png'
    # plotfile3 = infolder + 'line_plot2.png'

    fontsize1=30
    fontsize2=26
    fontsize3=26

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
    ylim2 = (-0.75, 0.55)

    maxy = ylim[1]-0.05
    minx = xlim[0]

    decid_ylim = (-0.35, 0.35)
    decid_miny = decid_ylim[1] - 0.05

    plt.rcParams.update({'font.size': 20, 'font.family': 'Calibri',
                         'hatch.color': 'k', 'hatch.linewidth': 0.5})
    plt.rcParams['axes.labelweight'] = 'regular'

    hatch_pattern = ('', '//', 'xx')

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

    decid_diff_0 = []
    decid_diff_25 = []

    tc_diff = list()
    tc_sd = list()
    tc_udiff = list()

    pos_decid = []
    neg_decid = []

    pos_tc = []
    neg_tc = []

    pos_spr_forc = []
    neg_spr_forc = []

    pos_sum_forc = []
    neg_sum_forc = []

    pos_fall_forc = []
    neg_fall_forc = []

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

        for elem in binned_dicts[j]:
            if elem['sum_forc'] > 0:
                pos_sum_forc.append(elem['sum_forc'])
            elif elem['sum_forc'] < 0:
                neg_sum_forc.append(elem['sum_forc'])

            if elem['spr_forc'] > 0:
                pos_spr_forc.append(elem['spr_forc'])
            elif elem['spr_forc'] < 0:
                neg_spr_forc.append(elem['spr_forc'])

            if elem['fall_forc'] > 0:
                pos_fall_forc.append(elem['fall_forc'])
            elif elem['fall_forc'] < 0:
                neg_fall_forc.append(elem['fall_forc'])

            if elem['decid_diff'] > 0:
                pos_decid.append(elem['decid_diff'])
            elif elem['decid_diff'] < 0:
                neg_decid.append(elem['decid_diff'])

            if elem['tc_diff'] > 0:
                pos_tc.append(elem['tc_diff'])
            elif elem['tc_diff'] < 0:
                neg_tc.append(elem['tc_diff'])

    pos_decid_mean = np.mean(pos_decid)
    pos_decid_sd = np.std(pos_decid)
    neg_decid_mean = np.mean(neg_decid)
    neg_decid_sd = np.std(neg_decid)

    pos_tc_mean = np.mean(pos_tc)
    pos_tc_sd = np.std(pos_tc)
    neg_tc_mean = np.mean(neg_tc)
    neg_tc_sd = np.std(neg_tc)

    pos_spr_forc_mean = np.mean(pos_spr_forc)
    pos_spr_forc_sd = np.std(pos_spr_forc)
    neg_spr_forc_mean = np.mean(neg_spr_forc)
    neg_spr_forc_sd = np.std(neg_spr_forc)

    pos_sum_forc_mean = np.mean(pos_sum_forc)
    pos_sum_forc_sd = np.std(pos_sum_forc)
    neg_sum_forc_mean = np.mean(neg_sum_forc)
    neg_sum_forc_sd = np.std(neg_sum_forc)

    pos_fall_forc_mean = np.mean(pos_fall_forc)
    pos_fall_forc_sd = np.std(pos_fall_forc)
    neg_fall_forc_mean = np.mean(neg_fall_forc)
    neg_fall_forc_sd = np.std(neg_fall_forc)

    fig = plt.figure(figsize=(28, 12))
    gs = gridspec.GridSpec(13, 44)  # rows, cols

    # forcing line plots
    ax1 = fig.add_subplot(gs[7:13, 0:13])
    # ax1.set_ylabel('Change in shortwave\n albedo radiative forcing (W/$\mathregular{m^2}$)')
    # ax1.set_ylabel('$\Delta$ Radiative $\mathregular{forcing_{\ SW\ albedo}}$ (W/$\mathregular{m^2}$)',fontsize=26)
    ax1.set_ylabel('$\Delta$ Rad. $\mathregular{forcing_{\ SWA}}$ (W/$\mathregular{m^2}$)', fontsize=fontsize1)
    ax1.set_xlabel('Fire occurrence date',fontsize=fontsize1)
    ax1.hlines(0, start_year-2, end_year+1, color='k', lw=1.5)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

    ax1.plot(range(start_year, end_year), moving_average(spr_mean, avg_window, True), color='#054C01', ls='-.',lw=1.5)
    ax1.fill_between(range(start_year, end_year),
                     moving_average(np.array(spr_mean) - np.array(spr_sd), avg_window, True),
                     moving_average(np.array(spr_mean) + np.array(spr_sd), avg_window, True),
                     color='#066001',
                     lw=0,
                     alpha=.2)

    ax1.plot(range(start_year, end_year), moving_average(fall_mean, avg_window, True), color='#02348D', ls='--',lw=1.5)
    ax1.fill_between(range(start_year, end_year),
                     moving_average(np.array(fall_mean) - np.array(fall_sd), avg_window, True),
                     moving_average(np.array(fall_mean) + np.array(fall_sd), avg_window, True),
                     color='#0350D8',
                     lw=0,
                     alpha=.3)

    ax1.plot(range(start_year, end_year), moving_average(sum_mean, avg_window, True), color='#912303', lw=1.5)
    ax1.fill_between(range(start_year, end_year),
                     moving_average(np.array(sum_mean) - np.array(sum_sd), avg_window, True),
                     moving_average(np.array(sum_mean) + np.array(sum_sd), avg_window, True),
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

    patch_hght = 0.075
    patch_wdth = 5
    pathc_dist = 0.02
    ax1.add_patch(patches.Rectangle((minx, maxy), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.2,
                                    edgecolor='none', facecolor='#066001'))
    ax1.plot([minx, minx+patch_wdth], [maxy - (patch_hght / 2), maxy - (patch_hght / 2)], color='#054C01', ls='-.', lw=1.5)
    ax1.text(minx+patch_wdth+1, maxy, 'Spring',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    ax1.add_patch(patches.Rectangle((1950, maxy-1*(patch_hght+pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.3,
                                    edgecolor='none', facecolor='#EA6A21'))
    ax1.plot([minx, minx+patch_wdth], [maxy-1*(patch_hght+pathc_dist) - (patch_hght / 2), maxy-1*(patch_hght+pathc_dist) - (patch_hght / 2)], color='#912303', ls='-', lw=1.5)
    ax1.text(minx+patch_wdth+1, maxy-1*(patch_hght+pathc_dist), 'Summer',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    ax1.add_patch(patches.Rectangle((minx, maxy-2*(patch_hght+pathc_dist)), patch_wdth, -patch_hght,
                                    fill=True, alpha=0.4,
                                    edgecolor='none', facecolor='#0350D8'))
    ax1.plot([minx, minx+patch_wdth], [maxy-2*(patch_hght+pathc_dist) - (patch_hght / 2), maxy-2*(patch_hght+pathc_dist) - (patch_hght / 2)], color='#02348D', ls='--', lw=1.5)
    ax1.text(minx+patch_wdth+1, maxy-2*(patch_hght+pathc_dist), 'Fall',
             horizontalalignment='left',
             verticalalignment='top', fontsize=fontsize3)

    # decid and tc line plots
    ax2 = fig.add_subplot(gs[0:6, 0:13], sharex=ax1)
    #ax2.set_ylabel('Fractional change')
    ax2.set_ylabel('$\Delta$ Forest composition',fontsize=fontsize1)

    ax2.hlines(0, start_year-2, end_year+1, color='k', lw=1.5)

    # ax2.vlines([2000, 2015], ylim[0], ylim[1], colors='k', linestyles='dotted', lw=2)

    ax2.plot(range(start_year, end_year), moving_average(tc_diff, avg_window, True), color='#035263', lw=1.5)
    ax2.fill_between(range(start_year, end_year),
                     moving_average(np.array(tc_diff) - np.array(tc_sd), avg_window, True),
                     moving_average(np.array(tc_diff) + np.array(tc_sd), avg_window, True),
                     color='#0C7287',
                     lw=0,
                     alpha=.4)

    ax2.plot(range(start_year, end_year), moving_average(decid_diff, avg_window, True), color='#A55606', ls='-.',lw=1.5)
    ax2.fill_between(range(start_year, end_year),
                     moving_average(np.array(decid_diff) - np.array(decid_sd), avg_window, True),
                     moving_average(np.array(decid_diff) + np.array(decid_sd), avg_window, True),
                     color='#CC6C0A',
                     lw=0,
                     alpha=.3)

    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

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

    rct = plt.Rectangle((1998, 1.0), 4, height=-2.2,
                        transform=ax2.get_xaxis_transform(), clip_on=False,
                        edgecolor="none", facecolor="#686968", alpha=.15)
    ax2.add_patch(rct)

    rct2 = plt.Rectangle((2013, 1.0), 5, height=-2.2,
                         transform=ax2.get_xaxis_transform(), clip_on=False,
                         edgecolor="none", facecolor="#686968", alpha=.15)
    ax2.add_patch(rct2)

    con = ConnectionPatch(xyA=[2000, 0.35], xyB=[2000, ylim[0]], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="gray", lw=2, ls='dotted')
    ax2.add_artist(con)

    con = ConnectionPatch(xyA=[2015.5, 0.35], xyB=[2015.5, ylim[0]], coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="gray", lw=2, ls='dotted')
    ax2.add_artist(con)

    '''
    conn_line = plt.Line2D([2000, 0.35], [2000, ylim[0]], linewidth=2,
                           linestyle='dotted',color='gray',
                           transform=ax2.get_xaxis_transform(), clip_on=False,)
    conn_line2 = plt.Line2D([2015.5, 0.35], [2015.5, ylim[0]], linewidth=2,
                            linestyle='dotted', color='gray',
                            transform=ax2.get_xaxis_transform(), clip_on=False,)

    ax2.add_patch(conn_line)
    ax2.add_patch(conn_line2)
    '''
    # overall boreal domain data

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

                'thresh': 0.15,

                'area tc': 4331521172124.239,
                'decid_area_val_lim_rec': 4122468761.4639707,
                'decid_area_val_0_lim_rec': 70000472282.71497,
                'decid_area_val__lim_0_rec': 162984629856.36752,
                'decid_area_val__lim_rec': 22229282194.64424,
                'decid_area_val_lim_intr': 47378711254.99669,
                'decid_area_val_0_lim_intr': 184739868290.34363,
                'decid_area_val__lim_0_intr': 62854035140.25784,
                'decid_area_val__lim_intr': 2766400898.82451,
                'decid_area_val_lim_old': 5072194560.688112,
                'decid_area_val_0_lim_old': 88441583763.99104,
                'decid_area_val__lim_0_old': 103752552771.91127,
                'decid_area_val__lim_old': 13392596194.014584,
                'decid_area_val_lim': 69929425102.2696,
                'decid_area_val_0_lim': 548986138851.53326,
                'decid_area_val__lim_0': 554240260603.5032,
                'decid_area_val__lim': 57535596426.002686,
                'decid_area_val_lim_all': 106140542389.43701,
                'decid_area_val_0_lim_all': 1835683756356.3582,
                'decid_area_val__lim_0_all': 2164479235229.2244,
                'decid_area_val__lim_all': 203060054891.99115,
                'tc_area_val_lim_rec': 4877789286.681128,
                'tc_area_val_0_lim_rec': 17403640657.795097,
                'tc_area_val__lim_0_rec': 62471118774.52524,
                'tc_area_val__lim_rec': 174584304376.18924,
                'tc_area_val_lim_intr': 153952123403.6244,
                'tc_area_val_0_lim_intr': 109002642709.71532,
                'tc_area_val__lim_0_intr': 26523689250.10282,
                'tc_area_val__lim_intr': 8260560220.980147,
                'tc_area_val_lim_old': 46122476063.84203,
                'tc_area_val_0_lim_old': 118554640130.8886,
                'tc_area_val__lim_0_old': 37557211654.317276,
                'tc_area_val__lim_old': 8424599441.557109,
                'tc_area_val_lim': 264176965409.6246,
                'tc_area_val_0_lim': 445227684662.91034,
                'tc_area_val__lim_0': 258626309112.77847,
                'tc_area_val__lim': 262660461797.99545,
                'tc_area_val_lim_all': 510793115562.9942,
                'tc_area_val_0_lim_all': 2005007411076.0833,
                'tc_area_val__lim_0_all': 1373365188107.2598,
                'tc_area_val__lim_all': 420197874120.6736,

                'pos_spr_area_val_rec': 0.16376330988022114,
                'neg_spr_area_val_rec': -0.6020891248819348,
                'pos_sum_area_val_rec': 0.02527307486635461,
                'neg_sum_area_val_rec': -0.055494152690450604,
                'pos_fall_area_val_rec': 0.07182862606742271,
                'neg_fall_area_val_rec': -0.19101729577935178,

                'pos_spr_area_val_intr': 0.33133482134013514,
                'neg_spr_area_val_intr': -0.04182692616271815,
                'pos_sum_area_val_intr': 0.04160393688501507,
                'neg_sum_area_val_intr': -0.03690114780314392,
                'pos_fall_area_val_intr': 0.14424066554449289,
                'neg_fall_area_val_intr': -0.08207962248975696,

                'pos_spr_area_val_old': 0.19946072892121647,
                'neg_spr_area_val_old': -0.04501460118822342,
                'pos_sum_area_val_old': 0.0327266617911413,
                'neg_sum_area_val_old': -0.03582027469626534,
                'pos_fall_area_val_old': 0.0737875067536922,
                'neg_fall_area_val_old': -0.06890670558582224,
                
                'pos_spr_area_val': 0.24175942475067114,
                'neg_spr_area_val': -0.3017947789356879,
                'pos_sum_area_val': 0.03444136227610692,
                'neg_sum_area_val': -0.04707386828600844,
                'pos_fall_area_val': 0.10172462203548405,
                'neg_fall_area_val': -0.1392368431361125,

                'pos_spr_area_val_all': 0.05472473837973414,
                'neg_spr_area_val_all': -0.07944514736446567,
                'pos_sum_area_val_all': 0.007169891392172423,
                'neg_sum_area_val_all': -0.008739161097757847,
                'pos_fall_area_val_all': 0.03405443692329397,
                'neg_fall_area_val_all': -0.01657878726267376,

                'pos_spr_area_val_rec_u': 0.16,
                'neg_spr_area_val_rec_u': 0.35,
                'pos_sum_area_val_rec_u': 0.025,
                'neg_sum_area_val_rec_u': 0.055,
                'pos_fall_area_val_rec_u': 0.06,
                'neg_fall_area_val_rec_u': 0.16,

                'pos_spr_area_val_intr_u': 0.32,
                'neg_spr_area_val_intr_u': 0.04,
                'pos_sum_area_val_intr_u': 0.04,
                'neg_sum_area_val_intr_u': 0.036,
                'pos_fall_area_val_intr_u': 0.14,
                'neg_fall_area_val_intr_u': 0.08,

                'pos_spr_area_val_old_u': 0.07,
                'neg_spr_area_val_old_u': 0.04,
                'pos_sum_area_val_old_u': 0.011,
                'neg_sum_area_val_old_u': 0.01,
                'pos_fall_area_val_old_u': 0.034,
                'neg_fall_area_val_old_u': 0.034,

                'pos_spr_area_val_fc_u': 0.18,
                'neg_spr_area_val_fc_u': 0.16,
                'pos_sum_area_val_fc_u': 0.024,
                'neg_sum_area_val_fc_u': 0.024,
                'pos_fall_area_val_fc_u': 0.072,
                'neg_fall_area_val_fc_u': 0.098
    }

    # decid tc stacked plots
    ax8 = fig.add_subplot(gs[0:6, 24:28])

    scale = 1e11

    bar_x = [0.1, 0.9]

    top_band = np.array([data_dict['decid_area_val_lim_rec'], data_dict['tc_area_val_lim_rec']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_rec'], data_dict['tc_area_val_0_lim_rec']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_rec'], -data_dict['tc_area_val__lim_0_rec']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_rec'], -data_dict['tc_area_val__lim_rec']]) / scale

    b1 = ax8.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.5)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])

    b2 = ax8.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65,alpha=1)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])

    b3 = ax8.bar(bar_x, second_band, color=['#EEC10B','#5A2D09'],  width=0.65, alpha=0.5)
    b3[0].set_hatch(hatch_pattern[0])
    b3[1].set_hatch(hatch_pattern[1])

    b4 = ax8.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1)
    b4[0].set_hatch(hatch_pattern[0])
    b4[1].set_hatch(hatch_pattern[1])

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

    top_band = np.array([data_dict['decid_area_val_lim_intr'], data_dict['tc_area_val_lim_intr']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_intr'], data_dict['tc_area_val_0_lim_intr']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_intr'], -data_dict['tc_area_val__lim_0_intr']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_intr'], -data_dict['tc_area_val__lim_intr']]) / scale

    b1 = ax9.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.5)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])

    b2 = ax9.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])

    b3 = ax9.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=0.5)
    b3[0].set_hatch(hatch_pattern[0])
    b3[1].set_hatch(hatch_pattern[1])

    b4 = ax9.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1)
    b4[0].set_hatch(hatch_pattern[0])
    b4[1].set_hatch(hatch_pattern[1])

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

    top_band = np.array([data_dict['decid_area_val_lim_old'], data_dict['tc_area_val_lim_old']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_old'], data_dict['tc_area_val_0_lim_old']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_old'], -data_dict['tc_area_val__lim_0_old']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_old'], -data_dict['tc_area_val__lim_old']]) / scale

    b1 = ax10.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.5)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])

    b2 = ax10.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])

    b3 = ax10.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=0.5)
    b3[0].set_hatch(hatch_pattern[0])
    b3[1].set_hatch(hatch_pattern[1])

    b4 = ax10.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1)
    b4[0].set_hatch(hatch_pattern[0])
    b4[1].set_hatch(hatch_pattern[1])

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
    for tick in ax10.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)
    ax10.text(np.mean(bar_x), lim*1.02 , '(i)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)


    # decid tc stacked plots: fire summary
    overall_area = fig.add_subplot(gs[0:6, 35:39])

    scale = 1e12

    bar_x = [0.1, 0.9]

    top_band = np.array([data_dict['decid_area_val_lim_all'], data_dict['tc_area_val_lim_all']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim_all'], data_dict['tc_area_val_0_lim_all']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0_all'], -data_dict['tc_area_val__lim_0_all']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim_all'], -data_dict['tc_area_val__lim_all']]) / scale

    b1 = overall_area.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.5)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])

    b2 = overall_area.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65,alpha=1)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])

    b3 = overall_area.bar(bar_x, second_band, color=['#EEC10B','#5A2D09'],  width=0.65, alpha=0.5)
    b3[0].set_hatch(hatch_pattern[0])
    b3[1].set_hatch(hatch_pattern[1])

    b4 = overall_area.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1)
    b4[0].set_hatch(hatch_pattern[0])
    b4[1].set_hatch(hatch_pattern[1])

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

    top_band = np.array([data_dict['decid_area_val_lim'], data_dict['tc_area_val_lim']]) / scale
    second_band = np.array([data_dict['decid_area_val_0_lim'], data_dict['tc_area_val_0_lim']]) / scale
    third_band = np.array([-data_dict['decid_area_val__lim_0'], -data_dict['tc_area_val__lim_0']]) / scale
    last_band = np.array([-data_dict['decid_area_val__lim'], -data_dict['tc_area_val__lim']]) / scale

    b1 = firesum1.bar(bar_x, third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=0.5)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])

    b2 = firesum1.bar(bar_x, last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.65, alpha=1)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])

    b3 = firesum1.bar(bar_x, second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=0.5)
    b3[0].set_hatch(hatch_pattern[0])
    b3[1].set_hatch(hatch_pattern[1])

    b4 = firesum1.bar(bar_x, top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.65, alpha=1)
    b4[0].set_hatch(hatch_pattern[0])
    b4[1].set_hatch(hatch_pattern[1])

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
    for tick in firesum1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)
    #ax10.text(np.mean(bar_x), lim*0.96 , '1950 - 1980',
    #         horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    firesum1.text(np.mean(bar_x), lim*1.02 , '(i)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)



    # decid tc stacked plots
    legend1 = fig.add_subplot(gs[0:6, 40:44])

    top_band = np.array([1,1])
    second_band = np.array([1,1])
    third_band = np.array([-1, -1])
    last_band = np.array([-1,-1])

    b1 = legend1.bar([0,1], third_band, color=['#046E03', '#7C388D'], width=0.8, alpha=0.5)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])

    b2 = legend1.bar([0,1], last_band, bottom=third_band, color=['#046E03', '#7C388D'], width=0.8, alpha=0.9)
    b2[0].set_hatch(hatch_pattern[0])
    b2[1].set_hatch(hatch_pattern[1])

    b3 = legend1.bar([0,1], second_band, color=['#EEC10B', '#5A2D09'], width=0.8, alpha=0.5)
    b3[0].set_hatch(hatch_pattern[0])
    b3[1].set_hatch(hatch_pattern[1])

    b4 = legend1.bar([0,1], top_band, bottom=second_band, color=['#EEC10B', '#5A2D09'], width=0.8, alpha=0.9)
    b4[0].set_hatch(hatch_pattern[0])
    b4[1].set_hatch(hatch_pattern[1])

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

    legend1.text(1.6, 1.5, '$\Delta \geq$ {0:.2f}'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)
    legend1.text(1.6, 0.5, '0 < $\Delta$ < {0:.2f}'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)

    legend1.text(1.6, -0.5, '$-${0:.2f} < $\Delta$ < 0'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize3)
    legend1.text(1.6, -1.5, '$\Delta \leq$ $-${0:.2f}'.format(data_dict['thresh']),
                 horizontalalignment='left', verticalalignment='center',fontsize=fontsize3)


    '''
    # decid bar plots
    ax3 = fig.add_subplot(gs[0:3, 10:12])


    # ax3.hlines(0, -0.5, 2, color='k', lw=1.5)

    decid_y = [data_dict['pos_decid_i'], data_dict['pos_decid_o'], data_dict['pos_decid']]
    decid_y_sd = [data_dict['pos_decid_u_i']/4., data_dict['pos_decid_u_o']/4., data_dict['pos_decid_u']/4.]

    decid_y_ = [data_dict['neg_decid_i'], data_dict['neg_decid_o'],data_dict['neg_decid']]
    decid_y__sd = [data_dict['neg_decid_u_i']/4., data_dict['neg_decid_u_o']/4., data_dict['neg_decid_u']/4.]

    ax3.bar(bar_x, decid_y, facecolor='#EEC10B', edgecolor='white', width=0.7, alpha=0.75, yerr=decid_y_sd, capsize=3)
    ax3.bar(bar_x, decid_y_, facecolor='#046E03', edgecolor='white', width=0.7, alpha=0.75, yerr=decid_y__sd, capsize=3)

    # ax2.set_ylim(-0.35,0.35)
    # ax3.yaxis.set_tick_params(width=2)
    # ax3.spines['left'].set_linewidth(2.0)
    # ax3.spines['left'].set_position(('data', -0.5))
    # ax3.locator_params(axis='y', nbins=4)

    ax3.spines['left'].set_color('none')
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.spines['bottom'].set_color('none')
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')
    ax3.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax3.hlines(0, -1, 3, color='k', lw=1.5)
    ax3.set_ylim(*decid_ylim)
    
    
    for x, y in zip(bar_x, decid_y):
        ax3.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(bar_x, decid_y_):
        ax3.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    '''
    bar_x = [0,0.95, 1.9]

    # spring forcing bar plots
    ax5 = fig.add_subplot(gs[7:13, 24:28])

    spr_y = [data_dict['pos_spr_area_val_rec'], data_dict['pos_sum_area_val_rec'], data_dict['pos_fall_area_val_rec']]
    spr_y_sd = [data_dict['pos_spr_area_val_rec_u'] / 2., data_dict['pos_sum_area_val_rec_u'] / 2., data_dict['pos_fall_area_val_rec_u'] / 2.]

    spr_y_ = [data_dict['neg_spr_area_val_rec'], data_dict['neg_sum_area_val_rec'], data_dict['neg_fall_area_val_rec']]
    spr_y__sd = [data_dict['neg_spr_area_val_rec_u'] / 2., data_dict['neg_sum_area_val_rec_u'] / 2., data_dict['neg_fall_area_val_rec_u'] / 2.]

    bartop = ax5.bar(bar_x, spr_y, facecolor='#E8924A', width=0.8, alpha=0.75,  capsize=3, yerr=spr_y_sd)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax5.bar(bar_x, spr_y_, facecolor='#0567AB', width=0.8, alpha=0.75, capsize=3, yerr=spr_y__sd)
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

    ylim_new = [ylim2[0]*1.1, ylim2[1]*1.1]

    ax5.set_ylim(*ylim_new)

    ax5.text(np.mean(bar_x), ylim_new[1], '(iii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)
    ax5.text(np.mean(bar_x), ylim_new[0]*1.05, '1998 - 2018',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)

    bar_x = [-0.15,1, 2.15]

    # summer forcing bar plots
    ax6 = fig.add_subplot(gs[7:13, 20:24])

    sum_y = [data_dict['pos_spr_area_val_intr'], data_dict['pos_sum_area_val_intr'], data_dict['pos_fall_area_val_intr']]
    sum_y_sd = [data_dict['pos_spr_area_val_intr_u'] / 2., data_dict['pos_sum_area_val_intr_u'] / 2., data_dict['pos_fall_area_val_intr_u'] / 2.]

    sum_y_ = [data_dict['neg_spr_area_val_intr'], data_dict['neg_sum_area_val_intr'], data_dict['neg_fall_area_val_intr']]
    sum_y__sd = [data_dict['neg_spr_area_val_intr_u'] / 2., data_dict['neg_sum_area_val_intr_u'] / 2., data_dict['neg_fall_area_val_intr_u'] / 2.]

    bartop = ax6.bar(bar_x, sum_y, facecolor='#E8924A', width=0.95, alpha=0.75,  capsize=3, yerr=sum_y_sd)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax6.bar(bar_x, sum_y_, facecolor='#0567AB', width=0.95, alpha=0.75, capsize=3, yerr=sum_y__sd)
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

    fall_y = [data_dict['pos_spr_area_val_old'], data_dict['pos_sum_area_val_old'], data_dict['pos_fall_area_val_old']]
    fall_y_sd = [data_dict['pos_spr_area_val_old_u'] / 2., data_dict['pos_sum_area_val_old_u'] / 2., data_dict['pos_fall_area_val_old_u'] / 2.]

    fall_y_ = [data_dict['neg_spr_area_val_old'], data_dict['neg_sum_area_val_old'], data_dict['neg_fall_area_val_old']]
    fall_y__sd = [data_dict['neg_spr_area_val_old_u'] / 2., data_dict['neg_sum_area_val_old_u'] / 2., data_dict['neg_fall_area_val_old_u'] / 2.]

    bartop = ax7.bar(bar_x, fall_y, facecolor='#E8924A', width=0.95, alpha=0.75, capsize=3, yerr=fall_y_sd)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = ax7.bar(bar_x, fall_y_, facecolor='#0567AB', width=0.95, alpha=0.75, capsize=3, yerr=fall_y__sd)
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

    ax7.set_ylabel('Mean $\Delta$ rad. $\mathregular{forcing_{\ SWA}}$ (W/$\mathregular{m^2}$)', fontsize=fontsize1)

    ax7.text(np.mean(bar_x), ylim_new[1], '(i)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    ax7.text(np.mean(bar_x), ylim_new[0]*1.05, '1950 - 1978',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)



    # fall forcing gar plots
    fireforc1 = fig.add_subplot(gs[7:13, 31:35])

    fall_y = [data_dict['pos_spr_area_val'], data_dict['pos_sum_area_val'], data_dict['pos_fall_area_val']]
    fall_y_sd = [data_dict['pos_spr_area_val_fc_u'] / 2., data_dict['pos_sum_area_val_fc_u'] / 2., data_dict['pos_fall_area_val_fc_u'] / 2.]

    fall_y_ = [data_dict['neg_spr_area_val'], data_dict['neg_sum_area_val'], data_dict['neg_fall_area_val']]
    fall_y__sd = [data_dict['neg_spr_area_val_fc_u'] / 2., data_dict['neg_sum_area_val_fc_u'] / 2., data_dict['neg_fall_area_val_fc_u'] / 2.]

    bartop = fireforc1.bar(bar_x, fall_y, facecolor='#E8924A', width=0.95, alpha=0.75, capsize=3, yerr=fall_y_sd)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = fireforc1.bar(bar_x, fall_y_, facecolor='#0567AB', width=0.95, alpha=0.75, capsize=3, yerr=fall_y__sd)
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

    fireforc1.set_ylim(ylim[0] * .6, ylim[1] * .625)

    fireforc1.yaxis.set_tick_params(width=2)
    fireforc1.spines['left'].set_linewidth(2.0)
    fireforc1.spines['left'].set_position(('data', -1))
    fireforc1.locator_params(axis='y', nbins=4)
    fireforc1.ticklabel_format(axis='y')

    fireforc1.set_ylabel('Mean $\Delta$ rad. $\mathregular{forcing_{\ SWA}}$ (W/$\mathregular{m^2}$)', fontsize=fontsize1)

    fireforc1.text(np.mean(bar_x), ylim[1] * .625, '(i)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)

    fireforc1.text(np.mean(bar_x), -ylim[1] * .6 *1.25, 'Fire scars',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)
    for tick in fireforc1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize2)

    bar_x =[0,0.95, 1.9]

    # spring forcing bar plots
    overallforc = fig.add_subplot(gs[7:13, 35:39])

    spr_y = [data_dict['pos_spr_area_val_all'], data_dict['pos_sum_area_val_all'], data_dict['pos_fall_area_val_all']]
    spr_y_sd = [data_dict['pos_spr_area_val_old_u'] / 2., data_dict['pos_sum_area_val_old_u'] / 2., data_dict['pos_fall_area_val_old_u'] / 2.]

    spr_y_ = [data_dict['neg_spr_area_val_all'], data_dict['neg_sum_area_val_all'], data_dict['neg_fall_area_val_all']]
    spr_y__sd = [data_dict['neg_spr_area_val_old_u'] / 2., data_dict['neg_sum_area_val_old_u'] / 2., data_dict['neg_fall_area_val_old_u'] / 2.]

    bartop = overallforc.bar(bar_x, spr_y, facecolor='#E8924A', width=0.8, alpha=0.75,  capsize=3, yerr=spr_y_sd)
    for bar, patt in zip(bartop, hatch_pattern):
        bar.set_hatch(patt)
    barbot = overallforc.bar(bar_x, spr_y_, facecolor='#0567AB', width=0.8, alpha=0.75, capsize=3, yerr=spr_y__sd)
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

    overallforc.set_ylim(ylim[0]*.6, ylim[1]*.625)

    overallforc.text(np.mean(bar_x), ylim[1]*.625, '(ii)',
             horizontalalignment='center', verticalalignment='center', fontsize=fontsize3)
    overallforc.text(np.mean(bar_x), -ylim[1]*0.6*1.25, 'Overall',
             horizontalalignment='center', verticalalignment='top', fontsize=fontsize3)



    # decid tc stacked plots
    legend2 = fig.add_subplot(gs[7:13, 40:44])

    top_band = np.array([1,1,1])
    second_band = np.array([1,1, 1])

    b1 = legend2.bar(bar_x, second_band, color='#0567AB', width=0.8,alpha=0.8)
    b1[0].set_hatch(hatch_pattern[0])
    b1[1].set_hatch(hatch_pattern[1])
    b1[2].set_hatch(hatch_pattern[2])

    b2 = legend2.bar(bar_x, top_band, bottom=second_band, color='#E8924A', width=0.8, alpha=0.8)
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

    ax1.text(1950, ylim[1], 'd',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2,fontweight='bold')
    ax2.text(1950, 0.35*1.02, 'a',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2, fontweight='bold')

    ax10.text(-0.3, lim0*1.02, 'b',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2,fontweight='bold')
    ax7.text(-0.6, ylim_new[1], 'e',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2, fontweight='bold')

    firesum1.text(-0.3, lim0*1.02, 'c',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2,fontweight='bold')
    fireforc1.text(-0.6, ylim[1] * .625, 'f',
                 horizontalalignment='left', verticalalignment='center', fontsize=fontsize1+2, fontweight='bold')

    fig.savefig(plotfile2, dpi=300)

    plt.close()

