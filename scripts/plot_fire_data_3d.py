from modules.plots import Plot
from modules import Handler, Opt, Sublist, _Regressor
from sys import argv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import axes3d, Axes3D

"""
this script is to read the reformatted decid and tc files and 
plot them on regression plots
I might also add plotting all the best fits to the same plot here
"""
plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})

plt.rcParams['axes.labelweight'] = 'bold'
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')

def apply_median(arr):

    loc = np.where(arr > 0.0)[0]

    if loc.shape[0] > 0:
        return np.median(arr[loc])
    else:
        return 0.0

def apply_count(arr):

    loc = np.where(arr > 0.0)[0]

    return loc.shape[0]


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

    xvar = 'tree_cover'
    yvar = 'time_since_fire'
    zvar = 'decid_frac'

    # ylim = (0, 80)  # time since fire
    ylim = (1950, 2018)
    xlim = (0, 1)  # tree cover
    zlim = (0, 1)  # deciduous fraction

    zcount_lim = 100000

    x_ = np.array(list(range(0, 101)))/100.0
    # y_ = np.array(Sublist.frange(0, 70, 1))
    y_ = np.array(Sublist.frange(1950, 2018, 1))
    lenx = x_.shape[0]
    leny = y_.shape[0]

    print(x_)
    print(y_)

    years = [1992, 2000, 2005, 2010, 2015]

    version = 10

    filelist = list(in_dir + 'year_{}_{}_{}_fire.csv'.format(str(year_edge[0])[2:],
                                                            str(year_edge[1])[2:],
                                                            fire_type) for year_edge in year_edges
                    for fire_type in fire_types)

    zvals_list = list(np.zeros((zcount_lim, leny, lenx)) for _ in years)
    zcount_list = list(np.zeros((leny, lenx), dtype='int16') for _ in years)

    for filename in filelist:

        if 'single' in filename:
            print(filename)
            val_dicts = Handler(filename).read_from_csv(return_dicts=True)

            print(len(val_dicts))

            for ii, decid_year in enumerate(years):
                max_time = 0

                decid_band = 'decid{}'.format(str(decid_year))
                tc_band = 'tc{}'.format(str(decid_year))

                for val_dict in val_dicts:

                    if (type(val_dict[decid_band]).__name__ in ('int', 'float')) and \
                            (type(val_dict[tc_band]).__name__ in ('int', 'float')):

                        if float(val_dict['longitude']) > -142.0:

                            if (float(val_dict[tc_band])/100.0 >= 0.0) and (decid_year >= int(val_dict['burnyear_1'])):

                                try:

                                    xloc = np.where(x_ == float(val_dict[tc_band])/100.0)[0][0]
                                    # yloc = np.where(y_ == (decid_year-int(val_dict['burnyear_1'])))[0][0]
                                    yloc = np.where(y_ == int(val_dict['burnyear_1']))[0][0]

                                    if max_time < (decid_year-int(val_dict['burnyear_1'])):
                                        max_time = (decid_year-int(val_dict['burnyear_1']))

                                    if zcount_list[ii][yloc, xloc] < zcount_lim:

                                        zvals_list[ii][zcount_list[ii][yloc, xloc], yloc, xloc] = \
                                            float(val_dict[decid_band])/100.0

                                        zcount_list[ii][yloc, xloc] += 1

                                except Exception as e:
                                    print(e)
                                    print(val_dict)
                                    print(decid_year-int(val_dict['burnyear_1']))
                                    print(float(val_dict[tc_band])/100.0)

    for i, z_vals in enumerate(zvals_list):

        plotfile = in_dir + 'fire_plot_3d_{}_v{}.png'.format(str(years[i]),
                                                             str(version))

        z = np.apply_along_axis(apply_count, 0, z_vals)

        # z = np.sum(z_vals, 0)/z_count

        # z[np.where(np.isnan(z))[0]] = 0.0

        x, y = np.meshgrid(x_, y_)

        print(x.shape)
        print(y.shape)
        print(z.shape)

        print(np.max(z))

        plt.pcolor(x, y, z, cmap=cm.coolwarm, vmin=0, vmax=10000)
        plt.gca().invert_yaxis()
        #  plt.ylim(ymax=max_time)
        plt.colorbar()

        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(zlim)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # ax.set_xlabel('Variable Split', fontproperties=font, fontsize='large', fontweight='bold')
        # ax.set_ylabel('Trees', fontproperties=font, fontsize='large', fontweight='bold')
        # ax.set_zlabel('R-squared', fontproperties=font, fontsize='large', fontweight='bold', rotation=180)

        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontproperties(font)

        zlabels = ax.zaxis.get_majorticklabels()
        for label in zlabels:
            label.set_fontproperties(font)

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(elev=45, azim=235)
        '''
        plt.savefig(plotfile)
        plt.close()



