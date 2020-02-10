from modules import *
import datetime
import random
import numpy as np
from scipy.interpolate import interpn
from scipy.signal import savgol_filter
import scipy.stats as stats
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import copy
import math


def percentile(inp_list, pctl):
    """
    Find the percentile of a list of values.
    :param inp_list: input list of int or float
    :param pctl: percentile (numbers between 0-100)
    """
    if not inp_list:
        return None

    sorted_arr = sorted(inp_list)
    k = (len(sorted_arr)-1) * (float(pctl)/100.0)
    f = int(math.ceil(k))

    return sorted_arr[f]


def ndvi_pctl_composite(site_dict,
                        pctl=75,
                        jday_start=0,
                        jday_end=365):
    """
    Create NDVI based percentile composite
    :param site_dict: Site data dictionary
    :param pctl: Percentile for compositing (0-100)
    :param jday_start: Julian day start
    :param jday_end:  Julian day end
    :return: site dictionary
    """

    ndvi_list = list()
    for index, data_dict in site_dict.items():
        ndvi = int(normalized_difference(data_dict['bands']['nir'],
                                         data_dict['bands']['red'],
                                         canopy_adj=0.0) * 10000.0)

        if jday_start <= data_dict['img_jday'] <= jday_end:

            ndvi_list.append((data_dict['img_jday'],
                              data_dict['img_year'],
                              index,
                              ndvi))

    if len(ndvi_list) > 0:
        pctl_th_ndvi = percentile(list(elem[3] for elem in ndvi_list), pctl)
        index = list(elem[2] for elem in ndvi_list if elem[3] == pctl_th_ndvi)[0]
        return {index: site_dict[index]}

    else:
        return {}


def normalized_difference(b1,
                          b2,
                          canopy_adj=0.5,
                          adj_index=1.0,
                          additive=0.0):

    """
    Calculate normalized difference
    :param b1: band 1
    :param b2: band 2
    :param canopy_adj: Canopy adjustment parameter
    :param adj_index: adjustment index - scaling parameter
    :param additive: additive index - scaling parameter
    :return: float
    """

    if float(b1)+float(b2)+canopy_adj != 0.0:
        return ((1 + canopy_adj**adj_index)*(float(b1)-float(b2)))/(float(b1)+float(b2)+canopy_adj)+additive
    else:
        return 0.0


def correct_landsat_sr(bands_dict,
                       sensor):
    """
    Scale Landsat 8 to Landsat 7 surface reflectance
    :param bands_dict: dictionary of input bands with band names: B1, B2, B3, B4, B5, B6, B7
    :param sensor: Landsat sensor acronym: Landsat 5 - LT05; Landsat 7 - LE07; Landsat 8 - LC08
    :return: dictionary with bands as: 'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
    """

    out_dict = dict()

    if sensor == 'LT05' or sensor == 'LE07':

        out_dict['blue'] = float(bands_dict['B1'])/10000.0
        out_dict['green'] = float(bands_dict['B2'])/10000.0
        out_dict['red'] = float(bands_dict['B3'])/10000.0
        out_dict['nir'] = float(bands_dict['B4'])/10000.0
        out_dict['swir1'] = float(bands_dict['B5'])/10000.0
        out_dict['swir2'] = float(bands_dict['B7'])/10000.0

    elif sensor == 'LC08':

        out_dict['blue'] = 0.8850 * (float(bands_dict['B2'])/10000.0) + 0.0183
        out_dict['green'] = 0.9317 * (float(bands_dict['B3'])/10000.0) + 0.0123
        out_dict['red'] = 0.9372 * (float(bands_dict['B4'])/10000.0) + 0.0123
        out_dict['nir'] = 0.8339 * (float(bands_dict['B5'])/10000.0) + 0.0448
        out_dict['swir1'] = 0.8639 * (float(bands_dict['B6'])/10000.0) + 0.0306
        out_dict['swir2'] = 0.9165 * (float(bands_dict['B7'])/10000.0) + 0.0116

    else:
        raise ValueError('Invalid sensor specified')

    return out_dict


def extract_date(string):
    # LT05_L1TP_035020_19960605_20170104_01_T1
    list_str = string.split('_')
    return {
        'sensor': list_str[0],
        'level': list_str[1],
        'pathrow': (int(list_str[2][0:3]), int(list_str[2][3:6])),
        'date': datetime.datetime.strptime(list_str[3], '%Y%m%d')
    }


def clear_value(x, length=16):
    bit_str = ''.join(reversed(np.binary_repr(int(x), length)))
    fill = bit_str[0] == '1'
    clear = bit_str[1] == '1'
    cloud_shadow = bit_str[3] == '1'
    cloud = bit_str[5] == '1'
    cloud_conf = bit_str[6:8] == '11' or bit_str[6:8] == '01'
    cirrus_conf = bit_str[8:10] == '11' or bit_str[8:10] == '01'
    terr_occ = bit_str[10] == '1'

    if fill or cloud or cloud_shadow or cloud_conf or cirrus_conf or terr_occ:
        return 0
    else:
        return 1


def saturated_bands(x, length=16):
    bit_str = ''.join(reversed(np.binary_repr(x, length)))
    fill = bit_str[0] == '1'

    if fill:
        return 0
    else:
        sat_bands = list()
        for i, elem in enumerate(bit_str[1:-1]):
            if elem == '1':
                sat_bands.append(i + 1)
        return sat_bands


def read_gee_extract_data(filename):

    lines = Handler(filename).read_from_csv(return_dicts=True)

    # sites = list(set(list(str(line['site']) for line in lines)))
    # print(len(sites))

    site_dict = dict()
    line_counter = 0
    for j, line in enumerate(lines):

        include = True
        for key, val in line.items():
            if type(val).__name__ == 'str':
                if val == 'None':
                    include = False

        if line['radsat_qa'] > 0.0 or line['GEOMETRIC_RMSE_MODEL'] > 15.0 or clear_value(line['pixel_qa']) == 0:
            include = False

        if include:
            line_counter += 1
            site_year = str(line['site']) + '_' + str(line['year'])

            if site_year not in site_dict:
                geom_wkt = Vector.wkt_from_coords((line['longitude'], line['latitude']))
                site_dict[site_year] = {'geom': geom_wkt,
                                        'decid_frac': line['decid_frac'],
                                        'data': dict(),
                                        'site_year': line['year']}

            temp_dict = dict()

            sensor_dict = extract_date(line['LANDSAT_ID'])

            temp_dict['img_jday'] = sensor_dict['date'].timetuple().tm_yday
            temp_dict['img_year'] = sensor_dict['date'].timetuple().tm_year
            temp_dict['sensor'] = sensor_dict['sensor']

            bands = list('B' + str(ii + 1) for ii in range(7))

            band_dict = dict()
            for band in bands:
                band_dict[band] = line[band]

            temp_dict['bands'] = correct_landsat_sr(band_dict,
                                                    sensor_dict['sensor'])

            site_dict[site_year]['data'].update({'{}_{}'.format(str(temp_dict['img_jday']),
                                                                str(temp_dict['img_year'])): temp_dict})

    # print(line_counter)

    return site_dict


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------

    version = 14

    canopy_adj = 0.5
    additive = 0.25
    adj_idx = 1.0

    # number of bins to divide the 0 - 365 date range into
    year_div = 52

    composite_pctl = 75

    min_samp = 200

    deg = 3
    window = 9

    # upper and lower percentile
    upctl = 95
    lpctl = 5

    decid_frac_limit_decid = 0.90
    decid_frac_limit_everg = 0.05

    cutoff = 55  # percentile per bin at which to cutoff the samples

    plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
    plt.rcParams['axes.labelweight'] = 'bold'
    # print(plt.rcParams.keys())

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')

    # ----------------------------------------------------------------------------------------------------

    folder = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/gee_extract/"

    datafile = folder + "gee_samp_extract_v2019_01_29T08_06_32.csv"

    # ----------------------------------------------------------------------------------------------------

    gee_data_dict = read_gee_extract_data(datafile)

    for elem in gee_data_dict.items()[0:10]:
        print(elem)

    # ----------------------------------------------------------------------------------------------------

    ndvi_decid_list = list()
    ndvi_everg_list = list()

    for site_year, site_year_data in gee_data_dict.items():

        if site_year_data['decid_frac'] >= decid_frac_limit_decid:
            for _, data_dict in site_year_data['data'].items():
                ndvi = normalized_difference(data_dict['bands']['nir'],
                                             data_dict['bands']['red'],
                                             canopy_adj=canopy_adj,
                                             adj_index=adj_idx,
                                             additive=additive)
                jday = data_dict['img_jday']

                if -0.1 < ndvi < 1.0:
                    ndvi_decid_list.append((jday, ndvi))

        elif site_year_data['decid_frac'] <= decid_frac_limit_everg:
            for _, data_dict in site_year_data['data'].items():
                ndvi = normalized_difference(data_dict['bands']['nir'],
                                             data_dict['bands']['red'],
                                             canopy_adj=canopy_adj,
                                             adj_index=adj_idx,
                                             additive=additive)
                jday = data_dict['img_jday']

                if -0.1 < ndvi < 1.0:
                    ndvi_everg_list.append((jday, ndvi))

    print('Number of Deciduous samples: {}'.format(str(len(ndvi_decid_list))))
    print('Number of Evergreen samples: {}'.format(str(len(ndvi_everg_list))))

    if len(ndvi_decid_list) != len(ndvi_everg_list):

        if len(ndvi_decid_list) < len(ndvi_everg_list):
            ndvi_everg_list = random.sample(ndvi_everg_list, len(ndvi_decid_list))
        else:
            ndvi_decid_list = random.sample(ndvi_decid_list, len(ndvi_everg_list))

    print('Number of Deciduous samples after sub-sampling: {}'.format(str(len(ndvi_decid_list))))
    print('Number of Evergreen samples after sub-sampling: {}'.format(str(len(ndvi_everg_list))))

    x_decid = list(elem[0] for elem in ndvi_decid_list)
    y_decid = list(elem[1] for elem in ndvi_decid_list)

    x_everg = list(elem[0] for elem in ndvi_everg_list)
    y_everg = list(elem[1] for elem in ndvi_everg_list)

    # construct a histogram of NDVI values
    step = 1.0 / float(year_div)
    hist, bin_edges = np.histogram(x_decid, bins=year_div)

    bin_edges = list(int(bin_edge) for bin_edge in bin_edges)

    print(bin_edges)

    decid_pctl_upper_curves = [list(), list()]  # list of upper pctl curve for deciduous
    decid_pctl_lower_curves = [list(), list()]  # list of lower pctl curve for deciduous

    everg_pctl_upper_curves = [list(), list()]  # list of upper pctl curve for evergreen
    everg_pctl_lower_curves = [list(), list()]  # list of lower pctl curve for evergreen

    for j in range(len(bin_edges) - 1):
        bin_list = list()
        for elem in ndvi_decid_list:
            if bin_edges[j] <= elem[0] < bin_edges[j+1]:
                bin_list.append(elem[1])

        if len(bin_list) >= min_samp:
            decid_pctl_upper_curves[0].append(np.percentile(bin_list, upctl))
            decid_pctl_upper_curves[1].append(int(np.mean([bin_edges[j], bin_edges[j+1]])))

            decid_pctl_lower_curves[0].append(np.percentile(bin_list, lpctl))
            decid_pctl_lower_curves[1].append(int(np.mean([bin_edges[j], bin_edges[j + 1]])))

        bin_list = list()
        for elem in ndvi_everg_list:
            if bin_edges[j] <= elem[0] < bin_edges[j+1]:
                bin_list.append(elem[1])

        if len(bin_list) >= min_samp:
            everg_pctl_upper_curves[0].append(np.percentile(bin_list, upctl))
            everg_pctl_upper_curves[1].append(int(np.mean([bin_edges[j], bin_edges[j + 1]])))

            everg_pctl_lower_curves[0].append(np.percentile(bin_list, lpctl))
            everg_pctl_lower_curves[1].append(int(np.mean([bin_edges[j], bin_edges[j + 1]])))

    y_e_u = savgol_filter(everg_pctl_upper_curves[0], window, deg)
    y_e_l = savgol_filter(everg_pctl_lower_curves[0], window, deg)
    y_d_u = savgol_filter(decid_pctl_upper_curves[0], window, deg)
    y_d_l = savgol_filter(decid_pctl_lower_curves[0], window, deg)

    # ----------------------------------------------------------------------------------------------------

    _, ax = plt.subplots()

    bins = (25, 25)

    data, x_e, y_e = np.histogram2d(x_decid, y_decid, bins=bins)

    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x_decid, y_decid]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0)

    # Sort the points by density, so that the densest points are plotted last
    idx = np.array(z).argsort()

    x = list(x_decid[i] for i in idx)
    y = list(y_decid[i] for i in idx)
    z = list(z[i] for i in idx)

    ax.scatter(x, y, c=z, alpha=0.75, s=1)
    ax.plot(decid_pctl_upper_curves[1], y_d_u)
    ax.plot(decid_pctl_lower_curves[1], y_d_l)

    plt.xlim(0, 365)
    plt.ylim(-0.05, 1.05)
    plt.savefig(folder + '/decid_v{}.png'.format(version), bbox_inches='tight')

    # ----------------------------------------------------------------------------------------------------

    _, ax = plt.subplots()

    bins = (25, 25)

    data, x_e, y_e = np.histogram2d(x_everg, y_everg, bins=bins)

    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x_everg, y_everg]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0)

    # Sort the points by density, so that the densest points are plotted last
    idx = np.array(z).argsort()

    x = list(x_everg[i] for i in idx)
    y = list(y_everg[i] for i in idx)
    z = list(z[i] for i in idx)

    ax.scatter(x, y, c=z, alpha=0.75, s=1)
    ax.plot(everg_pctl_upper_curves[1], y_e_u)
    ax.plot(everg_pctl_lower_curves[1], y_e_l)

    plt.xlim(0, 365)
    plt.ylim(-0.05, 1.05)
    plt.savefig(folder + '/everg_v{}.png'.format(version), bbox_inches='tight')

