from geosoup import Vector, Handler, Opt
from geosoupML import Mahalanobis
import datetime
import numpy as np
import copy
import math


def reformat_samples(site_dicts):
    """
    Method to reformat samples for use in Mahalanobis distance algo
    :param site_dicts: list of sample dictionaries
                example of a site dict:

    {'1_70066 G000005_1994': {'decid_frac': 0.24758,
                              'geom': 'POINT(-121.298569584 55.7763510752)',
                              'site_year': 1994,
                              'data':  {
                                       '262_1993': {'sensor': 'LT05', 'img_year': 1993, 'img_jday': 262,
                                                    'bands': {'blue': 0.0266, 'swir1': 0.0756, 'swir2': 0.0244,
                                                              'green': 0.0481, 'nir': 0.1523, 'red': 0.0344
                                                              }
                                                   },
                                       '263_1984': {'sensor': 'LT05', 'img_year': 1984, 'img_jday': 263,
                                                    'bands': {'blue': 0.0205, 'swir1': 0.0801, 'swir2': 0.0352,
                                                              'green': 0.0516, 'nir': 0.1649, 'red': 0.0491
                                                             }
                                                    }
                                       '264_1984': {'sensor': 'LT05', 'img_year': 1984, 'img_jday': 263,
                                                    'bands': {'blue': 0.0205, 'swir1': 0.0801, 'swir2': 0.0352,
                                                              'green': 0.0516, 'nir': 0.1649, 'red': 0.0491
                                                             }}}}}

    :return: list of dicts
    """

    out_dicts = list()
    for site_elem in site_dicts:
        out_dict = copy.deepcopy(site_elem)
        data = copy.deepcopy(out_dict['data'])
        out_dict.pop('data')

        count = 0
        for samp_dict in data:
            if len(samp_dict) != 0:
                count += 1
                jday_year, samp_data = samp_dict.items()[0]
                temp_elem = dict()
                temp_elem.update(samp_data)
                temp_elem.pop('bands')
                temp_elem.update(samp_data['bands'])

                for key, val in temp_elem.items():
                    key = key + '_' + str(count)
                    out_dict.update({key: val})

        if count == 3:
            out_dicts.append(out_dict)

    return out_dicts


def percentile(inp_list, pctl):
    """
    Find the percentile of a list of values.
    """
    if not inp_list:
        return None

    sorted_arr = sorted(inp_list)
    k = (len(sorted_arr)-1) * (float(pctl)/100.0)
    f = int(math.ceil(k))

    return sorted_arr[f]


def mean(inp_list):
    """
    compute mean of list
    :param inp_list: input list
    :return: float
    """
    return np.mean(inp_list)


def ndvi_pctl_composite(site_samp_dict,
                        pctl=75,
                        jday_start=0,
                        jday_end=365):

    """
    Calculate the percentile composite from a list of values in a site dictionary
    :param site_samp_dict: Dictionary of Extracted Landsat values for a site
    :param pctl: Percentile to use for compositing
    :param jday_start: Start julian day for compositing
    :param jday_end: End julian day for compositing
    :return: Dictionary
    """

    ndvi_list = list()
    for ii, data_dict in site_samp_dict.items():
        ndvi = int(normalized_difference(data_dict['bands']['nir'],
                                         data_dict['bands']['red'],
                                         adj=0.0) * 10000.0)

        if jday_start <= data_dict['img_jday'] <= jday_end:

            ndvi_list.append((data_dict['img_jday'],
                              data_dict['img_year'],
                              ii,
                              ndvi))

    if len(ndvi_list) > 0:
        pctl_th_ndvi = percentile(list(elem[3] for elem in ndvi_list), pctl)
        ii = list(elem[2] for elem in ndvi_list if elem[3] == pctl_th_ndvi)[0]

        out_dict = copy.deepcopy(site_samp_dict[ii])

        out_dict['bands'].update({'ndvi': normalized_difference(out_dict['bands']['nir'],
                                                                out_dict['bands']['red'])})

        out_dict['bands'].update({'ndwi': normalized_difference(out_dict['bands']['nir'],
                                                                out_dict['bands']['swir2'])})

        out_dict['bands'].update({'nbr': normalized_difference(out_dict['bands']['nir'],
                                                               out_dict['bands']['swir1'])})

        out_dict['bands'].update({'savi': normalized_difference(out_dict['bands']['nir'],
                                                                out_dict['bands']['red'],
                                                                adj=0.5)})

        out_dict['bands'].update({'vari': enhanced_normalized_difference(out_dict['bands']['red'],
                                                                         out_dict['bands']['green'],
                                                                         out_dict['bands']['blue'],
                                                                         gain=1.0,
                                                                         c3=-1.0,
                                                                         adj=0.0)})
        return {ii: out_dict}

    else:
        return {}


def normalized_difference(b1,
                          b2,
                          adj=0.0,
                          adj_index=1.0,
                          additive=0.0):

    """
    Normalized difference between two bands (based on NDVI formula)
    :param b1: First band
    :param b2: Second band
    :param adj: Adjustment (useful for indices such as SAVI, as canopy cover adjustment)
    :param adj_index: Adjustment index
    :param additive: Additive
    :return: Float
    """

    if float(b1)+float(b2)+adj != 0.0:
        return ((1 + adj**adj_index)*(float(b1)-float(b2)))/(float(b1)+float(b2)+adj)+additive
    else:
        return 0.0


def enhanced_normalized_difference(b1,
                                   b2,
                                   b3,
                                   adj=0.0,
                                   c1=1.0,
                                   c2=1.0,
                                   c3=1.0,
                                   gain=0.0):
    """
    Method to calculate enhanced normalized difference between two bands using a third band
    This is equivalent to:
                        GAIN * [(BAND1  -  BAND2)/(BAND1 * COEFF1  +  BAND2 * COEFF1  +  BAND3 * COEFF3 + ADJSTMNT)]

    :param b1: First band
    :param b2: Second Band
    :param b3: Third Band
    :param adj: Adjustment factor
    :param c1: First Coefficient
    :param c2: Second Coefficient
    :param c3: Third Coefficient
    :param gain: Gain value for the index
    :return: Float
    """
    if (c1*float(b1) + c2*float(b2) + c3*float(b3) + adj) != 0.0:
        return gain*((float(b1)-float(b2))/(c1*float(b1) + c2*float(b2) + c3*float(b3) + adj))
    else:
        return 0.0


def correct_landsat_sr(bands_dict,
                       sensor,
                       scale=1.0):
    """
    Function to correct reflectance values in the bands_dict to match Landsat 7 output
    Landsat 8 coefficients based on roy et al 2016 DOI: 10.1016/j.rse.2015.12.024
    Landsat 5 coefficients based on sulla-menashe et al 2016 DOI: 10.1016/j.rse.2016.02.041
    :param sensor: Options- LT05, LE07, LC08
    :param bands_dict: Dictionary of band values
    :param scale: Number to scale bands with
    :return:
    """

    in_bands = {
        'LT05': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
        'LE07': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        'LC08': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
    }

    if sensor not in in_bands:
        raise ValueError('Invalid sensor name. Valid names are: landsat_5, landsat_7, landsat_8')

    out_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

    multi_coeff = {
        'LT05': [0.91996, 0.92764, 0.88810, 0.95057, 0.96525, 0.99601],
        'LE07': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'LC08': [0.8850, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165]
    }

    add_coeff = {
        'LT05': [0.0037, 0.0084, 0.0098, 0.0038, 0.0029, 0.0020],
        'LE07': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'LC08': [0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116]
    }

    out_dict = {}
    for ib, in_band in enumerate(in_bands[sensor]):
        out_dict[out_bands[ib]] = (float(bands_dict[in_band]) * scale) * multi_coeff[sensor][ib] + \
                                  add_coeff[sensor][ib]

    for band_name in ('slope', 'aspect', 'elevation'):
        if band_name in bands_dict:
            out_dict[band_name] = bands_dict[band_name]

    return out_dict


def extract_date(string):
    """
    Method to extract dates from a Landsat file name
    :param string: Landsat file name
    :return: Dictionary
    """
    # LT05_L1TP_035020_19960605_20170104_01_T1
    list_str = string.split('_')
    return {
        'sensor': list_str[0],
        'level': list_str[1],
        'pathrow': (int(list_str[2][0:3]), int(list_str[2][3:6])),
        'date': datetime.datetime.strptime(list_str[3], '%Y%m%d')
    }


def unclear_value(x,
                  length=16,
                  only_clear=True,
                  verbose=False):
    """
    Clear value using pixel_qa band
        fill_bit = (0,)
        clear_bit = (1,)
        water_bit = (2,)
        cloud_shadow_bit = (3,)
        snow_bit = (4,)
        cloud_bit = (5,)
        cloud_conf_bit = (6, 7)
        cirrus_conf_bit = (8, 9)
        terrain_occlusion = (10,)

    :param x: pixel_qa band value
    :param length: length of bit string
    :param verbose: if the qa inference should be printed
    :param only_clear: If only pixels with clear tag should be selected
    :return: True if not clear OR False if clear
    """
    try:
        val = int(x)
    except ValueError:
        return True

    bit_str = ''.join(reversed(np.binary_repr(val, length)))
    fill = bit_str[0] == '1'
    clear = bit_str[1] == '1'
    water = bit_str[2] == '1'
    cloud_shadow = bit_str[3] == '1'
    snow = bit_str[4] == '1'
    cloud = bit_str[5] == '1'
    cloud_conf = bit_str[6:8] in ('11', '01')
    cirrus_conf = bit_str[8:10] in ('11', '01')
    terrain_occ = bit_str[10] == '1'
    low_cloud_conf = bit_str[6:8] == '10'
    low_cirrus_conf = bit_str[8:10] == '10'

    clear_condition = [clear, water]
    clear_result = ['clear', 'water']

    unclear_condition = [fill, cloud_shadow, snow, cloud,
                         cloud_conf, cirrus_conf, terrain_occ]
    unclear_result = ['fill', 'cloud_shadow', 'snow', 'cloud',
                      'cloud_conf', 'cirrus_conf', 'terrain_occlusion']

    if only_clear:
        unclear_condition += [low_cloud_conf, low_cirrus_conf]
        unclear_result += ['low_cloud_conf', 'low_cirrus_conf']
    else:
        clear_condition += [low_cloud_conf, low_cirrus_conf]
        clear_result += ['low_cloud_conf', 'low_cirrus_conf']

    if any(clear_condition):
        if verbose:
            print(', '.join(out_result for ii, out_result in
                            enumerate(clear_result) if clear_condition[ii]))
            print(np.binary_repr(int(x), length))
        return False

    elif any(unclear_condition):
        if verbose:
            print(', '.join(out_result for ii, out_result in
                            enumerate(unclear_result) if unclear_condition[ii]))
            print(np.binary_repr(int(x), length))
        return True

    else:
        if verbose:
            print('Out of range')
            print(np.binary_repr(int(x), length))
        return True


def saturated_bands(x, length=8):
    """
    Radiometric saturation
    :param x: pixel value
    :param length: bit length
    :return: 0 if saturated or 1 if clear
    """
    try:
        val = int(x)
    except ValueError:
        return True

    bit_str = ''.join(reversed(np.binary_repr(val, length)))

    if bit_str[0] == '1':
        return True
    else:
        return False


def read_gee_extract_data(filename):
    """
    Method to read sample data in the form of a site dictionary with samples dicts by year
    :param filename: Input data file name
    :return: dict of list of dicts by year
    """

    lines = Handler(filename).read_from_csv(return_dicts=True)

    site_dict = dict()
    line_counter = 0
    for j, line in enumerate(lines):

        include = True
        for key, val in line.items():
            if type(val).__name__ == 'str':
                if val == 'None':
                    include = False

        if saturated_bands(line['radsat_qa']) \
                or line['GEOMETRIC_RMSE_MODEL'] > 15.0 \
                or unclear_value(line['pixel_qa']):
            include = False

        if include:
            line_counter += 1
            site_year = str(line['site']) + '_' + str(line['year'])

            if site_year not in site_dict:
                geom_wkt = Vector.wkt_from_coords((line['longitude'], line['latitude']))
                site_dict[site_year] = {'geom': geom_wkt,
                                        'decid_frac': line['decid_frac'],
                                        'data': dict(),
                                        'site_year': line['year'],
                                        'site': line['site']}

            temp_dict = dict()

            sensor_dict = extract_date(line['LANDSAT_ID'])

            temp_dict['img_jday'] = sensor_dict['date'].timetuple().tm_yday
            temp_dict['img_year'] = sensor_dict['date'].timetuple().tm_year
            temp_dict['sensor'] = sensor_dict['sensor']

            bands = list('B' + str(ii + 1) for ii in range(7)) + ['slope', 'elevation', 'aspect']

            band_dict = dict()
            for band in bands:
                if band in line:
                    band_dict[band] = line[band]

            temp_dict['bands'] = correct_landsat_sr(band_dict,
                                                    sensor_dict['sensor'],
                                                    scale=0.0001)

            site_dict[site_year]['data'].update({'{}_{}'.format(str(temp_dict['img_jday']),
                                                                str(temp_dict['img_year'])): temp_dict})

    # print(line_counter)

    return site_dict


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------

    version = 30

    md_clean = True
    md_names = ['ndvi_1', 'ndvi_2', 'ndvi_3']
    md_cutoff = 90
    md_nbins = 25

    composite_pctl = 50

    # ----------------------------------------------------------------------------------------------------

    # bins for the calculated composites
    year_bins = [(1990, 1997), (1998, 2002), (2003, 2007), (2008, 2012), (2013, 2018)]

    s1_jday1 = 90
    s1_jday2 = 165

    s2_jday1 = 180
    s2_jday2 = 240

    s3_jday1 = 255
    s3_jday2 = 330

    # ----------------------------------------------------------------------------------------------------

    folder = "/home/richard" \
             "/Dropbox/projects/NAU/landsat_deciduous/data/samples/gee_extract/"

    # datafile = folder + "gee_samp_extract_v2019_01_29T08_06_32.csv"
    datafile = folder + "gee_samp_extract_postbin_v8_east_2019_08_29T_all.csv"

    outfile = folder + "gee_samp_extract_postbin_v8_east_2019_08_29T_all_formatted_md.csv".format(str(version))

    # ----------------------------------------------------------------------------------------------------

    gee_data_dict = read_gee_extract_data(datafile)

    for elem in gee_data_dict.items()[0:10]:
        Opt.cprint(elem)

    # ----------------------------------------------------------------------------------------------------
    Opt.cprint('Total number of data points before cleaning : {}'.format(len(gee_data_dict)))

    composite_data_dict = dict()
    total_sites = 0
    for site_index, site_dict in gee_data_dict.items():

        temp_dict = copy.deepcopy(site_dict)
        temp_dict.pop('data')

        temp_dict['data'] = list()

        temp_dict['data'].append(ndvi_pctl_composite(site_dict['data'],
                                                     pctl=composite_pctl,
                                                     jday_start=s1_jday1,
                                                     jday_end=s1_jday2))

        temp_dict['data'].append(ndvi_pctl_composite(site_dict['data'],
                                                     pctl=composite_pctl,
                                                     jday_start=s2_jday1,
                                                     jday_end=s2_jday2))

        temp_dict['data'].append(ndvi_pctl_composite(site_dict['data'],
                                                     pctl=composite_pctl,
                                                     jday_start=s3_jday1,
                                                     jday_end=s3_jday2))

        if len(temp_dict['data']) == 3:
            composite_data_dict.update({site_index: temp_dict})
        total_sites += 1

    Opt.cprint('\n')
    Opt.cprint('After removing samples with less than 3 temporal data points: {}'.format(str(len(composite_data_dict))))

    main_attr = ['geom', 'decid_frac', 'site', 'site_year']
    composite_attr = ['img_jday', 'img_year', 'sensor']
    topo_attr = ['elevation', 'slope', 'aspect']
    band_attr = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'nbr', 'vari', 'savi']

    fid = 0
    max_len = 0
    samp_list = list()
    for _, elem in composite_data_dict.items():
        samp_dict = dict()

        for attr in main_attr:
            samp_dict[attr] = elem[attr]

        for i, samp_data in enumerate(elem['data']):
            for _, prop in samp_data.items():
                for attr in composite_attr:
                    samp_dict[attr + '_' + str(i+1)] = prop[attr]
                for attr in band_attr:
                    samp_dict[attr + '_' + str(i+1)] = prop['bands'][attr]
                for attr in topo_attr:
                    if attr not in samp_dict:
                        samp_dict[attr] = prop['bands'][attr]
        samp_dict['id'] = fid
        samp_list.append(samp_dict)
        fid += 1

        if max_len < len(samp_dict):
            max_len = len(samp_dict)

    out_list = list(elem for elem in samp_list if len(elem) == max_len)

    Opt.cprint('After formatting: {}'.format(str(len(out_list))))

    # if the samples are to be cleaned using mahalanobis distance
    if md_clean:

        binned_samp_dict_list = list(list() for _ in range(md_nbins))
        out_samples = list()

        # histogram calculation
        step = 1.0 / float(md_nbins)
        hist, bin_edges = np.histogram(list(elem['decid_frac'] for elem in out_list), bins=md_nbins)

        hist_edges = list((bin_edges[i], bin_edges[i + 1]) for i in range(len(hist)))

        # place all samples in their respective bins
        for samp in out_list:
            for i, hist_edge in enumerate(hist_edges):
                if i != (len(hist_edges) - 1):
                    if hist_edge[0] <= samp['decid_frac'] < hist_edge[1]:
                        binned_samp_dict_list[i].append(samp)
                else:
                    if hist_edge[0] <= samp['decid_frac'] <= hist_edge[1]:
                        binned_samp_dict_list[i].append(samp)

        for binned_samp_dicts in binned_samp_dict_list:

            if len(binned_samp_dicts) > len(band_attr + topo_attr):

                md_data = Mahalanobis(samples=binned_samp_dicts,
                                      names=md_names)

                md_data.sample_matrix()
                md_data.cluster_center(method='median')
                md_vec = md_data.calc_distance()

                # eliminate NaNs in Mahalanobis dist vector
                num_list = list(i for i, x in enumerate(list(np.isnan(md_vec))) if not x)

                if len(num_list) > 1:

                    md_vec = [md_vec[x] for x in num_list]

                    # find all MD values that as less than cutoff percentile
                    loc = list(i for i, x in enumerate(md_vec)
                               if (x <= np.percentile(md_vec, md_cutoff) and x != np.nan))

                    out_samples += list(binned_samp_dicts[i] for i in loc)

                else:
                    out_samples += binned_samp_dicts

            else:

                Opt.cprint('Too few samples for cleaning')

                out_samples += binned_samp_dicts

        Opt.cprint('After Mahalanobis dist removal of all samp above {} percentile: {}'.format(str(md_cutoff),
                                                                                               str(len(out_samples))))
    else:
        out_samples = out_list

    Handler.write_to_csv(out_samples, outfile)

    Opt.cprint('Done!')



