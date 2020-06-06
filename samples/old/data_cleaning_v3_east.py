from geosoup import Vector, Handler, Sublist
from geosoupML import Euclidean
import numpy as np

'''
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
plt.rcParams['axes.labelweight'] = 'bold'
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
'''
'''
v3: This version of data cleaning is without Mahalanobis distance and uses only 
the histogram of decid fraction bins to clean samples. The samples have not been used to extract
Landsat data yet. Once the samples are cleaned they can then be used to extract Landsat data.
This cleaning code is for generating graphics data only
'''


if __name__ == '__main__':

    # static vars--------------------------------------------------------------------------------------------
    version = 8
    trn_perc = 66  # percentage of training samples
    nbins = 100
    cutoff = 55  # percentile at which to cutoff the samples in each bin
    divider_lon = -97.0
    divid_multiplier = 0.3  # multipler for the east side samples
    thresh = 0.025  # distance threshold

    # output file dirsctory
    outdir = "D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/"

    # input file directory
    infile = outdir + "all_samp_pre_v1.csv"

    outfile_east = outdir + "all_samp_postbin_east_v{}.csv".format(version)
    outshpfile_east = outdir + "all_samp_postbin_east_v{}.shp".format(version)

    trn_outfile_east = outdir + "all_samp_post_v{}_trn_samp_east.shp".format(version)
    val_outfile_east = outdir + "all_samp_post_v{}_val_samp_east.shp".format(version)

    boreal_bounds = "D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/STUDY_AREA/boreal/" \
                    "NABoreal_simple_10km_buffer_geo.shp"

    year_bins = [(1984, 1997), (1998, 2002), (2003, 2007), (2008, 2012), (2013, 2018)]

    # script-----------------------------------------------------------------------------------------------

    boreal_vec = Vector(boreal_bounds)
    boreal_geom = Vector.get_osgeo_geom(boreal_vec.wktlist[0])

    year_samp = list(list() for _ in range(len(year_bins)))
    year_samp_reduced = list(list() for _ in range(len(year_bins)))

    # get data and names
    file_data = Handler(infile).read_from_csv(return_dicts=True)
    header = list(file_data[0])

    print('\nTotal samples: {}'.format(str(len(file_data))))

    boreal_samp_count = 0

    # bin all samples based on sample years using year_bins
    for elem in file_data:
        for i, years in enumerate(year_bins):
            if years[0] <= elem['year'] <= years[1]:
                year_samp[i].append(elem)

    # take mean of all samples of the same site that fall in the same year bin
    for i, samp_list in enumerate(year_samp):
        print('year: {}'.format(str(year_bins[i])))
        samp_count = 0
        site_ids = list(set(list(attr_dict['site'] for attr_dict in samp_list)))

        for site_id in site_ids:
            same_site_samp_list = list(samp for samp in samp_list if samp['site'] == site_id)

            lat = same_site_samp_list[0]['Latitude']
            lon = same_site_samp_list[0]['Longitude']

            samp_wkt = Vector.wkt_from_coords([lon, lat])
            samp_geom = Vector.get_osgeo_geom(samp_wkt)

            if boreal_geom.Intersects(samp_geom):

                decid_frac = np.mean(list(site_samp['decid_frac'] for site_samp in same_site_samp_list))
                year = int(np.mean(list(site_samp['year'] for site_samp in same_site_samp_list)))

                # remove spaces in site names,
                # and add year to site name eg: 'site1' + '2007' = 'site1_2007'
                year_samp_reduced[i].append({'site': str(site_id).replace(' ', '') + '_' + str(year),
                                             'year': year,
                                             'decid_frac': decid_frac,
                                             'latitude': lat,
                                             'longitude': lon})
                boreal_samp_count += 1
                samp_count += 1
        print('samp_count: {}'.format(str(samp_count)))

    # flatten the 'by year' list of lists
    decid_frac_samp = list()
    for sublist in year_samp_reduced:
        for elem in sublist:
            decid_frac_samp.append(elem)

    print('Reduced samples: {}'.format(str(len(decid_frac_samp))))

    # extract all decid frac values for calculating histogram
    decid_frac_list = list(samp['decid_frac'] for samp in decid_frac_samp)

    # histogram calculation
    step = 1.0 / float(nbins)
    hist, bin_edges = np.histogram(decid_frac_list, bins=nbins)
    hist = hist.tolist()
    print(hist)

    # calculate the cutoff number per bin
    med = np.ceil(np.percentile(hist, cutoff)).astype(int)
    print('Max bin size: {}'.format(str(med)))

    # calculate number of elements to keep and number to eliminate in each bin
    # calculate histogram bin edges
    hist_diff = list()
    hist_count = list(0 for _ in hist)
    hist_edges = list()
    for i, hist_val in enumerate(hist):
        if hist_val > med:
            hist_diff.append(hist_val - med)
            hist_count[i] = med
        else:
            hist_diff.append(0)
            hist_count[i] = hist_val
        hist_edges.append((bin_edges[i], bin_edges[i + 1]))

    # initialize list of lists for each bin
    binned_decid_dicts = list(list() for _ in hist_edges)
    out_binned_decid_dicts = list(list() for _ in hist_edges)

    # place all samples in their respective bins
    for decid_samp in decid_frac_samp:
        for i, hist_edge in enumerate(hist_edges):
            if i != (len(hist_edges)-1):
                if hist_edge[0] <= decid_samp['decid_frac'] < hist_edge[1]:
                    binned_decid_dicts[i].append(decid_samp)
            else:
                if hist_edge[0] <= decid_samp['decid_frac'] <= hist_edge[1]:
                    binned_decid_dicts[i].append(decid_samp)

    # random selection of samples in bins where count exceeds the bin threshold
    for i, decid_dicts in enumerate(binned_decid_dicts):

        if len(decid_dicts) > hist_count[i]:
            out_binned_decid_dicts[i] = Sublist(decid_dicts).random_selection(hist_count[i])
        else:
            out_binned_decid_dicts[i] = decid_dicts

    print('Samples after bin thresholding: {}'.format(str(sum(hist_count))))

    # flatten the 'by bin' list of lists
    out_decid_frac_samp = list()
    for sublist in out_binned_decid_dicts:
        for elem in sublist:
            out_decid_frac_samp.append(elem)

    # This portion of code calculates the number of samples on each side
    # divider_lon longitude. The aim is to make the samples on the east side
    # equal in number to the samples on the west. The east side samples are
    # vastly large in number and need to be reduced.
    west_count = 0
    east_count = 0
    west_index = list()
    east_index = list()
    east_index_out = list()

    for i, elem in enumerate(out_decid_frac_samp):
        if elem['longitude'] <= divider_lon:
            west_count += 1
            west_index.append(i)
        else:
            east_count += 1
            east_index.append(i)

    print('West count:  {}'.format(west_count))
    print('East count:  {}'.format(east_count))

    east_index_out = Sublist(east_index).copy()
    east_samp_out = list(out_decid_frac_samp[i] for i in east_index)

    east_samp_out = Euclidean(samples=east_samp_out,
                              names=['longitude', 'latitude']).apply_proximity_filter(thresh=thresh)

    out_samp_index = list(range(len(east_samp_out)))
    np.random.shuffle(out_samp_index)

    out_decid_frac_samp_east = list(east_samp_out[i] for i in out_samp_index)

    # write csv file
    Handler.write_to_csv(out_decid_frac_samp_east,
                         outfile_east)

    # write shp file---------
    attribute_types = {'site': 'str',
                       'year': 'int',
                       'decid_frac': 'float'}

    trn_data = list()

    print('Total {} sites'.format(str(len(out_decid_frac_samp_east))))

    ntrn = int((trn_perc * len(out_decid_frac_samp_east))/100.0)
    nval = len(out_decid_frac_samp_east) - ntrn

    # randomly select training samples based on number
    trn_sites = Sublist(range(len(out_decid_frac_samp_east))).random_selection(ntrn)

    # get the rest of samples as validation samples
    val_sites = Sublist(range(len(out_decid_frac_samp_east))).remove(trn_sites)

    # print IDs
    print(len(trn_sites))
    print(len(val_sites))

    wkt_list = list()
    attr_list = list()

    trn_wkt_list = list()
    trn_attr_list = list()

    val_wkt_list = list()
    val_attr_list = list()

    for i, row in enumerate(out_decid_frac_samp_east):
        elem = dict()
        for header in list(attribute_types):
            elem[header] = row[header]

        wkt = Vector.wkt_from_coords([row['longitude'], row['latitude']],
                                     geom_type='point')

        if i in trn_sites:
            trn_wkt_list.append(wkt)
            trn_attr_list.append(elem)
        elif i in val_sites:
            val_wkt_list.append(wkt)
            val_attr_list.append(elem)

        wkt_list.append(wkt)
        attr_list.append(elem)

    vector = Vector.vector_from_string(wkt_list,
                                       out_epsg=4326,
                                       vector_type='point',
                                       attributes=attr_list,
                                       attribute_types=attribute_types,
                                       verbose=True,
                                       full=False)

    print(vector)
    vector.write_vector(outshpfile_east)

    trn_vec = Vector.vector_from_string(trn_wkt_list,
                                        out_epsg=4326,
                                        vector_type='point',
                                        attributes=trn_attr_list,
                                        attribute_types=attribute_types,
                                        verbose=True,
                                        full=False)
    print(trn_vec)
    trn_vec.write_vector(trn_outfile_east)

    val_vec = Vector.vector_from_string(val_wkt_list,
                                        out_epsg=4326,
                                        vector_type='point',
                                        attributes=val_attr_list,
                                        attribute_types=attribute_types,
                                        verbose=True,
                                        full=False)
    print(val_vec)
    val_vec.write_vector(val_outfile_east)
    # write shp file: end---------

    print('Done!')
