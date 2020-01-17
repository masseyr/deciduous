from modules import *
import numpy as np
import pandas as pd
import random

pd.set_option('display.max_columns', None)

if __name__ == '__main__':

    # static vars--------------------------------------------------------------------------------------------
    md_cutoff = 75  # cutoff for Mahalanobis distance
    trn_perc = 100  # percentage of training samples
    md_clean = True  # if the sample should be cleaned
    mean_site = True  # if all the pixels at a site should be averaged
    tree_cover_thresh = 50

    # output file dirsctory
    outdir = "C:\\users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\"

    # input file directory
    infile = outdir + "ABoVE_CAN_AK_all_2010_samp_ba_v2.csv"

    # filenames
    if md_clean:

        trn_outfile = outdir + "ABoVE_AK_CAN_all_2010_trn_samp_clean_md{}_mean_v3.csv".format(str(md_cutoff))
        val_outfile = outdir + "ABoVE_AK_CAN_all_2010_val_samp_clean_md{}_mean_v3.csv".format(str(md_cutoff))

    else:

        trn_outfile = outdir + "ABoVE_AK_CAN_all_2010_trn_samp_original.csv"
        val_outfile = outdir + "ABoVE_AK_CAN_all_2010_val_samp_original.csv"

    # names to append to samples' header
    header = ['site',
              'sample']

    # bands used as features for cleaning the samples
    bandnames = ['NDVI',
                 'NDVI_1',
                 'NDVI_2']

    # script-----------------------------------------------------------------------------------------------

    # get data and names
    names, data = Handler(infile).read_csv_as_array()

    for name in names[1:-1]:
        header.append(name)

    print(header)

    site_data = list()

    # convert strings like '1_125_3' into sites and samples
    for elem in data:
        index = elem[0].split('_')

        if len(index) == 3:

            site_id = int(index[0])*10000 + int(index[1])
            samp_id = int(index[2])

        elif len(index) == 4:

            site_id = int(index[0])*10000 + int(index[1])*1000 * int(index[2])
            samp_id = int(index[3])

        else:
            print('Invalid Sample ID: {}'.format(elem))
            continue

        vals = dict()

        vals['site'] = site_id
        vals['sample'] = samp_id

        for i in range(1, len(names)):
            vals[names[i]] = elem[i]

        site_data.append(vals)

    data = None
    trn_data = list()

    # list of unique sites
    sites = np.unique([elem['site'] for elem in site_data])

    print('Total {} sites'.format(str(len(sites))))

    ntrn = int((trn_perc * len(sites))/100.0)
    nval = len(sites) - ntrn

    # randomly select training samples based on number
    trn_sites = random.sample(sites, ntrn)

    # get the rest of samples as validation samples
    val_sites = Sublist(sites).remove(trn_sites)

    # print IDs
    print(trn_sites)
    print(len(trn_sites))

    print('-------------------------')

    print(val_sites)
    print(len(val_sites))

    print('-------------------------')

    # loop through all sites
    for site in trn_sites:

        # collect samples for the site
        samples = list(elem for elem in site_data if elem['site'] == site)

        # if the samples are to be cleaned
        if md_clean:

            if len(samples) > len(bandnames):

                # initialize mahalanobis class
                md_data = Mahalanobis(samples=samples,
                                      names=bandnames)

                md_data.sample_matrix()
                md_data.cluster_center(method='median')
                md_vec = md_data.calc_distance()

                # eliminate NaNs in Mahalanobis dist vector
                num_list = list(i for i, x in enumerate(list(np.isnan(md_vec))) if not x)
                if len(num_list) < 1:
                    continue

                md_vec = [md_vec[x] for x in num_list]

                # find all MD values that as less than cutoff percentile
                loc = list(i for i, x in enumerate(md_vec) if (x <= np.percentile(md_vec, md_cutoff) and x != np.nan))

                out_samples = list(samples[i] for i in loc)

            else:

                print('Too few samples for cleaning')

                out_samples = samples

        else:
            out_samples = samples

        if mean_site:

            out_nsamp = len(out_samples)

            if out_nsamp > 1:
                mean_out_sample = dict()

                names = [key for key, _ in out_samples[0].items()]

                for name in names:
                    out_value = float(sum(list(sample[name] for sample in out_samples))) / out_nsamp
                    mean_out_sample[name] = out_value

                mean_out_sample['sample'] = 0

                out_samples = list()

                out_samples.append(mean_out_sample)

        # add all filtered samples to out list
        for sample in out_samples:
            if sample['TREECOVER'] > tree_cover_thresh:
                trn_data.append(sample)

    print('\n\nTotal samples: {}'.format(str(len(trn_data))))

    val_data = list()

    if len(val_sites) > 0:

        for site in val_sites:

            samples = list(elem for elem in site_data if elem['site'] == site)

            if md_clean:

                if len(samples) > len(bandnames):

                    # initialize mahalanobis class
                    md_data = Mahalanobis(samples=samples,
                                          names=bandnames)

                    md_data.sample_matrix()
                    md_data.cluster_center(method='median')
                    md_vec = md_data.calc_distance()

                    # eliminate NaNs in Mahalanobis dist vector
                    num_list = list(i for i, x in enumerate(list(np.isnan(md_vec))) if not x)
                    if len(num_list) < 1:
                        continue

                    md_vec = [md_vec[x] for x in num_list]

                    # find all MD values that as less than cutoff percentile
                    loc = list(i for i, x in enumerate(md_vec) if (x <= np.percentile(md_vec, md_cutoff) and x != np.nan))

                    out_samples = list(samples[i] for i in loc)

                else:

                    print('Too few samples for cleaning')

                    out_samples = samples

            else:

                out_samples = samples

            if mean_site:

                out_nsamp = len(out_samples)

                if out_nsamp > 1:
                    mean_out_sample = dict()

                    names = [key for key, _ in out_samples[0].items()]

                    for name in names:
                        out_value = float(sum(list(sample[name] for sample in out_samples))) / out_nsamp
                        mean_out_sample[name] = out_value

                    mean_out_sample['sample'] = 0

                    out_samples = list()

                    out_samples.append(mean_out_sample)

            # val_samp = dict(site=samples[0]['site'], decid_frac=samples[0]['decid_frac'])

            for sample in out_samples:
                if sample['TREECOVER'] > tree_cover_thresh:
                    val_data.append(sample)

            # val_data.append(samples[0])

    # Make a pandas dataframe
    t_df = pd.DataFrame(trn_data)



    drop_columns = [
                    'geo', 'site', 'sample',  'TREECOVER',
                    'BLUE', 'BLUE_1', 'BLUE_2',
                    'GREEN', 'GREEN_1', 'GREEN_2',
                    'RED', 'RED_1', 'RED_2',
                    'SWIR2', 'SWIR2_1', 'SWIR2_2',
                    #  'RATIO57', 'RATIO57_1', 'RATIO57_2',
                    #  'NBR', 'NBR_1', 'NBR_2',
                    #  'ND57', 'ND57_1', 'ND57_2',
                    #  'NDVI', 'NDVI_1', 'NDVI_2',
                    #  'NMDI', 'NMDI_1', 'NMDI_2',
                    #  'SWIR1', 'SWIR1_1', 'SWIR1_2',
                    'EVI', 'EVI_1', 'EVI_2'
                    ]

    # remove useless columns
    t_df.drop(columns=drop_columns,
              inplace=True)

    # muliply some columns with 10000
    multiply_cols = [
                     'NIR', 'NIR_1', 'NIR_2',
                     'SWIR1', 'SWIR1_1', 'SWIR1_2',
                    ]

    for i in range(0, len(multiply_cols)):
        t_df[multiply_cols[i]] = t_df[multiply_cols[i]]*0.0001

    print(t_df.head())

    t_df = t_df[['NIR', 'SWIR1', 'NDVI', 'NDWI', 'NBR', 'RATIO57', 'ND57', 'NMDI',
                'NIR_1', 'SWIR1_1', 'NDVI_1', 'NDWI_1', 'NBR_1', 'RATIO57_1', 'ND57_1',
                'NMDI_1', 'NIR_2', 'SWIR1_2', 'NDVI_2', 'NDWI_2', 'NBR_2', 'RATIO57_2',
                'ND57_2', 'NMDI_2', 'decid_frac']]

    """
    all_cols = t_df.columns.tolist()

    for i in range(0, len(all_cols)):
        t_df[all_cols[i]] = [float(elem) for elem in t_df[all_cols[i]]]
    """
    # write training samples to csv file
    t_df.to_csv(trn_outfile,
                index=False)

    if len(val_sites) > 0:
        v_df = pd.DataFrame(val_data)

        v_df.drop(columns=drop_columns,
                  inplace=True)

        # write validation samples to csv file
        v_df.to_csv(val_outfile,
                    index=False)