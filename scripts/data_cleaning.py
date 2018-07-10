from modules import *
import numpy as np
import pandas as pd
import random

if __name__ == '__main__':

    # static vars--------------------------------------------------------------------------------------------
    md_cutoff = 95
    ratio = 0.95

    outdir = "C:\\users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\"

    infile = outdir + "ABoVE_CAN_AK_all_2010_samp.csv"
    trn_outfile = outdir + "ABoVE_CAN_AK_all_2010_trn_samp_clean_md{}.csv".format(str(md_cutoff))
    val_outfile = outdir + "ABoVE_CAN_AK_all_2010_val_samp_clean_md{}.csv".format(str(md_cutoff))

    header = ['site',
              'sample']

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

    # for data in site_data:
    #     print(data)

    data = None
    trn_data = list()

    # list of unique sites
    sites = np.unique([elem['site'] for elem in site_data])

    print('Total {} sites'.format(str(len(sites))))

    ntrn = int(ratio * len(sites))
    nval = len(sites) - ntrn

    trn_sites = random.sample(sites, ntrn)
    val_sites = Sublist(sites).remove(trn_sites)

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

        """
        print('-----------------------------------------------------------')
        print(site)
        for sample in samples:
            print(sample)
        """

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

        # add all filtered samples to out list
        for sample in out_samples:
            trn_data.append(sample)

    print('\n\nTotal samples: {}'.format(str(len(trn_data))))

    val_data = list()

    for site in val_sites:

        samples = list(elem for elem in site_data if elem['site'] == site)

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

        # val_samp = dict(site=samples[0]['site'], decid_frac=samples[0]['decid_frac'])

        for sample in out_samples:
            val_data.append(sample)

        # val_data.append(samples[0])

    # Make a pandas dataframe
    t_df = pd.DataFrame(trn_data)
    v_df = pd.DataFrame(val_data)

    # remove useless columns
    t_df.drop(columns=['geo', 'site', 'sample',
                       'BLUE', 'BLUE_1', 'BLUE_2',
                     'GREEN', 'GREEN_1', 'GREEN_2',
                     'RED', 'RED_1', 'RED_2',
                     'SWIR2', 'SWIR2_1', 'SWIR2_2',
                       "EVI", "EVI_1", "EVI_2"],
              inplace=True)

    v_df.drop(columns=['geo', 'site', 'sample',
                       'BLUE', 'BLUE_1', 'BLUE_2',
                     'GREEN', 'GREEN_1', 'GREEN_2',
                     'RED', 'RED_1', 'RED_2',
                     'SWIR2', 'SWIR2_1', 'SWIR2_2',
                       "EVI", "EVI_1", "EVI_2"],
              inplace=True)

    """
    # muliply some columns with 10000
    multiply_cols = ['BLUE', 'BLUE_1', 'BLUE_2',
                     'GREEN', 'GREEN_1', 'GREEN_2',
                     'RED', 'RED_1', 'RED_2',
                     'NIR', 'NIR_1', 'NIR_2',
                     'SWIR1', 'SWIR1_1', 'SWIR1_2',
                     'SWIR2', 'SWIR2_1', 'SWIR2_2']

    for i in range(0, len(multiply_cols)):
        df[multiply_cols[i]] = df[multiply_cols[i]]*10000
    """

    # write to csv file
    t_df.to_csv(trn_outfile,
                index=False)

    v_df.to_csv(val_outfile,
                index=False)
