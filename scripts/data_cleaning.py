from modules import *
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # static vars--------------------------------------------------------------------------------------------
    outdir = "c:\\users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\"

    infile = outdir + "ABoVE_all_2010_sampV1.csv"
    outfile = outdir + "ABoVE_all_2010_sampV1_clean_md60.csv"

    md_cutoff = 60

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

    site_data = list()

    # convert strings like '1_125_3' into sites and samples
    for elem in data:
        index = elem[0].split('_')
        site_id = int(index[0])*1000 + int(index[1])
        samp_id = int(index[2])

        vals = dict()

        vals['site'] = site_id
        vals['sample'] = samp_id

        for i in range(1, len(names)):
            vals[names[i]] = elem[i]

        site_data.append(vals)

    data = None
    out_data = list()

    # list of unique sites
    sites = np.unique([elem['site'] for elem in site_data])

    # loop through all sites
    for site in sites:

        # collect samples for the site
        samples = list(elem for elem in site_data if elem['site'] == site)

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

        # add all filtered samples to out list
        for sample in out_samples:
            out_data.append(sample)

    print(len(out_data))

    # Make a pandas dataframe
    df = pd.DataFrame(out_data)

    # remove useless columns
    df.drop(columns=['geo', 'site', 'sample',
                     'RATIO57', 'RATIO57_1', 'RATIO57_2'],
            inplace=True)

    # muliply some columns with 10000
    multiply_cols = ['BLUE', 'BLUE_1', 'BLUE_2',
                     'GREEN', 'GREEN_1', 'GREEN_2',
                     'RED', 'RED_1', 'RED_2',
                     'NIR', 'NIR_1', 'NIR_2',
                     'SWIR1', 'SWIR1_1', 'SWIR1_2',
                     'SWIR2', 'SWIR2_1', 'SWIR2_2']

    for i in range(0, len(multiply_cols)):
        df[multiply_cols[i]] = df[multiply_cols[i]]*10000

    # write to csv file
    df.to_csv(outfile,
              index=False)
