from modules import *
import numpy as np
import seaborn as sns

if __name__ == '__main__':

    infile = "D:\\Shared\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\ABoVE_all_2010_sampV1.csv"

    names, data = Handler(infile).read_csv_as_array()

    header = ['site',
              'sample']

    for name in names[1:-1]:
        header.append(name)

    #print(header)

    site_data = list()

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

    sites = np.unique([elem['site'] for elem in site_data])

    md_list = list()

    for site in sites:

        md_data = Mahalanobis(samples=[elem for elem in site_data if elem['site'] == site],
                              names=['NDVI', 'NDWI', 'NIR'])

        md_data.sample_matrix()
        md_data.cluster_center(method='median')
        md = md_data.calc_distance()

        md_list.append(md)

    print(len(md_list))

    tlen = 0

    for md_vec in md_list:

        num_list = list(i for i, x in enumerate(list(np.isnan(md_vec))) if not x)
        if len(num_list) < 1:
            continue

        md_vec = [md_vec[x] for x in num_list]

        loc = list(i for i, x in enumerate(md_vec) if (x <= np.percentile(md_vec, 50) and x != np.nan))
        tlen += len(loc)
        md_out = list(md_vec[i] for i in loc)

    print(tlen)
