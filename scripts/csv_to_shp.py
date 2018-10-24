from modules import Vector
import pandas as pd


if __name__ == '__main__':

    csvfile = "/projects/NAU/landsat_deciduous/data/RSN_Lat_Longs.csv"
    outfile = "/projects/NAU/landsat_deciduous/data/RSN_Lat_Longs.shp"

    samp_data = pd.read_csv(csvfile)
    headers = list(samp_data)

    print(headers)

    spref_str = '+proj=longlat +datum=WGS84'

    wkt_list = list()
    attr_list = list()
    attribute_types = {'age': 'str',
                       'rsn_name': 'str',
                       'site_id': 'str'}

    for _, row in samp_data.iterrows():
        elem = dict()
        for header in headers:
            elem[header] = row[header]

        wkt_list.append(Vector.wkt_from_coords((elem['lon'], elem['lat']),
                                               geom_type='point'))

        attr_list.append({'age': elem['age'],
                          'rsn_name': elem['rsn_name'],
                          'site_id': elem['site_id']})

    vector = Vector.vector_from_string(wkt_list,
                                       spref_string=spref_str,
                                       spref_string_type='proj4',
                                       vector_type='point',
                                       attributes=attr_list,
                                       attribute_types=attribute_types)

    print(vector)

    vector.write_to_file(outfile)
