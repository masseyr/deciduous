from modules import Vector


if __name__ == '__main__':

    csvfile = 'c:/temp/cafi_site_basal_area_deciduousness_smry_1994_2014.csv'
    outfile = 'c:/temp/cafi_site_basal_area_deciduousness_smry_1994_2014.shp'

    vec = Vector(vectorfile=outfile,
                 name='cafi_sites',
                 source_type='ESRI Shapefile')

    vec.geometry_type = 'point'
    vec.epsg = 4326
    vec.point_features_from_csv(csvfile=csvfile,
                                geometry_columns=['lat',
                                                  'lon'])
    vec.attribute_def = {'site': 'int',
                         'year': 'int',
                         'decid_frac': 'float',
                         'decid_sd': 'float'}

    vec.construct_point_vector()

