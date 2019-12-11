from geosoup import Vector, Handler
import pandas as pd
import json
from osgeo import ogr
import numpy as np


if __name__ == '__main__':

    infilename = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/samples/CAN_PSP/" \
        "CAN_PSPs_Hember-20180207T213138Z-001/CAN_PSPs_Hember/NAFP_L4_SL_ByJur_R16d_ForBrendanRogers1.csv"

    outfilename = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/samples/CAN_PSP/" \
        "CAN_PSPs_Hember-20180207T213138Z-001/CAN_PSPs_Hember/NAFP_L4_SL_ByJur_R16d_ForBrendanRogers1_lat52_ABoVE.shp"

    bounds = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/STUDY_AREA/ABoVE_Study_Domain_geo.shp"

    bounds_vec = Vector(bounds)
    bounds_geom = bounds_vec.features[0].GetGeometryRef()

    attr = {'ID_Plot': 'str',
            'Lat': 'float',
            'Lon': 'float'}

    samp_data = Handler(infilename).read_from_csv(return_dicts=True)

    wkt_list = list()
    attr_list = list()

    spref_str = '+proj=longlat +datum=WGS84'
    latlon = list()
    count = 0
    for row in samp_data:
        print('Reading elem: {}'.format(str(count + 1)))

        elem = dict()
        for header in list(attr):
            elem[header] = row[header]

        samp_geom = Vector.get_osgeo_geom(Vector.wkt_from_coords([row['Lon'], row['Lat']]))

        latlon.append([row['Lon'], row['Lat']])

        if elem['Lat'] < 52.0 and samp_geom.Intersects(bounds_geom):
            wkt_list.append(Vector.wkt_from_coords([row['Lon'], row['Lat']]))

            attr_list.append(elem)

        count += 1

    uniq, indices, inverse, count = np.unique(ar=latlon,
                                              axis=0,
                                              return_index=True,
                                              return_counts=True,
                                              return_inverse=True)

    print(uniq.shape)
    exit()
    vector = Vector.vector_from_string(wkt_list,
                                       spref_string=spref_str,
                                       spref_string_type='proj4',
                                       vector_type='point',
                                       attributes=attr_list,
                                       attribute_types=attr,
                                       verbose=True)


    print(vector)
    vector.write_vector(outfilename)




    '''
    file0 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/albedo_data_2000_2010_full_by_tc.csv"
    
    infile1 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/FIRE/ak_fire/FireAreaHistory_gt500ha.shp"
    infile2 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/FIRE/can_fire/NFDB_poly_20171106_gt500ha_ABoVE_geo.shp"


    vec1 = Vector(infile1)
    vec2 = Vector(infile2)

    print(vec1)
    print(vec2)
    
    keep_list = list()
    for i, wkt in vec2.wktlist:
        
    
    
    
    infilename = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/all_fire_NA.csv"
    outfilename = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/all_fire_NA.shp"

    attr = {'FIRE_ID': 'str',
            'SIZE_HA': 'float',
            'YEAR': 'float'}

    with open(infilename) as f:
        lines = f.readlines()
        header = list(elem.strip() for elem in lines[0].split(','))
        header[-1] = 'geom'
        header[1] = 'FIRE_ID'

        outlines = list()

        for line in lines[1:]:
            if "CAN_1989-L907,L939" in line:
                line = line.replace("CAN_1989-L907,L939","CAN_1989_L907_L939")
            content = line.split(',')
            geom = ','.join(elem.strip() for elem in content[4:]).replace('"','')
            geom = geom.replace('type','"type"').replace('coordinates','"coordinates"')

            if ('GeometryCollection' not in geom) and \
                    ('LineString' not in geom) and \
                    ('Point' not in geom):

                if 'MultiPolygon' in geom:
                    geom = geom.replace('MultiPolygon','"MultiPolygon"')
                else:
                    geom = geom.replace('Polygon', '"Polygon"')

                    json_str = json.loads(geom)
                    coords = [json_str['coordinates']]
                    json_str['coordinates'] = coords
                    geom = json.dumps(json_str)

                outlines.append(dict(zip(header, list(Handler.string_to_type(elem.strip()) for elem in content[:4]) + \
                                     [geom])))

    for line in outlines[0:5]:
        print(line)

    print(len(outlines))

    wkt_list = list()
    attr_list = list()

    count = 0
    for row in outlines[0:2]:
        # print('Reading elem: {}'.format(str(count + 1)))

        elem = dict()
        for header in list(attr):
            elem[header] = row[header]

        wkt_list.append(row['geom'])

        attr_list.append(elem)

        count += 1

    vector = Vector.vector_from_string(wkt_list,
                                       geom_string_type='json',
                                       out_epsg=4326,
                                       vector_type='polygon',
                                       attributes=attr_list,
                                       attribute_types=attr,
                                       verbose=False)

    print(vector)
    vector.write_vector(outfilename)






    infilename = "D:/Shared/Dropbox/projects/NAU/landsat_diva/data/DiVA_lat_long_09152019.csv"
    outfilename = "D:/Shared/Dropbox/projects/NAU/landsat_diva/data/DiVA_lat_long_09152019.shp"

    attr = {'site_id': 'str',
            'latitude': 'float',
            'longitude': 'float'}

    samp_data = Handler(infilename).read_from_csv(return_dicts=True)

    wkt_list = list()
    attr_list = list()

    spref_str = '+proj=longlat +datum=WGS84'

    count = 0
    for row in samp_data:
        print('Reading elem: {}'.format(str(count + 1)))

        elem = dict()
        for header in list(attr):
            elem[header] = row[header]

        wkt_list.append(Vector.wkt_from_coords([row['longitude'], row['latitude']]))

        attr_list.append(elem)

        count += 1

    vector = Vector.vector_from_string(wkt_list,
                                       spref_string=spref_str,
                                       spref_string_type='proj4',
                                       vector_type='point',
                                       attributes=attr_list,
                                       attribute_types=attr,
                                       verbose=True)


    print(vector)
    vector.write_vector(outfilename)




    
    folder = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/gee_extract/"
    
    
    csvfile = folder + "gee_data_cleaning_v28_median_formatted.csv"
    outfile = folder + "gee_data_cleaning_v28_median_formatted.shp"

    samp_data = pd.read_csv(csvfile)
    headers = list(samp_data)

    print(headers)

    spref_str = '+proj=longlat +datum=WGS84'

    wkt_list = list()
    attr_list = list()

    attribute_types = { 'img_jday_3': 'int',
                        'img_jday_2': 'int',
                        'img_jday_1': 'int',
                        'decid_frac': 'float',
                        'slope': 'float',
                        'swir1_2': 'float',
                        'blue_3': 'float',
                        'swir1_1': 'float',
                        'site': 'str',
                        'ndvi_3': 'float',
                        'swir2_1': 'float',
                        'site_year': 'int',
                        'swir2_2': 'float',
                        'nir_3': 'float',
                        'elevation': 'float',
                        'nir_2': 'float',
                        'nir_1': 'float',
                        'savi_1': 'float',
                        'savi_2': 'float',
                        'savi_3': 'float',
                        'sensor_3': 'str',
                        'sensor_2': 'str',
                        'sensor_1': 'str',
                        'id': 'int',
                        'vari_3': 'float',
                        'vari_2': 'float',
                        'vari_1': 'float',
                        'nbr_2': 'float',
                        'nbr_3': 'float',
                        'aspect': 'float',
                        'ndwi_1': 'float',
                        'img_year_1': 'int',
                        'img_year_2': 'int',
                        'img_year_3': 'int',
                        'nbr_1': 'float',
                        'red_1': 'float',
                        'red_3': 'float',
                        'red_2': 'float',
                        'green_1': 'float',
                        'ndwi_3': 'float',
                        'green_3': 'float',
                        'ndwi_2': 'float',
                        'green_2': 'float',
                        'swir1_3': 'float',
                        'ndvi_2': 'float',
                        'blue_2': 'float',
                        'blue_1': 'float',
                        'ndvi_1': 'float',
                        'swir2_3': 'float'}
    count = 0
    for _, row in samp_data.iterrows():
        print('Reading elem: {}'.format(str(count + 1)))

        elem = dict()
        for header in list(attribute_types):
            elem[header] = row[header]

        wkt_list.append(row['geom'])

        attr_list.append(elem)

        count += 1

    vector = Vector.vector_from_string(wkt_list,
                                       spref_string=spref_str,
                                       spref_string_type='proj4',
                                       vector_type='point',
                                       attributes=attr_list,
                                       attribute_types=attribute_types,
                                       verbose=True)

    print(vector)

    vector.write_vector(outfile)
    

    outfile = "d:/shared/Dropbox/projects/NAU/landsat_deciduous/data/east_bounds.shp"

    # boundary of the region
    boreal_coords = [[[-170.170703125, 50.55860413479424],
          [-158.217578125, 52.68617167041279],
          [-153.30939285434152, 55.911669002771404],
          [-150.04375, 58.80503543579855],
          [-139.233203125, 58.068985987821954],
          [-136.15703125, 54.714910470066776],
          [-132.28984375, 47.267822642617666],
          [-131.4109375, 41.59465704101877],
          [-123.9066476728093, 41.3211582877145],
          [-116.25939622443843, 41.46034634091618],
          [-108.55937499999999, 41.36419260424219],
          [-102.89042968749999, 41.397166266344954],
          [-96.21074218749999, 41.26517132419739],
          [-90.49785156249999, 41.23213080790694],
          [-84.25761718749999, 41.23213080790694],
          [-77.05058593749999, 41.29819513068916],
          [-68.12968749999999, 41.26517132419742],
          [-56.70390624999999, 43.7589530883319],
          [-50.72734374999999, 47.38696854291394],
          [-52.92460937499999, 52.79259685897027],
          [-57.84648437499999, 56.64698491697063],
          [-64.87773437499999, 61.5146781712937],
          [-73.96213049438268, 62.80048928805028],
          [-78.32093919613442, 62.78853877630148],
          [-80.52226562499999, 59.74792657078106],
          [-84.9374198886535, 59.27782159718193],
          [-88.37276592344836, 60.37639431989281],
          [-94.32109374999999, 64.65432185309979],
          [-110.84453124999999, 69.12528180058297],
          [-127.45585937499999, 70.69439418030035],
          [-145.209765625, 71.54742032805514],
          [-151.99237509694262, 71.7394004113125],
          [-158.920703125, 71.46378059608598],
          [-163.9438033714605, 70.33273202428589],
          [-167.358203125, 68.74622173946399],
          [-168.588671875, 67.76966607015038],
          [-168.764453125, 66.43638686378715],
          [-168.54359182813454, 65.57009532838762],
          [-169.643359375, 64.74821921769586],
          [-172.1921875, 63.91066619210742],
          [-177.66337890625, 60.31862431477453],
          [-180.0, 60.31862431477453],
          [-180.0, 50.38214994852445],
          [-178.3430865436669, 50.38214994852445],
          [-170.170703125, 50.55860413479424]],

          [[180.0, 60.31862431477453],
          [178.16636398271964, 57.45355647434897],
          [171.06398672131763, 52.978982153380194],
          [174.72537102620095, 51.303685681238484],
          [180.0, 50.38214994852445],
          [180.0, 60.31862431477453]]]

    east_coords = [[-93.14973698647538, 59.61676873926032],
          [-102.37825261147538, 53.71504505488069],
          [-102.77376042397538, 41.301085122242654],
          [-91.83137761147538, 41.26806277858752],
          [-76.62629948647538, 40.7707238844176],
          [-57.773760423975375, 41.00328268461561],
          [-49.116533861475375, 40.93692071714397],
          [-50.390947923975375, 55.89268713712933],
          [-51.050127611475375, 60.58059989979749],
          [-59.443682298975375, 60.211550691425074],
          [-67.17805729897538, 61.516512793447774],
          [-74.64876042397538, 62.86927694581044],
          [-80.31770573647538, 62.74879024515305],
          [-93.14973698647538, 59.61676873926032]]

    above_wkt = Vector.wkt_from_coords(east_coords,
                                       geom_type='polygon')
    
    above_attr = {'name': 'str'}

    vec = Vector(filename=outfile,
                 epsg=4326,
                 in_memory=True,
                 attr_def=above_attr,
                 geom_type='polygon',
                 primary_key=None)

    vec.wktlist.append(above_wkt)
    # vec.attributes.append({'name': 'NA_boreal_bounds'})
    vec.attributes.append({'name': 'east_bounds'})

    vec.write_vector()

    
    east_wkt = Vector.wkt_from_coords(east_coords, geom_type='polygon')

    east_geom = Vector.get_osgeo_geom(east_wkt)


    vec = Vector("D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/STUDY_AREA/all_tiles.shp")

    file_list = list()

    for i, wkt in enumerate(vec.wktlist):
        if east_geom.Intersects(Vector.get_osgeo_geom(wkt)):
            print('True')
            file_list.append(vec.attributes[i]['filename'])

    for filename in file_list:
        print(filename)

    outfile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/STUDY_AREA/east_tiles.txt"

    Handler(outfile).write_list_to_file(file_list)



    file0 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/albedo_data_2000_2010_full_by_tc.csv"
    outfile0 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/albedo_data/albedo_data_2000_2010_full_by_tc.shp"
    dicts = Handler(file0).read_from_csv(return_dicts=True)
    print(len(dicts))

    for dict_ in dicts[0:10]:
        print(dict_)

    coord_list = list([elem['x'], elem['y']] for elem in dicts)

    print(len(coord_list))

    uniq_coords = np.unique(coord_list, axis=0).tolist()

    print(len(uniq_coords))

    for elem in uniq_coords[0:10]:
        print(elem)

    attr = {'site': 'long'}

    vec = Vector(filename=outfile0,
                 epsg=4326,
                 in_memory=True,
                 attr_def=attr,
                 geom_type='point',
                 primary_key=None)

    vec.wktlist = list(Vector.wkt_from_coords(pt) for pt in uniq_coords)

    vec.attributes = list({'site':i} for i in range(len(uniq_coords)))

    for elem in vec.wktlist[0:10]:
        print(elem)

    vec.write_vector()


    '''


