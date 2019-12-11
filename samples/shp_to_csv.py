from modules import *

if __name__ == '__main__':

    infile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/BNZ_LTER/mack_data_transects_ba_with_age_pts.shp"
    outfile = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/SAMPLES/BNZ_LTER/mack_data_transects_ba_with_age_pts_.csv"

    vec = Vector(infile)

    print(vec)

    out_list = vec.attributes

    for i, wkt in enumerate(vec.wktlist):

        out_list[i]['Longitude'] = wkt.split(' ')[1].replace('(', '')

        out_list[i]['Latitude'] = wkt.split(' ')[2].replace(')', '')

    print(outfile)

    Handler(outfile).write_to_csv(out_list,outfile)

    print('Done!')







