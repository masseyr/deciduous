from geosoup import Vector
import numpy as np


if __name__ == '__main__':

    # file1 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/samples/all_samp_pre_v1.shp"
    file1 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/samples/CAN_PSP/PSP_data/CAN_PSP_v1.shp"
    file2 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/STUDY_AREA/ABoVE_Study_Domain_geo.shp"

    lat_x = 52.0

    vec = Vector(file1)
    vec2 = Vector(file2)

    geom2 = vec2.features[0].GetGeometryRef()

    lonlat = []

    for i in range(vec.nfeat):
        geom = vec.features[i].GetGeometryRef()

        pt = geom.GetPoint()

        lonlat.append(pt[0:2])

    print(len(lonlat))


    uniq, indices, inverse, count = np.unique(ar=lonlat,axis=0,
                                              return_index=True,
                                              return_counts=True,
                                              return_inverse=True)

    print(uniq.shape)


    loc = np.where(uniq[:, 1] < lat_x)

    print(uniq[loc])

    print(uniq[loc].shape)

    coords = uniq[loc].tolist()

    geoms = [Vector.get_osgeo_geom('POINT ({} {})'.format(str(coord[0]), str(coord[1]))) for coord in coords]

    outlist = []
    for geom in geoms:
        if geom.Intersects(geom2):
            outlist.append(geom)

    print(len(outlist))
