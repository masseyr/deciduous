from netCDF4 import Dataset
import numpy as np
from geosoup import Raster, Handler, GDAL_FIELD_DEF
from osgeo import osr


if __name__ == '__main__':

    file1 = "D:/Shared/Dropbox/projects/NAU/landsat_deciduous/data/kernels/CAM5/alb.kernel.nc"

    albedo = Dataset(file1)

    # list month names
    months = ['JANUARY',
              'FEBRUARY',
              'MARCH',
              'APRIL',
              'MAY',
              'JUNE',
              'JULY',
              'AUGUST',
              'SEPEMBER',
              'OCTOBER',
              'NOVEMBER',
              'DECEMBER']

    # list names of the variables in the kernel
    var_names = {'FSNS': 'SURFACE_ALL_SKY',
                 'FSNSC': 'SURFACE_CLEAR_SKY',
                 'FSNT': 'TOA_ALL_SKY',
                 'FSNTC': 'TOA_CLEAR_SKY'}

    # make a grid of lat lon
    lon, lat = np.meshgrid(np.array(albedo.variables['lon']),
                           np.array(albedo.variables['lat']))

    # this dataset starts at 0 deg lon, extract slice position for start at -180
    cut_loc = np.min(np.where(lon[0] >= 180.0)[0])
    lon = np.hstack([lon[:, cut_loc:] - 360.0, lon[:, :cut_loc]])
    lat = np.vstack([lat[(lat.shape[0] - indx - 1), :] for indx in range(lat.shape[0])])

    # pixel sizes
    pixely = lat[1, 0] - lat[0, 0]
    pixelx = lon[0, 1] - lon[0, 0]

    # image coordinates
    transform = [lon[0, 0], pixelx, 0, lat[0, 0], 0, pixely]

    # spatial reference for geographic lat/lon
    spref = osr.SpatialReference()
    spref.ImportFromEPSG(4326)

    # extract variables one by one
    for variable, data in albedo.variables.items():

        if variable in var_names:

            print('Processing {} kernel'.format(var_names[variable]))

            # for attr in dir(data):
            #    print('{} - {}'.format(str(attr), str(getattr(data, attr, None))))

            arr = np.array(data)

            resliced_arr_list = []
            for channel_indx in range(arr.shape[0]):

                # first slice west of 0 deg lon
                first_slice = np.vstack([arr[channel_indx, (arr.shape[1] - row_indx - 1), cut_loc:]
                                         for row_indx in range(arr.shape[1])])

                # second slice east of 0 deg lon
                second_slice = np.vstack([arr[channel_indx, (arr.shape[1] - row_indx - 1), :cut_loc]
                                         for row_indx in range(arr.shape[1])])

                resliced_arr_list.append(np.hstack([first_slice, second_slice]))

            # stack all months
            arr = np.stack(resliced_arr_list, 0)

            # name output file
            outfile = Handler(file1).dirname + Handler().sep + 'ALBEDO_CAM5_{}_KERNEL.tif'.format(var_names[variable])

            # define raster object
            ras = Raster(outfile,
                         array=arr,
                         bnames=months,
                         dtype=GDAL_FIELD_DEF['double'],
                         shape=arr.shape,
                         transform=transform,
                         crs_string=spref.ExportToWkt())

            # define no data value
            ras.nodatavalue = data._FillValue

            # write raster object
            ras.write_to_file()
