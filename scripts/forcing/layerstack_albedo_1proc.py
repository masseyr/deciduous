from modules import *
import multiprocessing as mp
from osgeo import gdal_array
import numpy as np
from sys import argv


if __name__ == '__main__':

    script, file_folder, outdir, startyear, endyear, startdate, enddate, reducer, ver = argv

    tile_size = (1024, 1024)
    image_bounds = (-179.999, -50.0, 30.0, 75.0)  # xmin, xmax, ymin, ymax

    startyear = int(startyear)
    endyear = int(endyear)
    startdate = int(startdate)
    enddate = int(enddate)

    all_files = Handler(dirname=file_folder).find_all('*_albedo.tif')
    num_list = np.array(list(list(int(elem_) for elem_
                                  in Handler(elem).basename.replace('_albedo.tif', '').replace('bluesky_albedo_',
                                                                                               '').split('_'))
                             for elem in all_files))

    tile_specs = (tile_size[0], tile_size[1], image_bounds, 'crs')

    Opt.cprint((startdate, enddate))
    Opt.cprint((startyear, endyear))
    Opt.cprint(len(all_files))

    outfile = outdir + '/albedo_composite_{}_{}_{}_{}_{}_v{}.tif'.format(reducer,
                                                                         str(startyear),
                                                                         str(endyear),
                                                                         str(startdate),
                                                                         str(enddate),
                                                                         str(ver))

    file_loc_on_list = np.where((num_list[:, 0] >= startyear) & (num_list[:, 0] <= endyear) &
                                (num_list[:, 1] >= startdate) & (num_list[:, 1] <= enddate))[0]

    filelist = list(all_files[i] for i in file_loc_on_list.tolist())

    for file_ in filelist:
        Opt.cprint(file_)

    Opt.cprint(outfile)

    mraster = MultiRaster(filelist=filelist)

    Opt.cprint(mraster)

    ls_vrt = mraster.layerstack(return_vrt=True, outfile=outfile)

    lras = Raster('tmp_layerstack')
    lras.datasource = ls_vrt
    lras.initialize()

    xmin, xmax, ymin, ymax = lras.get_pixel_bounds(image_bounds, 'crs')

    lras.shape = [1, (ymax - ymin), (xmax - xmin)]

    lras.make_tile_grid(*tile_specs)

    Opt.cprint(len(lras.tile_grid))
    Opt.cprint(lras)

    band_order = list(range(1, len(mraster.rasters) + 1))

    out_arr = np.zeros((lras.shape[1], lras.shape[2]),
                       dtype=gdal_array.GDALTypeCodeToNumericTypeCode(lras.dtype)) + lras.nodatavalue

    Opt.cprint('Input transform: {}'.format(str(lras.transform)),
               newline='\n\n')

    # output transform
    lras.transform = (image_bounds[0],
                      lras.transform[1],
                      lras.transform[2],
                      image_bounds[3],
                      lras.transform[4],
                      lras.transform[5])

    Opt.cprint('output transform: {}'.format(str(lras.transform)),
               newline='\n\n')

    result_count = 0

    for tile_dict in lras.tile_grid:

        tile_coords = tile_dict['block_coords']
        _x, _y, _cols, _rows = tile_coords
        _xmin, _ymin = tile_dict['first_pixel']

        tile_arr = lras.get_tile(band_order, tile_coords).copy()

        if reducer == 'mean':
            temp_arr = np.apply_along_axis(lambda x: np.mean(x[x != lras.nodatavalue])
                                           if (x[x != lras.nodatavalue]).shape[0] > 0
                                           else lras.nodatavalue, 0, tile_arr)

        elif reducer == 'median':
            temp_arr = np.apply_along_axis(lambda x: np.median(x[x != lras.nodatavalue])
                                           if (x[x != lras.nodatavalue]).shape[0] > 0
                                           else lras.nodatavalue, 0, tile_arr)

        elif reducer == 'max':
            temp_arr = np.apply_along_axis(lambda x: np.max(x[x != lras.nodatavalue])
                                           if (x[x != lras.nodatavalue]).shape[0] > 0
                                           else lras.nodatavalue, 0, tile_arr)

        elif reducer == 'min':
            temp_arr = np.apply_along_axis(lambda x: np.min(x[x != lras.nodatavalue])
                                           if (x[x != lras.nodatavalue]).shape[0] > 0
                                           else lras.nodatavalue, 0, tile_arr)

        elif 'pctl' in reducer:
            pctl = int(reducer.split('_')[1])
            temp_arr = np.apply_along_axis(lambda x: np.percentile(x[x != lras.nodatavalue], pctl)
                                           if (x[x != lras.nodatavalue]).shape[0] > 0
                                           else lras.nodatavalue, 0, tile_arr)

        else:
            temp_arr = None

        y_s, y_e, x_s, x_e = (_y-_ymin), ((_y-_ymin) + _rows), (_x-_xmin), ((_x-_xmin) + _cols)

        Opt.cprint((result_count + 1, tile_dict['tie_point'], tile_dict['block_coords'], y_s, y_e, x_s, x_e))

        if temp_arr is not None:
            out_arr[y_s:y_e, x_s:x_e] = temp_arr

        result_count += 1

    lras.array = out_arr
    lras.write_to_file(outfile)

    Opt.cprint('Written {}'.format(outfile))

