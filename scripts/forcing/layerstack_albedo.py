from modules import *
import multiprocessing as mp
from osgeo import gdal_array
import numpy as np
from sys import argv


def _tile_process_(args):
    _filelist, _outfile, _band_order, _tile_dict, _composite_type = args
    _tile_coords = _tile_dict['block_coords']
    _xmin, _ymin = _tile_dict['first_pixel']

    _x, _y, _cols, _rows = _tile_coords

    _outfile = '/vsimem/' + Handler(_outfile).basename.split('.tif')[0] + \
               '_{}.tif'.format('_'.join([str(j) for j in _tile_coords]))

    _mraster = MultiRaster(_filelist)
    _layerstack_vrt = _mraster.layerstack(return_vrt=True, outfile=_outfile)

    _lras = Raster('_tmp_layerstack')
    _lras.datasource = _layerstack_vrt
    _lras.initialize()
    _tile_arr = _lras.get_tile(_band_order, _tile_coords).copy()

    if _composite_type == 'mean':
        _temp_arr = np.apply_along_axis(lambda x: np.mean(x[x != _lras.nodatavalue])
                                        if (x[x != _lras.nodatavalue]).shape[0] > 0
                                        else _lras.nodatavalue, 0, _tile_arr)
    elif _composite_type == 'median':
        _temp_arr = np.apply_along_axis(lambda x: np.median(x[x != _lras.nodatavalue])
                                        if (x[x != _lras.nodatavalue]).shape[0] > 0
                                        else _lras.nodatavalue, 0, _tile_arr)
    elif _composite_type == 'max':
        _temp_arr = np.apply_along_axis(lambda x: np.max(x[x != _lras.nodatavalue])
        if (x[x != _lras.nodatavalue]).shape[0] > 0
        else _lras.nodatavalue, 0, _tile_arr)

    elif _composite_type == 'min':
        _temp_arr = np.apply_along_axis(lambda x: np.min(x[x != _lras.nodatavalue])
        if (x[x != _lras.nodatavalue]).shape[0] > 0
        else _lras.nodatavalue, 0, _tile_arr)

    elif 'pctl' in _composite_type:
        pctl = int(_composite_type.split('_')[1])
        _temp_arr = np.apply_along_axis(lambda x: np.percentile(x[x != _lras.nodatavalue], pctl)
                                        if (x[x != _lras.nodatavalue]).shape[0] > 0
                                        else _lras.nodatavalue, 0, _tile_arr)
    else:
        _temp_arr = None

    _lras = None
    _mraster = None
    Handler(_outfile).file_delete()
    _tile_arr = None

    return (_y-_ymin), ((_y-_ymin) + _rows), (_x-_xmin), ((_x-_xmin) + _cols), _temp_arr


def _get_tile_data_(_filelist, _outfile, _band_order, _tile_specs, _composite_type):
    _vrtfile = '/vsimem/' + Handler(_outfile).basename.split('.tif')[0] + '_tmp_layerstack.vrt'
    _outfile = '/vsimem/' + Handler(_outfile).basename.split('.tif')[0] + '_tmp_layerstack.tif'

    _mraster = MultiRaster(_filelist)
    _layerstack_vrt = _mraster.layerstack(return_vrt=True, outfile=_outfile)

    _lras = Raster('_tmp_layerstack')
    _lras.datasource = _layerstack_vrt
    _lras.initialize()
    _lras.make_tile_grid(*_tile_specs)

    for _ii in range(_lras.ntiles):
        yield _filelist, _outfile, _band_order, _lras.tile_grid[_ii], _composite_type


if __name__ == '__main__':
    '''
    nthreads = 4
    file_folder = 'C:/temp/albedo'
    seasons=[(300,360)]
    years = [(1998,2005)]
    outdir = 'C:/temp/albedo'
    all_files = Handler(dirname=file_folder).find_all('*_quality.tif')
    num_list = np.array(list(list(int(elem_) for elem_
                                  in Handler(elem).basename.replace('_quality.tif', '').replace('bluesky_albedo_',
                                                                                               '').split('_'))
                             for elem in all_files))
    seasons = [(60, 120),  # spring
               (180, 240),  # summer
               (255, 315),  # fall
               (330, 45)]  # winter

    years = [(2000, 2002),
             (2002, 2007),
             (2008, 2012),
             (2013, 2018)]
    
    version = 1
    thread_offset = 0
    tile_size = (1024, 1024)

    image_bounds = (-130.999, -90.0, 40.0, 50.0)  # xmin, xmax, ymin, ymax
    '''
    script, file_folder, outdir, startyear, endyear, startdate, enddate, reducer, ver, nthreads = argv

    tile_size = (1024, 1024)
    image_bounds = (-179.999, -50.0, 30.0, 75.0)  # xmin, xmax, ymin, ymax

    startyear = int(startyear)
    endyear = int(endyear)
    startdate = int(startdate)
    enddate = int(enddate)
    nthreads = int(nthreads)-1

    all_files = Handler(dirname=file_folder).find_all('*_albedo.tif')
    num_list = np.array(list(list(int(elem_) for elem_
                                  in Handler(elem).basename.replace('_albedo.tif', '').replace('bluesky_albedo_',
                                                                                               '').split('_'))
                             for elem in all_files))

    tile_specs = (tile_size[0], tile_size[1], image_bounds, 'crs')

    pool = mp.Pool(processes=nthreads)

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

    lras.transform = (image_bounds[0],
                      lras.transform[1],
                      lras.transform[2],
                      image_bounds[3],
                      lras.transform[4],
                      lras.transform[5])

    lras.shape = [1, (ymax-ymin), (xmax-xmin)]

    lras.make_tile_grid(*tile_specs)
    print(len(lras.tile_grid))

    Opt.cprint(lras)

    band_order = list(range(1, len(mraster.rasters) + 1))

    out_arr = np.zeros((lras.shape[1], lras.shape[2]),
                       dtype=gdal_array.GDALTypeCodeToNumericTypeCode(lras.dtype)) + \
        lras.nodatavalue

    result_count = 0
    for result in pool.map(_tile_process_, _get_tile_data_(filelist,
                                                           outfile,
                                                           band_order,
                                                           tile_specs,
                                                           reducer)):
        y_s, y_e, x_s, x_e, temp_arr = result

        Opt.cprint((result_count + 1, y_s, y_e, x_s, x_e))

        if temp_arr is not None:
            out_arr[y_s:y_e, x_s:x_e] = temp_arr

        result_count += 1

    lras.array = out_arr
    lras.write_to_file(outfile)

    Opt.cprint('Written {}'.format(outfile))

