from modules import *
import multiprocessing as mp
from osgeo import gdal_array
import numpy as np
from sys import argv


def _tile_process_(args):
    _tie_pt, _tile_coords, _tile_arr, _nodatavalue, _composite_type = args
    _x, _y, _cols, _rows = _tile_coords

    if _composite_type == 'mean':
        _temp_arr = np.apply_along_axis(lambda x: np.mean(x[x != _nodatavalue]) if (x[x != _nodatavalue]).shape[0] > 0
        else _nodatavalue, 0, _tile_arr)
    elif _composite_type == 'median':
        _temp_arr = np.apply_along_axis(lambda x: np.median(x[x != _nodatavalue]) if (x[x != _nodatavalue]).shape[0] > 0
        else _nodatavalue, 0, _tile_arr)
    elif 'pctl' in _composite_type:
        pctl = int(_composite_type.split('_')[1])
        _temp_arr = np.apply_along_axis(
            lambda x: np.percentile(x[x != _nodatavalue], pctl) if (x[x != _nodatavalue]).shape[0] > 0
            else _nodatavalue, 0, _tile_arr)
    else:
        _temp_arr = None

    return _y, (_y + _rows), _x, (_x + _cols), _temp_arr


def _get_tile_data_(_layerstack_vrt, _band_order, _tile_size, _composite_type):
    _lras = Raster('_tmp_layerstack')
    _lras.datasource = _layerstack_vrt
    _lras.initialize()
    _lras.make_tile_grid(*_tile_size)

    tile_count = 0
    for _tie_pt, _tile_arr in _lras.get_next_tile(bands=_band_order):
        _tile_coords = _lras.tile_grid[tile_count]['block_coords']

        yield _tie_pt, _tile_coords, _tile_arr, _lras.nodatavalue, _composite_type
        tile_count += 1


if __name__ == '__main__':

    script, file_folder, outdir, nthreads = argv

    seasons = [(60, 120),  # spring
               (180, 240),  # summer
               (255, 315),  # fall
               (330, 45)]  # winter

    years = [(2000, 2002),
             (2002, 2007),
             (2008, 2012),
             (2013, 2018)]

    composite_type = 'median'
    version = 1
    thread_offset = 1
    tile_size = (1024, 1024)

    nthreads = int(nthreads)

    pool = mp.Pool(processes=nthreads)

    all_files = Handler(dirname=file_folder).find_all('*_albedo.tif')

    Opt.cprint(seasons)
    Opt.cprint(years)
    Opt.cprint(len(all_files))

    num_list = np.array(list(list(int(elem_) for elem_
                                  in Handler(elem).basename.replace('_albedo.tif', '').replace('bluesky_albedo_',
                                                                                               '').split('_'))
                             for elem in all_files))

    for season_start, season_end in seasons:

        for year_start, year_end in years:

            outfile = outdir + '/albedo_composite_{}_{}_{}_{}_{}_v{}.tif'.format(composite_type,
                                                                                 str(year_start),
                                                                                 str(year_end),
                                                                                 str(season_start),
                                                                                 str(season_end),
                                                                                 str(version))

            file_loc_on_list = np.where((num_list[:, 0] >= year_start) & (num_list[:, 0] <= year_end) &
                                        (num_list[:, 1] >= season_start) & (num_list[:, 1] <= season_end))[0]

            filelist = list(all_files[i] for i in file_loc_on_list.tolist())

            for file_ in filelist:
                Opt.cprint(file_)

            Opt.cprint(outfile)

            mraster = MultiRaster(filelist=filelist)

            Opt.cprint(mraster)

            ls_vrt = mraster.layerstack(return_vrt=True)
            lras = Raster('tmp_layerstack')
            lras.datasource = ls_vrt
            lras.initialize()

            lras.shape[0] = 1

            Opt.cprint(lras)

            band_order = list(range(1, len(mraster.rasters) + 1))

            out_arr = np.zeros((lras.shape[1], lras.shape[2]),
                               dtype=gdal_array.GDALTypeCodeToNumericTypeCode(lras.dtype))

            result_count = 0
            for result in pool.imap_unordered(_tile_process_,
                                              _get_tile_data_(ls_vrt, band_order, tile_size, composite_type)):
                y_s, y_e, x_s, x_e, temp_arr = result
                Opt.cprint((result_count + 1, y_s, y_e, x_s, x_e))
                out_arr[y_s:y_e, x_s:x_e] = temp_arr
                result_count += 1

            lras.array = out_arr
            lras.write_to_file(outfile)

            Opt.cprint('Written {}'.format(outfile))

