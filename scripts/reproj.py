from modules import *
from osgeo import gdal
from sys import argv

if __name__ == '__main__':

    script, basefile, infile = argv

    fptr = gdal.Open(basefile)

    out_crs = fptr.GetProjection()

    Opt.cprint('Out Proj: ' + out_crs)

    fptr = None
    Opt.cprint('Reprojecting...')
    out_ras = Raster(infile).reproject(out_crs)

    Opt.cprint(out_ras)

