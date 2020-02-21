from modules import *
from sys import argv

if __name__ == '__main__':

    script, filename, cutfile, outfolder = argv

    # epsg = 3571
    out_proj4 = '+proj=laea +lat_0=90 +lon_0=180 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs '

    outfile = outfolder + Handler(Handler(filename).add_to_filename('_NPLAEA_clipped')).basename

    ras = Raster(filename)
    ras.initialize()

    print(ras)

    ras.reproject(outfile=outfile,
                  out_proj4=out_proj4,
                  output_res=(100, 100),
                  out_nodatavalue=0.0,
                  bigtiff='yes',
                  cutline_file=cutfile,
                  compress='lzw')

    Raster(outfile).add_overviews(bigtiff='yes',
                                  compress='lzw')






