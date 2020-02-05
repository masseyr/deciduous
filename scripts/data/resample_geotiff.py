from modules import *
from sys import argv


if __name__ == '__main__':
    script, infile, out_res = argv

    out_proj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84'

    outfile = Handler(infile).add_to_filename('_resampled_{}'.format(str(out_res).replace('.', '')))

    ras = Raster(infile)
    ras.initialize()

    Opt.cprint(ras)

    ras.reproject(outfile=outfile,
                  out_proj4=out_proj4,
                  verbose=True,
                  resampling='cubic',
                  output_res=(float(out_res), float(out_res)),
                  out_nodatavalue=0.0,
                  bigtiff='yes',
                  compress='lzw')

    Raster(outfile).add_overviews(bigtiff='yes',
                                  compress='lzw')

    Opt.cprint('Written file: {}'.format(outfile))


