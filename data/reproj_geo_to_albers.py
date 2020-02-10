from modules import *


if __name__ == '__main__':

    folder = 'c:/temp/'
    filelist = ['decid_diff_incl_east_2000_2015.tif'
                'decid_diff_2000_2015.tif',
                'tc_diff_2000_2015.tif',
                'spr_forc_2000_2015.tif',
                'sum_forc_2000_2015.tif',
                'fall_forc_2000_2015.tif',
                ]

    out_proj4 = '+proj=aea +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'

    for infile in filelist:

        ras = Raster(folder + infile)
        ras.initialize()

        print(ras)

        outfile = folder + Handler(infile).add_to_filename('_albers')

        ras.reproject(outfile=outfile,
                      out_proj4=out_proj4,
                      verbose=True,
                      output_res=(250, 250),
                      out_nodatavalue=0.0,
                      bigtiff='yes',
                      compress='lzw')

        Raster(outfile).add_overviews(bigtiff='yes',
                                      compress='lzw')
        





