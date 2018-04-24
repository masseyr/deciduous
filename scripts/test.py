from modules import *
import seaborn as sns
from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties


if __name__ == '__main__':

    """

    infile = "C:\\Users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\Alaska_all_2010_sampV2.csv"
    plotfile = "C:\\Users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\heatmap_var_1.png"

    plot_heatmap = {
        'type': 'heatmap',
        'datafile': infile,
        'title': 'Correlation among variables',
        'plotfile': plotfile,
        'show_values': True,
        'heat_range': [0.0, 1.0],
        'color_str': "YlGnBu"
    }

    print(plot_heatmap)

    heatmap = Plot(plot_heatmap)
    heatmap.draw()

    print(data_mat.shape)
    corr = np.zeros([ncols, ncols], dtype=float)

    for i in range(0, ncols):
        for j in range(0, ncols):
            corr[i, j] = np.abs(pearsonr(Sublist.column(data_mat, i),
                                  Sublist.column(data_mat, j))[0])

    print(corr)

    nir = Sublist.column(data_mat, 14)
    ndvi = Sublist.column(data_mat, 8)

    print(nir)
    print(ndvi)
    print(Sublist.column(data_mat, 7))

    print('******')

    print(pearsonr(nir, ndvi)[0])
    print(np.corrcoef(nir, ndvi)[0,1])
    print('******')

    cov_mat = np.cov(data_mat, rowvar=False,)

    var_names = list(bname_dict[elem] for elem in trn_samp.x_name)

    print(data_mat)
    print(trn_samp.x_name)
    print(var_names)
    print(cov_mat[14,8])

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=0)] = True

    xticklabels = var_names[:-1] + ['']
    yticklabels = [''] + var_names[1:]

    font = FontProperties()
    font.set_family('serif')
    font.set_style('normal')
    font.set_variant('normal')
    font.set_weight('bold')
    font.set_size('small')

    sns.set(font=font.get_fontconfig_pattern())
    sns.set(font_scale=2)

    with sns.axes_style("white"):
        ax = sns.heatmap(corr,  mask=mask, xticklabels=xticklabels,
                         yticklabels=yticklabels, annot=False, annot_kws={"size": 15},
                         vmax=1.0, vmin=0.0, square=True, cmap="YlGnBu")
        plt.title('Correlation among variables')
        plt.show()

    




    plot_rmse = {
        'type': 'histogram',  # plot types: histogram, surface, relative, regression
        'data': rmse_data,  # list or 1d array
        'xtitle': 'Random Forest model RMSE',  # title of x axis
        'ytitle': 'Frequency',  # title of y axis
        'title': 'Model RMSE plot (n = 2000)',  # plot title
        'filename': "C:\\Users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\rmse_plot_t.png",  # output file name
    }

    plot_rsq = {
        'type': 'histogram',  # plot types: histogram, surface, relative, regression
        'data': rsq_data,  # list or 1d array
        'xtitle': 'Random Forest model Pearson R-squared',  # title of x axis
        'ytitle': 'Frequency',  # title of y axis
        'title': 'Model R-squared plot (n = 2000)',  # plot title
        'filename': "C:\\Users\\rm885\\Dropbox\\projects\\NAU\\landsat_deciduous\\data\\rsq_plot_t.png",  # output file name
        'color': 'purple'
    }

    


    rsq = Plot(plot_rsq)
    print(rsq)
    rsq.draw()

    rmse = Plot(plot_rmse)
    print(rmse)
    rmse.draw()

    


    def reproject(self,
                  out_epsg=4326,
                  out_pixel=0.00027):
        \"""
        Reproject a raster based on EPSG codes
        :param out_epsg: EPSG SRID code for output raster
        :param out_pixel: Output pixel size
        \"""

        # get outfile name
        components = os.path.basename(self.fullpath_on_disk).split('.')
        if len(components) >= 2:
            outfile = os.path.basename(self.fullpath_on_disk) + os.path.sep + ''.join(components[0:-1]) + \
                      '_reproj' + str(out_epsg) + components[-1]
            name = ''.join(components[0:-1]) + '_reproj' + str(out_epsg) + components[-1]
            datasetid = ''.join(components[0:-1]) + '_reproj' + str(out_epsg)
        else:
            outfile = os.path.basename(self.fullpath_on_disk) + os.path.sep + components[0] + '_reproj' + str(out_epsg)
            name = components[0] + '_reproj' + str(out_epsg)
            datasetid = components[0] + '_reproj' + str(out_epsg)

        # define target spatial ref
        spref = osr.SpatialReference()
        spref.ImportFromEPSG(out_epsg)
        spref_wkt = spref.ExportToWkt()

        # open file
        fileptr = gdal.Open(self.fullpath_on_disk)
        in_transform = fileptr.GetGeoTransform()
        fileref_wkt = fileptr.GetProjectionRef()
        in_spref = osr.SpatialReference()
        in_spref.ImportFromWkt(fileref_wkt)

        # resampling parameters
        error_threshold = 0.125

        # resampling types: GRA_NearestNeighbour,GRA_Bilinear,GRA_Cubic,GRA_Lanczos,GRA_Average,GRA_Mode
        resampling = gdal.GRA_NearestNeighbour

        # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
        tempfileptr = gdal.AutoCreateWarpedVRT(fileptr,
                                               fileref_wkt,  # source wkt
                                               spref_wkt,  # destination wkt
                                               resampling,  # re-sampling type
                                               error_threshold)

        # out tie point
        out_tie_pt = Vector.reproject_point([in_transform[0], in_transform[3]],
                                            in_spatial_ref=in_spref,
                                            out_spatial_ref=spref)

        # set transform
        temp_transform = tempfileptr.GetGeoTransform()
        out_transform = [out_tie_pt[0],
                         out_pixel,
                         temp_transform[2],
                         out_tie_pt[1],
                         temp_transform[4],
                         -1.0*out_pixel]
        temp_action = tempfileptr.SetGeoTransform(out_transform)

        # Create the final warped raster
        gtiffdriver = gdal.GetDriverByName('GTiff')
        tempcopy = gtiffdriver.CreateCopy(outfile, tempfileptr)
        tempcopy.FlushCache()
        tempfileptr.FlushCache()
        sys.stdout.flush()
        tempcopy = None
        tempfileptr = None
        fileptr = None
        out_metadata = Raster.query_raster(outfile)
        out_metadata['name'] = name
        out_metadata['datasetid'] = datasetid
        return out_metadata
    """
