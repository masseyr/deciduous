"""
Script to preprocess Landsat 5, 7, and 8 datasets into Mosaicked layerstacks
for deciduous fraction and tree cover regression
"""

if __name__ == '__main__':
    import ee
    import math
    import sys
    import os

    module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(module_path)
    from modules import EEFunc

    ee.Initialize()

    # ------------------functions -----------------------------------------------------------------------------------
    def ndvi_calc(img, scale_factor=10000):
        """ Normalized difference vegetation index"""
        return img.normalizedDifference(['NIR', 'RED']).select([0], ['NDVI']).multiply(scale_factor).toInt16()

    def vari_calc(img, scale_factor=10000):
        """ Visible Atmospherically Resistant Index"""
        return (img.select(['RED']).subtract(img.select(['GREEN'])))\
            .divide(img.select(['RED']).add(img.select(['GREEN'])).subtract(img.select(['BLUE'])))\
            .select([0], ['VARI']).multiply(scale_factor).toInt16()

    def ndwi_calc(img, scale_factor=10000):
        """ Normalized difference wetness index"""
        return img.normalizedDifference(['NIR', 'SWIR2']).select([0], ['NDWI']).multiply(scale_factor).toInt16()

    def nbr_calc(img, scale_factor=10000):
        """ Normalized burn ratio"""
        return img.normalizedDifference(['NIR', 'SWIR1']).select([0], ['NBR']).multiply(scale_factor).toInt16()

    def savi_calc(img, const=0.5, scale_factor=10000):
        """ Soil adjusted vegetation index"""
        return (img.select(['NIR']).subtract(img.select(['RED'])).multiply(1 + const))\
            .divide(img.select(['NIR']).add(img.select(['RED'])).add(const))\
            .select([0], ['SAVI']).multiply(scale_factor).toInt16()

    def add_indices(in_image, const=0.5, scale_factor=10000):
        """ Function to add indices to an image:  NDVI, NDWI, VARI, NBR, SAVI"""
        temp_image = in_image.float().divide(scale_factor)
        return in_image.addBands(ndvi_calc(temp_image, scale_factor))\
            .addBands(ndwi_calc(temp_image, scale_factor))\
            .addBands(vari_calc(temp_image, scale_factor))\
            .addBands(nbr_calc(temp_image, scale_factor))\
            .addBands(savi_calc(temp_image, const, scale_factor))

    def add_suffix(in_image, suffix_str):
        """ Add suffix to all band names"""
        bandnames = in_image.bandNames().map(lambda elem: ee.String(elem).toLowerCase().cat('_').cat(suffix_str))
        nb = bandnames.length()
        return in_image.select(ee.List.sequence(0, ee.Number(nb).subtract(1)), bandnames)

    def ls8_sr_corr(img):
        """ Method to correct Landsat 8 based on Landsat 7 reflectance.
            This method scales the SR reflectance values to match LS7 reflectance
            The returned values are generally lower than input image
            based on roy et al 2016"""
        return img.select(['B2'], ['BLUE']).float().multiply(0.8850).add(183).int16()\
            .addBands(img.select(['B3'], ['GREEN']).float().multiply(0.9317).add(123).int16())\
            .addBands(img.select(['B4'], ['RED']).float().multiply(0.9372).add(123).int16())\
            .addBands(img.select(['B5'], ['NIR']).float().multiply(0.8339).add(448).int16())\
            .addBands(img.select(['B6'], ['SWIR1']).float().multiply(0.8639).add(306).int16())\
            .addBands(img.select(['B7'], ['SWIR2']).float().multiply(0.9165).add(116).int16())\
            .addBands(img.select(['pixel_qa'], ['PIXEL_QA']).int16())\
            .addBands(img.select(['radsat_qa'], ['RADSAT_QA']).int16())\
            .copyProperties(img)\
            .copyProperties(img, ['system:time_start', 'system:time_end', 'system:index', 'system:footprint'])

    def ls5_sr_corr(img):
        """ Method to correct Landsat 5 based on Landsat 7 reflectance.
            This method scales the SR reflectance values to match LS7 reflectance
            The returned values are generally lower than input image
            based on sulla-menashe et al 2016"""
        return img.select(['B1'], ['BLUE']).float().multiply(0.91996).add(37).int16()\
            .addBands(img.select(['B2'], ['GREEN']).float().multiply(0.92764).add(84).int16())\
            .addBands(img.select(['B3'], ['RED']).float().multiply(0.8881).add(98).int16())\
            .addBands(img.select(['B4'], ['NIR']).float().multiply(0.95057).add(38).int16())\
            .addBands(img.select(['B5'], ['SWIR1']).float().multiply(0.96525).add(29).int16())\
            .addBands(img.select(['B7'], ['SWIR2']).float().multiply(0.99601).add(20).int16())\
            .addBands(img.select(['pixel_qa'], ['PIXEL_QA']).int16())\
            .addBands(img.select(['radsat_qa'], ['RADSAT_QA']).int16())\
            .copyProperties(img)\
            .copyProperties(img, ['system:time_start', 'system:time_end', 'system:index', 'system:footprint'])

    def ls_sr_band_correction(img):
        """ This method renames LS5, LS7, and LS8 bands and corrects LS5 and LS8 bands
            this method should be used with SR only"""
        return \
            ee.Algorithms.If(
                ee.String(img.get('SATELLITE')).compareTo('LANDSAT_8'),
                ee.Algorithms.If(
                    ee.String(img.get('SATELLITE')).compareTo('LANDSAT_5'),
                    ee.Image(img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa', 'radsat_qa'],
                                        ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'PIXEL_QA', 'RADSAT_QA'])
                             .int16()
                             .copyProperties(img)
                             .copyProperties(img,
                                             ['system:time_start',
                                              'system:time_end',
                                              'system:index',
                                              'system:footprint'])),
                    ee.Image(ls5_sr_corr(img))
                ),
                ee.Image(ls8_sr_corr(img))
            )

    def ls_sr_only_clear(image):
        """ Method to calcluate clear mask based on pixel_qa and radsat_qa bands"""
        clearbit = 1
        clearmask = math.pow(2, clearbit)
        qa = image.select('PIXEL_QA')
        qa_mask = qa.bitwiseAnd(clearmask)

        ra = image.select('RADSAT_QA')
        ra_mask = ra.eq(0)

        return ee.Image(image.updateMask(qa_mask).updateMask(ra_mask))

    def get_landsat_images(collection, bounds, start_date, end_date, start_julian, end_julian):
        """ Make collections based on given parameters"""
        return ee.ImageCollection(collection)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.calendarRange(start_julian, end_julian))\
            .filterBounds(bounds)\
            .map(ls_sr_band_correction)\
            .map(ls_sr_only_clear)\
            .map(add_indices)

    def maxval_comp_ndvi(collection, pctl=50, index='NDVI'):
        """ function to make pctl th value composite"""
        index_band = collection.select(index).reduce(ee.Reducer.percentile([pctl]))
        with_dist = collection.map(lambda(image): image.addBands(image.select(index)
                                                                 .subtract(index_band).abs().multiply(-1)
                                                                 .rename('quality')))
        return with_dist.qualityMosaic('quality')

    def interval_mean(collection, min_pctl, max_pctl, internal_bands):
        """ function to make interval mean composite"""
        temp_img = collection.reduce(ee.Reducer.intervalMean(min_pctl, max_pctl))
        return temp_img.select(ee.List.sequence(0, internal_bands.length().subtract(1)), internal_bands)

    def make_ls(args):
        """ function to make image collection"""
        collection, elevation_image, elev_scale_factor, bounds, bands, \
            start_date, end_date, start_julian, end_julian, \
            pctl, index, unmask_val = args

        all_images1 = ee.ImageCollection(get_landsat_images(collection, bounds, start_date, end_date,
                                                            start_julian[0], end_julian[0]))
        all_images2 = ee.ImageCollection(get_landsat_images(collection, bounds, start_date, end_date,
                                                            start_julian[1], end_julian[1]))
        all_images3 = ee.ImageCollection(get_landsat_images(collection, bounds, start_date, end_date,
                                                            start_julian[2], end_julian[2]))

        img_season1 = ee.Image(add_suffix(maxval_comp_ndvi(all_images1, pctl, index).select(bands), '1'))\
            .unmask(unmask_val)
        img_season2 = ee.Image(add_suffix(maxval_comp_ndvi(all_images2, pctl, index).select(bands), '2'))\
            .unmask(unmask_val)
        img_season3 = ee.Image(add_suffix(maxval_comp_ndvi(all_images3, pctl, index).select(bands), '3'))\
            .unmask(unmask_val)

        slope = ee.Terrain.slope(elevation_image).multiply(elev_scale_factor)
        aspect = ee.Terrain.aspect(elevation_image)
        topo_image = elevation_image.addBands(slope).addBands(aspect).select([0, 1, 2],
                                                                             ['elevation', 'slope', 'aspect']).int16()

        return img_season1.addBands(img_season2).addBands(img_season3).addBands(topo_image).clip(bounds)

    # geometries begin -----------------------------------------------------------------------------------------------
    # NWT, Yukon
    zone1 = ee.Geometry.Polygon(
        [[[-141.328125, 61.73152565113397],
          [-136.98804910875856, 61.55754198059565],
          [-130.25390625, 61.64816245852395],
          [-118.388671875, 61.64816245852392],
          [-109.775390625, 61.68987220045999],
          [-109.248046875, 79.88973717174639],
          [-124.8386769987568, 75.69839539226707],
          [-127.29971233251035, 73.05352679871419],
          [-127.76447139555012, 71.18241823975474],
          [-136.57574846816965, 69.76160143677127],
          [-141.416015625, 70.19999407534661]]])

    # Nunawut
    zone2 = ee.Geometry.Polygon(
        [[[-109.599609375, 79.87429692631285],
          [-110.390625, 61.56457388515459],
          [-102.12890625, 61.56457388515459],
          [-92.63671875, 61.56457388515459],
          [-80.33203125, 61.5226949459836],
          [-58.88671875, 61.18562468142283],
          [-60.75359838478971, 66.69128444147464],
          [-68.115234375, 72.5544984966527],
          [-79.46268521182355, 75.82654435670185],
          [-90.3515625, 76.59854506890709],
          [-95.44188013163352, 78.96511286239308],
          [-100.76670066607682, 79.22444104665169]]])

    # BC, Alberta
    zone3 = ee.Geometry.Polygon(
        [[[-138.076171875, 58.031372421776396],
          [-134.208984375, 53.17311920264064],
          [-124.892578125, 47.694974341862824],
          [-118.388671875, 48.574789910928885],
          [-109.775390625, 48.516604348867475],
          [-109.86328125, 50.56928286558243],
          [-109.86328125, 62.103882522897855],
          [-112.5, 61.98026726504401],
          [-119.1796875, 61.93895042666061],
          [-125.33203125, 61.93895042666061],
          [-130.693359375, 61.85614879566797],
          [-136.494140625, 61.85614879566797],
          [-140.44921875, 61.77312286453145],
          [-141.416015625, 61.83541335794044],
          [-141.44003213100132, 59.30091542995645]]])

    # saskatchewan - newfoundland
    zone4 = ee.Geometry.Polygon(
        [[[-102.0821983401313, 48.343732182918444],
          [-96.28723300378971, 48.39859262718133],
          [-92.33619096396069, 47.69095915583918],
          [-87.81162057295387, 46.796920302596604],
          [-83.42980022947569, 41.17566199379897],
          [-75.21556264793293, 43.47894951694195],
          [-74.1337105671289, 44.314986738924844],
          [-71.14921369191927, 44.75275797950917],
          [-69.01569850009935, 46.123823821620945],
          [-65.54991505307743, 43.04775213049375],
          [-59.10953400025778, 45.63721256282798],
          [-52.28045548936922, 46.41967042543777],
          [-53.408531066369335, 52.30015394449034],
          [-57.663708890390865, 54.94669881803222],
          [-61.97826979439657, 60.00768963705834],
          [-62.701459572887245, 61.781130426408225],
          [-74.52581787486929, 61.918873824879256],
          [-77.50806588283604, 61.91829272622474],
          [-81.63042692545287, 61.710622267579524],
          [-88.99818618459153, 61.91763439088837],
          [-93.99771914137938, 61.918135426805065],
          [-98.99746257904286, 61.83642058422084],
          [-103.64659209642855, 61.75483188993627],
          [-110.57640085568312, 61.92313920977558],
          [-110.77515484366108, 48.58088542654131]]])

    # alaska
    zone5 = ee.Geometry.Polygon(
        [[[-167.783203125, 68.07330474079025],
          [-164.970703125, 66.96447630005638],
          [-168.486328125, 66.26685631430843],
          [-168.22265625, 64.54844014422517],
          [-163.740234375, 63.89873081524393],
          [-166.728515625, 63.11463763252092],
          [-168.3984375, 59.355596110016315],
          [-159.78515625, 57.326521225217064],
          [-165.498046875, 55.727110085045986],
          [-164.70703125, 53.225768435790194],
          [-158.818359375, 54.87660665410869],
          [-153.193359375, 55.7765730186677],
          [-152.490234375, 57.27904276497778],
          [-149.4140625, 58.99531118795094],
          [-145.810546875, 59.977005492196],
          [-140.7568359375, 59.153403092050375],
          [-140.90746584324359, 65.67611691802753],
          [-140.888671875, 70.1851027549897],
          [-146.162109375, 70.55417853776078],
          [-154.248046875, 71.49703690095419],
          [-160.3125, 71.24435551310674],
          [-167.255859375, 68.97416358340674]]])

    boreal = ee.Geometry.Polygon(
        [[[-170.170703125, 50.55860413479424],
          [-158.217578125, 52.68617167041279],
          [-153.30939285434152, 55.911669002771404],
          [-150.04375, 58.80503543579855],
          [-139.233203125, 58.068985987821954],
          [-136.15703125, 54.714910470066776],
          [-132.28984375, 47.267822642617666],
          [-131.4109375, 41.59465704101877],
          [-123.9066476728093, 41.3211582877145],
          [-116.25939622443843, 41.46034634091618],
          [-108.55937499999999, 41.36419260424219],
          [-102.89042968749999, 41.397166266344954],
          [-96.21074218749999, 41.26517132419739],
          [-90.49785156249999, 41.23213080790694],
          [-84.25761718749999, 41.23213080790694],
          [-77.05058593749999, 41.29819513068916],
          [-68.12968749999999, 41.26517132419742],
          [-56.70390624999999, 43.7589530883319],
          [-50.72734374999999, 47.38696854291394],
          [-52.92460937499999, 52.79259685897027],
          [-57.84648437499999, 56.64698491697063],
          [-64.87773437499999, 61.5146781712937],
          [-73.96213049438268, 62.80048928805028],
          [-78.32093919613442, 62.78853877630148],
          [-80.52226562499999, 59.74792657078106],
          [-84.9374198886535, 59.27782159718193],
          [-88.37276592344836, 60.37639431989281],
          [-94.32109374999999, 64.65432185309979],
          [-110.84453124999999, 69.12528180058297],
          [-127.45585937499999, 70.69439418030035],
          [-145.209765625, 71.54742032805514],
          [-151.99237509694262, 71.7394004113125],
          [-158.920703125, 71.46378059608598],
          [-163.9438033714605, 70.33273202428589],
          [-167.358203125, 68.74622173946399],
          [-168.588671875, 67.76966607015038],
          [-168.764453125, 66.43638686378715],
          [-168.54359182813454, 65.57009532838762],
          [-169.643359375, 64.74821921769586],
          [-172.1921875, 63.91066619210742],
          [-177.66337890625, 60.31862431477453],
          [178.16636398271964, 57.45355647434897],
          [171.06398672131763, 52.978982153380194],
          [174.72537102620095, 51.303685681238484],
          [-178.3430865436669, 50.38214994852445]]])

    # geometries end -------------------------------------------------------------------------------------------------

    ls5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")
    ls7 = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
    ls8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")

    elevation = ee.Image('USGS/GMTED2010')

    # collections end ----------------------------------------------------------------------------------------

    elev_scale_factor = 10000
    pctl = 50
    index = 'NDVI'
    unmask_val = -9999

    startJulian1 = 90
    endJulian1 = 165
    startJulian2 = 180
    endJulian2 = 240
    startJulian3 = 255
    endJulian3 = 330

    # start year , end year , year
    years = {
        '1992': (1987, 1997),
        '2000': (1998, 2002),
        '2005': (2003, 2007),
        '2010': (2008, 2012),
        '2015': (2013, 2018)
    }

    # zone name, bounds
    zones = {
        'zone1': zone1,
        'zone2': zone2,
        'zone3': zone3,
        'zone4': zone4,
        'zone5': zone5,
        # 'boreal': boreal,
    }

    internal_bands = ee.List(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'PIXEL_QA',
                              'RADSAT_QA', 'NDVI', 'NDWI', 'NBR', 'VARI', 'SAVI'])

    bands = ee.List(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2',
                     'NDVI', 'NDWI', 'NBR', 'VARI', 'SAVI'])

    startJulian = [startJulian1, startJulian2, startJulian3]
    endJulian = [endJulian1, endJulian2, endJulian3]

    # static definitions end ----------------------------------------------------------------------------------------

    all_images = ls5.merge(ls7).merge(ls8)

    print(EEFunc.expand_image_meta(all_images.first()))

    for year, dates in years.items():

        startDate = ee.Date.fromYMD(dates[0], 1, 1)
        endDate = ee.Date.fromYMD(dates[1], 12, 31)

        for zone, bounds in zones.items():

            args = (all_images, elevation, elev_scale_factor, bounds, bands,
                    startDate, endDate, startJulian, endJulian,
                    pctl, index, unmask_val)

            output_img = ee.Image(make_ls(args))

            out_name = 'Boreal_NA_median_SR_NDVI_LS5C_' + zone + '_' + year

            task_config = {
                'driveFileNamePrefix': out_name,
                'crs': 'EPSG:4326',
                'scale': 30,
                'maxPixels': 1e13,
                'fileFormat': 'GeoTIFF',
                'region': bounds,
                'driveFolder': 'Boreal_NA_median_SR_NDVI'
            }

            task1 = ee.batch.Export.image(output_img,
                                          out_name,
                                          task_config)

            task1.start()
            print(task1)
