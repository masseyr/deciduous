if __name__ == '__main__':
    import ee
    import sys
    import os

    module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(module_path)
    from modules import Vector

    infile = "C:/temp/arctic_oroarctic_wgs84_buf50km.shp"
    outfile = "C:/temp/arctic_oroarctic_wgs84_buf50km.tif"
    pixel_size = (0.0168, 0.0168)

    vec = Vector(infile)

    print(vec)

    vec.rasterize(outfile, pixel_size, 1, 0, None)
