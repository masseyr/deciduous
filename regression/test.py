if __name__ == '__main__':
    import sys
    import os
    module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(module_path)
    from modules import *

    print(Raster)
    print(Vector)
    print(HRFRegressor)
    print(EEFunc)
