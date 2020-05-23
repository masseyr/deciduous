import numpy as np
import copy
import matplotlib.pyplot as plt
from geosoup import Raster


def calc_parabola_param(pt1, pt2, pt3):
    """
    define a parabola using three points
    """
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    _m_ = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a_param = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / _m_
    b_param = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / _m_
    c_param = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / _m_

    return a_param, b_param, c_param


def moving_average(arr,
                   n=3,
                   cascaded=True):
    """
    Method to smooth an array of numbers using moving array method
    :param arr: Input array
    :param n: Number of elements to consider for moving average. Must be an odd number (default: 3)
    :param cascaded: If the smoothing should be cascaded from the specified moving average to the lowest(1)
    :return: smoothed array
    """
    if type(arr) in (list, tuple, dict, set):
        arr_copy = np.array(list(copy.deepcopy(arr)))
    elif type(arr) == np.ndarray:
        arr_copy = arr.copy()
    else:
        raise ValueError("Input array type not understood")

    dtype = arr_copy.dtype

    if n < 1:
        raise ValueError('n cannot be less than 1')

    if cascaded:
        ker_list = list(ker_size for ker_size in reversed(range(n+1)) if ker_size % 2 != 0)
        if ker_list[-1] == 0:
            ker_list = ker_list[:-1]
    else:
        ker_list = [n]

    print(ker_list)

    for ker_size in ker_list:

        if ker_size == 1:
            left_param = calc_parabola_param((0, arr_copy[0]),
                                             (2, arr_copy[2]),
                                             (3, arr_copy[3]))
            arr_copy[1] = np.sum(left_param)

            rght_param = calc_parabola_param((len(arr_copy), arr_copy[-1]),
                                             (len(arr_copy)-2, arr_copy[-3]),
                                             (len(arr_copy)-3, arr_copy[-4]))

            arr_copy[-2] = rght_param[0]*((len(arr_copy)-1) ** 2) + \
                           rght_param[1]*(len(arr_copy)-1) + \
                           rght_param[2]

            '''
            out_arr = np.concatenate([
                                      np.array([arr_copy[0]]),
                                      np.array([(arr_copy[0]+arr_copy[2])/2.]),
                                      np.array(arr_copy[2:-2]),
                                      np.array([(arr_copy[-3]+arr_copy[-1])/2.]),
                                      np.array([arr_copy[-1]])
                                      ])
            '''
        else:
            tail = (ker_size - 1) / 2

            ret = np.cumsum(arr_copy,
                            dtype=np.float32)

            ret[ker_size:] = ret[ker_size:] - ret[:-ker_size]

            out_arr = np.concatenate([arr_copy[0:tail],
                                      ret[ker_size - 1:] / ker_size,
                                      arr_copy[-tail:]])
        arr_copy = out_arr

        print(ker_size)
        print(arr_copy)
        plt.plot(range(len(arr)), arr_copy)

    plt.show()
    arr_copy = arr_copy.astype(dtype)

    if type(arr) in (list, tuple, dict, set):
        return arr_copy.tolist()
    else:
        return arr_copy


if __name__ == '__main__':

    arr = [12, 4, 7, 13, 9, 8, 1, 13, 7, 2, 1, 36, 33]
    arr2 = moving_average(arr, n=7)
    print(arr)
    print(arr2)


    file1 = "D:/temp/above2017_0629_lvis2b_tif/LVIS2_ABoVE2017_0629_R1803_061571_umd_l2b.tif"
    outfile1 = "D:/temp/above2017_0629_lvis2b_tif/LVIS2_ABoVE2017_0629_R1803_061571_umd_l2b_clip2.tif"
    cutfile = "D:/temp/LVIS2_ABoVE2017_0629_R1803_061571_umd_l2b.shp"

    ras = Raster(file1)

    ras.clip(cutline_file=cutfile, outfile=outfile1, apply_mask=False)
