import matplotlib.pyplot as plt
import numpy as np
from modules import *


def extract_array_from_file(filename,
                    nodataval=-99.0):
    h = Handler(filename)
    mat = h.read_array_from_csv()
    lenr, lenc = mat.shape
    arr = mat.reshape(lenr*lenc)
    arr = arr[arr != nodataval]
    return arr


def make_heatmap(points,
                 xbins=50,
                 ybins=50,
                 xbinvals=None,
                 ybinvals=None,
                 xlim=None,
                 ylim=None,
                 plotfile=None,
                 line=True,
                 color=plt.cm.gnuplot2_r):

    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]

    print(len(points))

    if xlim is None:
        xlim = (min(x), max(x))

    if ylim is None:
        ylim = (min(y), max(y))

    if xlim is not None:
        xloc = list(k for k in range(0, len(x)) if xlim[0] <= x[k] <= xlim[1])
    else:
        xloc = list(range(0, len(x)))

    if ylim is not None:
        yloc = list(k for k in range(0, len(y)) if ylim[0] <= y[k] <= ylim[1])
    else:
        yloc = list(range(0, len(y)))

    loc = list(set(xloc) & set(yloc))

    points_ = list(points[k] for k in loc)
    print(len(points_))

    x_ = list(pt[0] for pt in points_)
    y_ = list(pt[1] for pt in points_)

    print(max(x_), min(x_))
    print(max(y_), min(y_))

    if xbinvals is not None and ybinvals is not None:
        zi, xedges, yedges = np.histogram2d(x_, y_, bins=[xbinvals, ybinvals])
    else:
        zi, xedges, yedges = np.histogram2d(x_, y_, bins=[xbins, ybins])

    xi, yi = np.meshgrid(xedges, yedges)

    plt.pcolormesh(xi, yi, zi, cmap=color)
    plt.xlabel('Variance in tree predictors')
    plt.ylabel('Abs. difference in observed and predicted')
    plt.colorbar().set_label('Data-points per bin')

    if line:
        fit = np.polyfit(x, y, 1)
        print(fit)
        plt.plot(x_, [fit[0] * x_[i] + fit[1] for i in range(0, len(x_))], 'r-', lw=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)

    if plotfile is None:
        plt.show()
    else:
        plt.savefig(plotfile)

    plt.close()


if __name__ == '__main__':

    yhb_arr = extract_array_from_file("D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/RF_bootstrap_2000_y_hat_bar_v1.csv")

    vary_arr = extract_array_from_file("D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/RF_bootstrap_2000_var_y_v1.csv")

    yf_arr = extract_array_from_file("D:/shared/Dropbox/projects/NAU/landsat_deciduous/data/RF_bootstrap_2000_y_v1.csv")

    y = np.abs(yf_arr-yhb_arr)
    x = vary_arr

    pts = [(y[i],x[i]) for i in range(0, len(x))]

    print('----------------------------------------------')

    ez=1

    plotfile = 'c:/temp/test_plot10_.png'
    make_heatmap(pts,
                 xbins=50,
                 ybins=50,
                 plotfile=plotfile,
                 line=True,
                 xlim=(0, 0.0025),
                 ylim=(0, 0.5))

