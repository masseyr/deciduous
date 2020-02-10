from modules import *
import numpy as np
import matplotlib
import warnings
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.interpolate import interpn
from scipy.signal import savgol_filter
import statsmodels.api as sm
import statsmodels.formula.api as smf

# plt.rcParams['agg.path.chunksize'] = 100000
# plt.rcParams["patch.force_edgecolor"] = True
# plt.interactive(False)
plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})
plt.rcParams['axes.labelweight'] = 'bold'

import matplotlib.ticker as mtick
def div_10(x, *args):
    """
    The function that will you be applied to your y-axis ticks.
    """
    x = float(x)/1000000
    return "{:.1f}".format(x)



if __name__ == '__main__':

    file1 = "C:/Users/Richard/Downloads/tc_2015_2000_show.tif"
    file2 = "C:/Users/Richard/Downloads/tc_2015_2000_show_fire.tif"
    file3 = "C:/Users/Richard/Downloads/decid_diff_2015_2000_show_wotc.tif"
    file4 = "C:/Users/Richard/Downloads/decid_diff_2015_2000_show_wotc_fire.tif"

    ras1 = Raster(file3)
    ras1.initialize(get_array=True)

    arr = np.reshape(ras1.array, ras1.shape[1]*ras1.shape[2])

    arr[arr==0] = -9999

    hist, bin_edges = np.histogram(arr, bins=50, range=(-100, 100))

    print(bin_edges)
    print(hist)


    plt.hist(arr,
                    alpha=0.8,
                    color='#107714',  #'#0C92CA',
                    edgecolor='black',
                    bins=bin_edges)
    # Apply to the major ticks of the y-axis the function that you defined.
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(div_10))
    plt.savefig(file3.split('.tif')[0]+'.png', dpi=1200)
    plt.close()


