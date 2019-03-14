import csv
from modules import Handler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties


def get_next_row_dict(filename):

    with open(filename, "r") as f:
        reader = csv.reader(f)
        count = 0
        colnames = list()
        for file_row in reader:
            if count == 0:
                colnames = file_row
            else:
                values = list(Handler.string_to_type(elem.strip())
                              for elem in file_row)

                if len(values) == len(colnames):
                    row_dict = dict(zip(colnames, values))
                    yield row_dict
            count += 1


if __name__ == '__main__':
    file1 = "C:/temp/rf_pickle_test_v13_compilation.csv"

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')

    const_var1 = 'trees'
    const_var1_val = 500

    const_var2 = 'samp_leaf'
    const_var2_val = 1

    n_runs = 1000

    xvar = 'samp_split'
    yvar = 'max_feat'

    zvar = 'rsq'

    dict_list = list()
    read_count = 0

    for file_dict in get_next_row_dict(file1):

        if read_count % 1000000 == 0:
            print('Read {} lines'.format(str(read_count)))

        if file_dict[const_var1] == const_var1_val and \
                file_dict[const_var2] == const_var2_val:

            dict_list.append(file_dict)

        read_count += 1

    x = np.array(list(set(list(elem[xvar] for elem in dict_list))))
    y = np.array(list(set(list(elem[yvar] for elem in dict_list))))

    lenx = len(x)
    leny = len(y)

    z_vals = np.zeros((n_runs, leny, lenx))
    z_count = np.zeros((leny, lenx), dtype='int16')

    for elem_dict in dict_list:
        if elem_dict[const_var1] == const_var1_val and \
                elem_dict[const_var2] == const_var2_val:

            xloc = np.where(x == elem_dict[xvar])[0][0]
            yloc = np.where(y == elem_dict[yvar])[0][0]

            z_vals[z_count[yloc, xloc], yloc, xloc] = elem_dict[zvar]

            z_count[yloc, xloc] += 1

    z = np.mean(z_vals, 0)

    x, y = np.meshgrid(x, y)

    print(x.shape)
    print(y.shape)
    print(z.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z*0.011, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0.67, 0.71)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # ax.set_xlabel('Variable Split', fontproperties=font, fontsize='large', fontweight='bold')
    # ax.set_ylabel('Trees', fontproperties=font, fontsize='large', fontweight='bold')
    # ax.set_zlabel('R-squared', fontproperties=font, fontsize='large', fontweight='bold', rotation=180)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontproperties(font)

    zlabels = ax.zaxis.get_majorticklabels()
    for label in zlabels:
        label.set_fontproperties(font)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=45, azim=235)

    plt.show()
