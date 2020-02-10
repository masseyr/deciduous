import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')

file1 = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/TEST_RUNS/summary_rsq_parameters.csv"
file2 = "C:/Users/rm885/Dropbox/projects/NAU/landsat_deciduous/data/TEST_RUNS/summary_rmse_parameters.csv"

df1 = pd.read_csv(file1)
array1 = df1.values.tolist()
x = list(int(elem) for elem in list(df1)[1:])
y = list(int(row[0]) for row in array1)
z = np.matrix(list(list(small_elem*0.85 for small_elem in elem[1:]) for elem in array1))

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.meshgrid(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0.45, 0.75)
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

