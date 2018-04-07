from common import Handler, Sublist
import numpy as np
import pandas as pd
from numpy.random import randn
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["patch.force_edgecolor"] = True
plt.interactive(False)



if __name__ == '__main__':
    dataset1 = randn(1000)

    file1='c:/temp/plot1.png'
    file1=Handler(file1).file_remove_check()
    #plt.hist(dataset1, color='red')
    #plt.savefig(file1)

    dataset2=randn(1000)

    """
    plt.hist(dataset1, normed=True, color='indianred', alpha=0.5, bins=20, edgecolor='black', linewidth=1)
    plt.hist(dataset2, normed=True, alpha=0.5, bins=20, edgecolor='black', linewidth=1)
    plt.savefig(file1)
    """

    file2 = 'c:/temp/plot2.png'
    file2 = Handler(file2).file_remove_check()

    p2 = sns.jointplot(dataset1, dataset2, kind='hex')
    p2.savefig(file2)



