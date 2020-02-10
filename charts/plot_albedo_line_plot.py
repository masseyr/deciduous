from modules import *
import numpy as np


if __name__ == '__main__':

    folder = 'c:/temp/'

    filelist = ['forcing_samples1.csv',
                'forcing_samples2.csv',
                'forcing_samples3.csv',
                'forcing_samples4.csv',
                'forcing_samples5.csv',
                'forcing_samples6.csv',
                'forcing_samples7.csv']

    dict_list = list()

    for infile in filelist:
        dict_list += Handler(folder+infile).read_from_csv(return_dicts=True)

    print(len(dict_list))

    count = 0
    for dict_ in dict_list:
        if dict_['sum_forc'] > 0.0:
            print(dict_)
            count += 1

    print(count)
