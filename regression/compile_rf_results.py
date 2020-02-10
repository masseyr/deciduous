"""
This script is used to compile RF model r-squared and RMSE values
for plotting.
"""

if __name__ == '__main__':
    import sys
    import os

    module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(module_path)
    from modules import *

    pattern = 'all'

    indir = '/scratch/rm885/gdrive/sync/decid/excel/pred_uncert_out'

    outfile = '/scratch/rm885/gdrive/sync/decid/excel/rf_info.csv'

    filelist = Handler(dirname=indir).find_all(pattern)

    listdict = [get_rfpickle_info(file_) for file_ in filelist]

    names = [dict_['name'] for dict_ in listdict]
    rsqs = [dict_['rsq'] for dict_ in listdict]
    rmses = [dict_['rmse'] for dict_ in listdict]

    outlist = [', '.join([names[i], str(rsqs[i]), str(rmses[i])]) for i in range(0, len(names))]

    outlist = ['name, rsq, rmse'] + outlist

    Handler(filename=outfile).write_list_to_file(outlist)
