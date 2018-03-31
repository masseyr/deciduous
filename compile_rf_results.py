from modules import *

if __name__ == '__main__':

    pattern = 'all'

    indir = '/scratch/rm885/gdrive/sync/decid/excel/pred_uncert_out'

    outfile = '/scratch/rm885/gdrive/sync/decid/excel/rf_info.csv'

    filelist = find_all(pattern, indir)

    listdict = [get_rfpickle_info(file_) for file_ in filelist]

    names = [dict_['name'] for dict_ in listdict]
    rsqs = [dict_['rsq'] for dict_ in listdict]
    rmses = [dict_['rmse'] for dict_ in listdict]

    outlist = [', '.join([names[i], str(rsqs[i]), str(rmses[i])]) for i in range(0, len(names))]

    outlist = ['name, rsq, rmse'] + outlist

    write_list_to_file(outfile, outlist)
