from modules import *
import os

sep = os.path.sep

if __name__ == '__main__':

    pyfile = "/home/rm885/projects/decid/src/run_saved_rf_model.py"

    # input folder that contains tif files, dont forget the '/' at the end
    rf_picklefile = "/scratch/rm885/gdrive/sync/decid/pypickle/temp_data_iter_335.pickle"

    # dir containing rasters,
    datadir = "/scratch/rm885/gdrive/sync/decid/alaska_data/uncomp/"

    # out sh directory
    outshfolder = "/scratch/rm885/support/sh"

    # output prediction directory,
    outdir = "/scratch/rm885/gdrive/sync/decid/alaska_data/classifV1"

    # run all sbatch files
    runsh = outshfolder + sep + 'runallsbatch.sh'

    filelist = find_all('.tif', datadir)

    shfilenamelist = list()

    for i, file_ in enumerate(filelist):

        rf_sbatch = outshfolder + sep + 'run_saved_rf_model_' + str(i+1) + '.sh'

        shfilenamelist.append(' '.join(['sbatch', rf_sbatch]))

        script1_line1 = 'python "' + '" "'.join([pyfile, rf_picklefile, file_, outdir]) + '"'
        write_slurm_script(rf_sbatch, job_name='rf_class', time_in_mins=1439,
                           cpus=1, ntasks=1, mem_per_cpu=48000,
                           script_line1=script1_line1)

    print(shfilenamelist)

    write_list_to_file(runsh, shfilenamelist)

    os.system('sh ' + runsh)
