from modules import *
import random
import datetime
import os

"""
This script is used to generate most of the python and bash scripts
used in the deciduous fraction analysis. The modules are not generated
by this code. This code generates random selection of 75% and 25% of training 
and held-out samples, respectively, from the sample set. 

"""

# main program
if __name__ == '__main__':

    # import filename from file arguments
    # samp_file, work_dir, support_dir, prog_dir, pickle_dir = argv
    samp_file = "/scratch/rm885/gdrive/sync/decid/excel/samp/Alaska_all_2010_sampV2.csv"
    work_dir = "/scratch/rm885/gdrive/sync/decid/excel"
    support_dir = "/scratch/rm885/support"
    prog_dir = "/home/rm885/projects/decid/src"
    pickle_dir = "/scratch/rm885/gdrive/sync/decid/pypickle"

    sep = os.path.sep

    # create folder to write temp data to
    outtempfolder = work_dir + sep + 'pred_uncert_temp'
    Handler(dirname=outtempfolder).dir_create()

    # create folder to write out data to
    outfilefolder = work_dir + sep + 'pred_uncert_out'
    Handler(dirname=outfilefolder).dir_create()

    # create folder for shell scripts
    outshfolder = support_dir + sep + 'sh'
    Handler(dirname=outshfolder).dir_create()

    # create out folder
    outoutfolder = support_dir + sep + 'out'
    Handler(dirname=outoutfolder).dir_create()

    # bash script to run everything
    final_bash_file = outshfolder + sep + 'rf_model_submit.sh'

    # run one saved random forest model
    rf_prog = '"' + prog_dir + sep + 'run_saved_rf_model.py"'

    # bash script for job array to run one saved random forest model
    array_bash_file = outshfolder + sep + 'run_saved_rf_model.sh'

    # compile all rf models output files
    compile_prog = prog_dir + sep + 'compile_rf_model_outputs.py'

    # compile all results submit file
    res_coll_sh = outshfolder + sep + 'compile_rf_model_outputs.sh'

    # get data
    colnames, inData = Handler(filename=samp_file).read_csv_as_array()

    # number of iterations
    iter = 2000

    # number of samples
    nsamp = len(inData)

    # index for all the samples
    indx = [i for i in range(0, nsamp)]

    # split samples into 75%,25%
    ntrnsamp = int(0.75 * len(inData))
    nvalsamp = nsamp - ntrnsamp
    print('training samp: '+str(ntrnsamp))
    print('validation samp: '+str(nvalsamp))

    print(str(datetime.datetime.now()))

    # loop thru all iterations and select samples
    for i in range(1, iter+1):

        print('Iteration ' + str(i))

        # resample the input data
        tempIndx = sorted(random.sample(indx, ntrnsamp))
        tempData = inData[tempIndx]

        # get the held out data
        tempCIndx = list(set(indx)-set(tempIndx))
        tempCData = inData[tempCIndx]

        # file names
        outfile = outtempfolder + sep + 'temp_data_iter_' + str(i) + '.csv'
        outCfile = outtempfolder + sep + 'temp_Cdata_iter_' + str(i) + '.csv'

        # remove files if already exist
        outfile = Handler(filename=outfile).file_remove_check()
        outCfile = Handler(filename=outCfile).file_remove_check()

        # save data to file
        Handler(filename=outfile).write_numpy_array_to_file(tempData)
        Handler(filename=outCfile).write_numpy_array_to_file(tempCData)

    print(str(datetime.datetime.now()))

    # all filenames
    file_data = '"' + outtempfolder + sep + 'temp_data_iter_"$SLURM_ARRAY_TASK_ID".csv"'
    file_Cdata = '"' + outtempfolder + sep + 'temp_Cdata_iter_"$SLURM_ARRAY_TASK_ID".csv"'
    outfile = '"' + outfilefolder + sep + 'RF_pred_y_data_"$SLURM_ARRAY_TASK_ID".csv"'

    # write slurm array script
    script1_line1 = 'python ' + ' '.join([rf_prog, file_data, file_Cdata, outfile, '"' + pickle_dir + '"'])
    Handler(filename=array_bash_file).write_slurm_script(job_name='RFP',
                                                         time_in_mins=60,
                                                         cpus=1,
                                                         ntasks=1,
                                                         mem_per_cpu=1024,
                                                         array=True,
                                                         iterations=iter,
                                                         script_line1=script1_line1)

    # write array for result collection
    script2_line1 = 'python ' + ' '.join([compile_prog, outfilefolder])
    Handler(filename=res_coll_sh).write_slurm_script(job_name='RFPr',
                                                     time_in_mins=60,
                                                     cpus=1,
                                                     ntasks=1,
                                                     mem_per_cpu=1024,
                                                     array=False,
                                                     script_line1=script2_line1)

    # write job submission script
    script_lines = [
        "array_job_uncert=$(sbatch " + array_bash_file + " | awk '{ print $4 }')",
        "sbatch --dependency=afterany:$array_job_uncert" + res_coll_sh
    ]

    # write to file
    Handler(filename=final_bash_file).write_list_to_file(script_lines)

    # run the final file
    # os.system('sh ' + final_bash_file)
