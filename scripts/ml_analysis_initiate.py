from modules import *
from sys import argv

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
    dir_create(outtempfolder)

    # create folder to write out data to
    outfilefolder = work_dir + sep + 'pred_uncert_out'
    dir_create(outfilefolder)

    # create folder for shell scripts
    outshfolder = support_dir + sep + 'sh'
    dir_create(outshfolder)

    # create out folder
    outoutfolder = support_dir + sep + 'out'
    dir_create(outoutfolder)

    # bash script to run everything
    final_bash_file = outshfolder + sep + 'y_param_tau_submit.sh'

    # random forest program
    rf_prog = '"' + prog_dir + sep + 'y_param_predict.py"'

    # bash script for job array
    array_bash_file = outshfolder + sep + 'y_param_predict_job_array.sh'

    # compile all y files program
    compile_prog = prog_dir + sep + 'compile_y_param_files.py'

    # compile all results submit file
    res_coll_sh = outshfolder + sep + 'compile_y_param_files.sh'

    # run saved rf model sbatch file
    rf_sbatch = outshfolder + sep + 'run_saved_rf_model.sh'

    # get data
    colnames, inData = read_csv_as_array(samp_file)

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
        outfile = file_remove_check(outfile)
        outCfile = file_remove_check(outCfile)

        # save data to file
        write_numpy_array_to_file(outfile, tempData)
        write_numpy_array_to_file(outCfile, tempCData)

    print(str(datetime.datetime.now()))

    # all filenames
    file_data = '"' + outtempfolder + sep + 'temp_data_iter_"$SLURM_ARRAY_TASK_ID".csv"'
    file_Cdata = '"' + outtempfolder + sep + 'temp_Cdata_iter_"$SLURM_ARRAY_TASK_ID".csv"'
    outfile = '"' + outfilefolder + sep + 'RF_pred_y_data_"$SLURM_ARRAY_TASK_ID".csv"'

    # write slurm array script
    script1_line1 = 'python ' + ' '.join([rf_prog, file_data, file_Cdata, outfile, '"' + pickle_dir + '"'])
    write_slurm_script(array_bash_file, job_name='RFP', time_in_mins=60,
                       cpus=1, ntasks=1, mem_per_cpu=1024,
                       array=True, iterations=iter, script_line1=script1_line1)

    # write array for result collection
    script2_line1 = 'python ' + ' '.join([compile_prog, outfilefolder])
    write_slurm_script(res_coll_sh, job_name='RFPr', time_in_mins=60,
                       cpus=1, ntasks=1, mem_per_cpu=1024,
                       array=False, script_line1=script2_line1)

    # write job submission script
    script_lines = [
        "array_job_uncert=$(sbatch " + array_bash_file + " | awk '{ print $4 }')",
        "sbatch --dependency=afterany:$array_job_uncert" + res_coll_sh
    ]

    # write to file
    write_list_to_file(final_bash_file, script_lines)

    # run the final file
    # os.system('sh ' + final_bash_file)
