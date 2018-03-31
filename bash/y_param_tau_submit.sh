array_job_uncert=$(sbatch /scratch/rm885/support/sh/y_param_predict_job_array.sh | awk '{ print $4 }')
sbatch --dependency=afterany:$array_job_uncert/scratch/rm885/support/sh/compile_y_param_files.sh
