#!/bin/bash
#SBATCH --job-name=RFPr
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024
#SBATCH --partition=all
#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/slurm_%j.out
date
python /home/rm885/projects/decid/compile_y_param_files.py /scratch/rm885/gdrive/sync/decid/excel/pred_uncert_out
date
