#!/bin/bash
#SBATCH --job-name=rclone
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000
module load rclone
rclone --include "Alaska_decid_frac_2010_uncertainty_v2*.tif" copy masseyr44:/work/ABoVE_files/alaska_files/ /scratch/rm885/gdrive/sync/decid/alaska_data/uncert
exit 0
