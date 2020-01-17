#!/bin/bash
#SBATCH --job-name=wt_r2015
#SBATCH --time=11:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000
#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/wt_resmp_2015_slurm_%A_%a.out

date

file1="/scratch/rm885/gdrive/sync/decid/ABoVE_data_boreal_output_h2_files_only/ABoVE_median_SR_NDVI_boreal_2015_prediction_vis_nd_2.tif"
file2="/scratch/rm885/gdrive/sync/decid/ABoVE_data_boreal_output_tc1_files_only/ABoVE_median_SR_NDVI_boreal_2015_tc_prediction_vis_nd.tif"
outfile="/scratch/rm885/gdrive/sync/decid/ABoVE_data_boreal_output_h2_files_only/decid_2015_prediction_vis_nd_2_05deg_v2.tif"

python '/home/rm885/projects/decid/src/weighted_resampling.py' $file1 $file2 $outfile

#compfile="/scratch/rm885/gdrive/sync/decid/ABoVE_data_boreal_output_h2/decid_2015_prediction_vis_nd_2_250m_lzw.tif"
#gdal_translate -ot Byte -of GTiff -co COMPRESS=LZW -co BIGTIFF=YES $outfile $compfile
#gdaladdo -ro $compfile 2 4 8 16 32 64 128 256 --config COMPRESS_OVERVIEW LZW --config BIGTIFF_OVERVIEW YES
date
