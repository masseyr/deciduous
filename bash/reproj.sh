#!/bin/bash
#SBATCH --job-name=reproj
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem=54000
#SBATCH --partition=all
#SBATCH --output=/home/rm885/slurm-jobs/py_reproj_%j.out

# This is a bash script to reproject tif files

# File to use spatial reference information from
basefile='/scratch/rm885/gdrive/sync/decid/NLCD/ak_nlcd_2011_forest_mask_.tif'


# File to reproject
infile='/scratch/rm885/gdrive/sync/decid/NLCD/ak_nlcd_2011_forest_mask_.tif'

# Reproject
date

python /home/rm885/project/decid/src/reproj.py $basefile $infile

date