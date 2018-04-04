#!/bin/bash
#SBATCH --job-name=rf_test
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem=18000
#SBATCH --partition=all
#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/run_saved_rf_model_test_%j.out

#This is a bash script to make tiles from uncompressed tif files
date
echo 'Begin!~~~~~~~~~~'
#input folder that contains tif files, dont forget the '/' at the end
rf_picklefile="/scratch/rm885/gdrive/sync/decid/pypickle/temp_data_iter_335.pickle"

#output prediction directory,
outdir="/scratch/rm885/gdrive/sync/decid/alaska_data/test"

#what this statement does: for this element in job array, pick filename based on task ID
inraster="/scratch/rm885/gdrive/sync/decid/alaska_data/test/AK_LS_coll1_2010-0000013824-0000076032_uncomp_4096_5568.tif"

echo '********************************************************************************************'

echo 'Random Forest file: '$rf_picklefile
echo 'Inraster: '$inraster
echo 'Output folder: '$outdir

echo '********************************************************************************************'

#run saved rf models
srun python "/home/rm885/projects/decid/src/run_saved_rf_model.py" $rf_picklefile $inraster $outdir


echo '********************************************************************************************'
echo 'Done!~~~~~~~~~~'
date
