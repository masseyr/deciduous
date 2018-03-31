#!/bin/bash
#SBATCH --job-name=gdal_tif
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=54000
#SBATCH --partition=all
#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/gdal_mosaic_tif_%j.out
module purge
module load gdal/2.2.1

# bash script to make mosaics

date
echo 'Begin!~~~~~~~~~~'
datadir='/scratch/rm885/gdrive/sync/decid/alaska_data/classifV1/'

files=(${datadir}*.tif)

mosaic=$datadir'uncert_mosaic.tif'
compmosaic=$datadir'uncert_mosaic_vis.tif'

echo '********************************************************************************************'

echo 'Data folder: '$datadir
echo 'Mosaic: '$mosaic
echo 'Compressed mosaic: '$compmosaic

echo '********************************************************************************************'

gdal_merge.py -init "0" -o $mosaic -of GTiff -ps 0.00027 0.00027 -ot Float32 ${files[*]}
gdal_translate -of GTiff -co COMPRESS=LZW -co BIGTIFF=YES $mosaic $compmosaic
gdaladdo -ro $compmosaic 2 4 8 16 32 64 128 256 --config COMPRESS_OVERVIEW LZW

echo '********************************************************************************************'
echo 'Done!~~~~~~~~~~'
date
