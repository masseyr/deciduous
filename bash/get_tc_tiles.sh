#!/bin/bash
#SBATCH --job-name=tcf
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000

date

python '/home/rm885/projects/decid/src/get_tc_tiles.py' 2010 '/scratch/rm885/gdrive/sync/decid/tc_files/gz' '/scratch/rm885/gdrive/sync/decid/bounds/above_domain/domain_simple.shp' '/scratch/rm885/gdrive/sync/decid/bounds/wrs2/wrs2_descending.shp'

date
