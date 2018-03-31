#!/bin/bash
#SBATCH --job-name=ML_TAU
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --partition=all
#SBATCH --output=/home/rm885/slurm-jobs/y_param_tau_submit_slurm_%j.out

date
sh /scratch/rm885/support/sh/y_param_tau_submit.sh
date
