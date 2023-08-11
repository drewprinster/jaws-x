#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10
#SBATCH --mem=50G 
python ../run_JAWS-X_FCS_expts.py --fitness_str blue --n_trains 192 --lmbdas 0 --n_seed 20 --K_vals 8 12 16 24 32 48 96 192
