#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=15
#SBATCH --mem=75G 
python ../run_JAWS-X_FCS_expts.py --fitness_str red --n_trains 384 --lmbdas 1 --n_seed 20 --K_vals 8 12 16 24 32 48 96 192 384