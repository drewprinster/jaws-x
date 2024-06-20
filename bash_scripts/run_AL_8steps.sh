#!/bin/sh
#SBATCH -t 1- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=20G
python ../run_JAWS-X_active.py --dataset airfoil --n_steps 8 --n_queries_ann 16 --n_train_initial 32 --seed_initial 0 --n_seed 30 --K_vals 16
