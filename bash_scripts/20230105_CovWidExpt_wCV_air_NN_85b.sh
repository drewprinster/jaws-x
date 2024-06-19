#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10
#SBATCH --mem=50G 
python3 ../run_JAW_wCV_expts.py --dataset airfoil --muh_fun_name NN --bias 0.85 --ntrial 1 --K_vals 5 10 25 50
