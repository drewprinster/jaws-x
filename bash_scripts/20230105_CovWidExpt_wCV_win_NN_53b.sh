#!/bin/sh
#SBATCH -t 2- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 
#SBATCH --mem=50G 
python3 ../run_JAW_wCV_expts.py --dataset wine --muh_fun_name NN --bias 0.53 --ntrial 20 --K_vals 5 10 25 50
