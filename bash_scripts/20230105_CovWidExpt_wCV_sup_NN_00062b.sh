#!/bin/sh
#SBATCH -t 3- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20
#SBATCH --mem=100G 
python3 ../run_JAW_wCV_expts.py --dataset superconduct --muh_fun_name NN --bias 0.00062 --ntrial 20 --K_vals 5 10 25 50
