#!/bin/sh
#SBATCH -t 3- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20
#SBATCH --mem=100G 
python3 ../run_JAW_wCV_expts.py --dataset communities --muh_fun_name NN --bias 0.825 --ntrial 20 --K_vals 5 10 25 50
