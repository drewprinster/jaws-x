#!/bin/sh
#SBATCH -t 3- 
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=80G 
python3 ../run_JAW_wCV_expts.py --dataset wave --muh_fun_name NN --bias 0.0000925 --ntrial 20 --K_vals 5 10 25 50
