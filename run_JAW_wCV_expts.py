## File last updated June 19, 2024
## Code for running experiments in Figure 5 of main paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
np.random.seed(98765)

import tqdm as tqdm
import random
from sklearn import decomposition
from datetime import date
import argparse
import warnings

import sys,os
sys.path.insert(0,os.getcwd() + '../')


from utils.JAWS_CV_utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run JAW experiments with given bias, # trials, mu func, and dataset.')
    
    parser.add_argument('--dataset', type=str, default='airfoil', help='Dataset for experiments.')
    parser.add_argument('--muh_fun_name', type=str, default='RF', help='Mu (mean) function predictor.')
    parser.add_argument('--bias', type=float, default=1.0, help='Scalar bias magnitude parameter for exponential tilting covariate shift.')
    parser.add_argument('--ntrial', type=int, default=10, help='Number of trials (experiment replicates) to complete.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
    parser.add_argument('--ntrain', type=int, default=200, help='Number of training datapoints')
    parser.add_argument('--run_effective_sample_size', type=bool, default=False, help='Run effective sample size expts?')
    parser.add_argument('--K_vals', nargs='+', help='Values of K to try', required = True)
    
    ## python run_JAW.py dataset muh_fun bias ntrial alpha
    ## python run_JAW.py airfoil RF 1 20 0.1
    
    args = parser.parse_args()
    dataset = args.dataset
    muh_fun_name = args.muh_fun_name
    bias = args.bias
    ntrial = args.ntrial
    ntrial=50 ## Override to 50 for now
    alpha = args.alpha
    n = args.ntrain
    run_effective_sample_size = bool(args.run_effective_sample_size)
#     K_vals = [int(K) for K in args.K_vals]
    K_vals = [5, 10, 20, 25, 50] ## Override for now
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    
    if (dataset == 'airfoil'):
        airfoil = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'SCS_datasets/airfoil/airfoil.txt', sep = '\t', header=None)
        airfoil.columns = ["Frequency","Angle","Chord","Velocity","Suction","Sound"]
        X_airfoil = airfoil.iloc[:, 0:5].values
        X_airfoil[:, 0] = np.log(X_airfoil[:, 0])
        X_airfoil[:, 4] = np.log(X_airfoil[:, 4])
        Y_airfoil = airfoil.iloc[:, 5].values
        n_airfoil = len(Y_airfoil)
        print("X_airfoil shape : ", X_airfoil.shape)
        
    elif (dataset == 'wine'):
        winequality_red = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'SCS_datasets/wine/winequality-red.csv', sep=';')
        X_wine = winequality_red.iloc[:, 0:11].values
        Y_wine = winequality_red.iloc[:, 11].values
        n_wine = len(Y_wine)
        print("X_wine shape : ", X_wine.shape)
        
    elif (dataset == 'wave'):
        wave = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'SCS_datasets/WECs_DataSet/Adelaide_Data.csv', header = None)
        X_wave = wave.iloc[0:2000, 0:48].values
        Y_wave = wave.iloc[0:2000, 48].values
        n_wave = len(Y_wave)
        print("X_wave shape : ", X_wave.shape)
        
    elif (dataset == 'superconduct'):
        superconduct = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'SCS_datasets/superconduct/train.csv')
        X_superconduct = superconduct.iloc[0:2000, 0:81].values
        Y_superconduct = superconduct.iloc[0:2000, 81].values
        n_superconduct = len(Y_superconduct)
        print("X_superconduct shape : ", X_superconduct.shape)
        
    elif (dataset == 'communities'):
        # UCI Communities and Crime Data Set
        # download from:
        # http://archive.ics.uci.edu/ml/datasets/communities+and+crime
        communities_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + 'SCS_datasets/communities/communities.data',delimiter=',',dtype=str)
        # remove categorical predictors
        communities_data = np.delete(communities_data,np.arange(5),1)
        # remove predictors with missing values
        communities_data = np.delete(communities_data,\
                    np.argwhere((communities_data=='?').sum(0)>0).reshape(-1),1)
        communities_data = communities_data.astype(float)
        X_communities = communities_data[:,:-1]
        Y_communities = communities_data[:,-1]
        n_communities = len(Y_communities)
        print("X_communities shape : ", X_communities.shape)
        
    else:
        raise Exception("Invalid dataset name")
        
        
    if (muh_fun_name in ['RF', 'random_forest']):
        muh_fun = random_forest
        
    elif (muh_fun_name in ['NN', 'neural_net']):
        muh_fun = neural_net
        
    elif (muh_fun_name in ['RR', 'leastsq_ridge']):
        muh_fun = leastsq_ridge



    if (n >= eval('n_'+dataset)):
        raise Exception("Error: number of training datapoints is greater than total number of datapoints")

    if (run_effective_sample_size):
        method_names = ['naive','jackknife','jackknife+','jackknife-mm'] ## EFF
    else:
        method_names = ['naive','jackknife','jackknife+','jackknife-mm', 'split', 
                        'weighted_split_oracle', 'weighted_split_lr', 'weighted_split_rf',
                        'JAW_oracle', 'JAW_lr', 'JAW_rf', 'JAWmm_oracle', 'JAWmm_lr', 'JAWmm_rf']
        K_based_method_base_names = ['CV+_K', 'wCV_oracle_K', 'wCV_lr_K', 'wCV_rf_K','JAW_KLOO_oracle_K', 'JAW_KLOO_lr_K', 'JAW_KLOO_rf_K','JAW_KLOO_oracle_det_K', 'JAW_KLOO_lr_det_K', 'JAW_KLOO_rf_det_K']
        for K in K_vals:
            method_names = np.concatenate([method_names, [K_base_name + str(K) for K_base_name in K_based_method_base_names]])
                                      
        # ,

    results_by_iter = pd.DataFrame(columns = ['itrial','dataset','muh_fun','method','coverage','width'])
    results_all = pd.DataFrame(columns = ['itrial','test_pt','dataset','muh_fun','method','coverage','width'])

    
    print("Running dataset " + dataset + ", with muh fun " + muh_fun_name + ", with bias " + str(bias) + ", for ntrial" + str(ntrial))

    for itrial in range(ntrial):
        np.random.seed(itrial)
        print("Trial # = ", itrial)
        train_inds = np.random.choice(eval('n_'+dataset),n,replace=False)
        test_inds = np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds)

        X = eval('X_'+dataset)[train_inds]
        Y = eval('Y_'+dataset)[train_inds]
        X1 = eval('X_'+dataset)[test_inds]
        Y1 = eval('Y_'+dataset)[test_inds]

        X1_unshifted = X1
        n1=len(X1)

        biased_test_indices = exponential_tilting_indices(X, X1, dataset, bias=bias)

        ## Bias the test data if bias != 0 and not running effective sample size expts
        if (bias != 0 and not run_effective_sample_size):
            ## Test data exponential tilting indices
            X1 = X1[biased_test_indices] ## Apply shift to X1
            Y1 = Y1[biased_test_indices] ## Apply shift to Y1

        X_full = np.concatenate((X, X1), axis = 0)
        
        ## Oracle weights:
        weights_oracle = get_w(X, X_full, dataset, bias).reshape(len(X_full)) 
            
        ## Estimated weights (logistic regression and random forest)
        source_target_labels = np.concatenate([np.zeros(len(X)), np.ones(len(X1))])
        weights_lr = logistic_regression_weight_est(X_full, source_target_labels)
        weights_rf = random_forest_weight_est(X_full, source_target_labels)
        

            
        PIs = compute_PIs(X,Y,X1,alpha,muh_fun, weights_oracle, dataset, bias, run_effective_sample_size, weights_lr, weights_rf, K_vals)
        for method in method_names:
            coverage_by_iter = ((PIs[method]['lower'] <= Y1)&(PIs[method]['upper'] >= Y1)).mean()
            coverage_all = ((PIs[method]['lower'] <= Y1)&(PIs[method]['upper'] >= Y1))
            width_by_iter = (PIs[method]['upper'] - PIs[method]['lower']).median()
            width_all = (PIs[method]['upper'] - PIs[method]['lower'])
            
            results_by_iter.loc[len(results_by_iter)]=\
            [itrial,dataset,muh_fun.__name__,method,coverage_by_iter,width_by_iter]
            
#             print(n1, len(coverage_all))
            for test_pt in range(0, len(coverage_all)):
                results_all.loc[len(results_all) + test_pt]=[itrial,test_pt,dataset,muh_fun.__name__,method,\
                                                             coverage_all[test_pt],width_all[test_pt]]


        if ((itrial + 1) % 5 == 0):
            results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_' + dataset + '_' + muh_fun_name + '_' + str(bias) + 'Bias_' + str(itrial + 1) +'Trials_ALL.csv',index=False)
            results_by_iter.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_' + dataset + '_' + muh_fun_name + '_' + str(bias) + 'Bias_' + str(itrial + 1) +'Trials_byIters.csv',index=False)
            
    results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_' + dataset + '_' + muh_fun_name + '_' + str(bias) + 'Bias_' + str(itrial + 1) +'Trials_ALL.csv',index=False)
    results_by_iter.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_' + dataset + '_' + muh_fun_name + '_' + str(bias) + 'Bias_' + str(itrial + 1) +'Trials_byIters.csv',index=False)
    
