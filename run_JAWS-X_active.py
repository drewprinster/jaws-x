import os
import sys
import time
from importlib import reload
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
    
import assay
import calibrate as cal

## Drew added
import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import argparse
from datetime import date

## Added for active learning experiments
from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF

## test
if __name__ == "__main__":
    start_time = time.time()
    
    ### Running JAW with neural network predictor
    parser = argparse.ArgumentParser(description='Run JAW FCS experiments.')
    
    
#     parser.add_argument('--fitness_str', type=str, default='red', help='Red or blue fluorescence experiments.')
    parser.add_argument('--n_train_initial', type=int, default=64, help='Initial number of training points')
    parser.add_argument('--n_val', type=int, default=800, help='Number of validation points')
    parser.add_argument('--n_steps', type=int, default=8, help='Number of active learning steps')
    parser.add_argument('--n_queries_ann', type=int, default=16, help='Number of queries to annotate')
    parser.add_argument('--n_queries_cov', type=int, default=20, help='Number of queries for evaluating coverage')
    parser.add_argument('--n_seed', type=int, default=1, help='Number of trials')
    parser.add_argument('--seed_initial', type=int, default=0, help='Initial seed')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
    parser.add_argument('--K_vals', nargs='+', help='Values of K to try', required = True)
    parser.add_argument('--muh', type=str, default='GP', help='Muh predictor.')
    parser.add_argument('--dataset', type=str, default='airfoil', help='Dataset name')
    
    ## python run_JAW_FCS_active.py --dataset airfoil --n_steps 10 --K_vals 16
    
   
    
    args = parser.parse_args()
    n_train_initial = args.n_train_initial
    n_val = args.n_val
    n_steps = args.n_steps
    n_queries_ann = args.n_queries_ann
    n_queries_cov = args.n_queries_cov
    n_seed = args.n_seed
    alpha = args.alpha
    K_vals = [int(K) for K in args.K_vals]
    muh = args.muh
    dataset = args.dataset
    seed_initial = args.seed_initial
    
    n_train_initial = 32 ## OVERRIDE
    n_steps = 8
    
    print("Running with n_seed ", str(n_seed), "n_steps ", str(n_steps), "n_queries_cov ", str(n_queries_cov))
    
    if (muh == 'GP'):
        kernel = DotProduct() + WhiteKernel()
        muh_fun = GaussianProcessRegressor(kernel=kernel,random_state=0)

    method_names = ['split', 'weighted_split', 'JAW-FCS', 'JAW-SCS', 'jackknife+']

#     K_vals = [8, 12, 16, 24, 32, 48]
    K_based_method_base_names = ['CV+_K', 'wCV_FCS_K', 'wCV_SCS_K', 'JAW_FCS_KLOO_rep_K', 'JAW_SCS_KLOO_rep_K', 'JAW_FCS_KLOO_det_K', 'JAW_SCS_KLOO_det_K']
    for K in K_vals:
        method_names = np.concatenate([method_names, [K_base_name + str(K) for K_base_name in K_based_method_base_names]])

    results_by_seed = pd.DataFrame(columns = ['seed', 'step', 'dataset', 'muh_fun','method','coverage','width', 'MSE'])
    results_all = pd.DataFrame(columns = ['seed','step', 'test_pt', 'dataset','muh_fun','method','coverage','width', 'muh_test', 'y_test'])

#     # likelihood under training input distribution, p_X in paper (uniform distribution)
    ptrain_fn = cal.KDE_density_estimates
#     ptrain_fn_pointwise = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]])

    # Read dataset
    if (dataset == 'airfoil'):
        airfoil = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'AL_datasets/airfoil.txt', sep = '\t', header=None)
        airfoil.columns = ["Frequency","Angle","Chord","Velocity","Suction","Sound"]
        X_airfoil = airfoil.iloc[:, 0:5].values
        X_airfoil[:, 0] = np.log(X_airfoil[:, 0])
        X_airfoil[:, 4] = np.log(X_airfoil[:, 4])
        Y_airfoil = airfoil.iloc[:, 5].values
        n_airfoil = len(Y_airfoil)
        
    elif (dataset == 'wine'):
        winequality_red = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'AL_datasets/wine/winequality-red.csv', sep=';')
        X_wine = winequality_red.iloc[:, 0:11].values
        Y_wine = winequality_red.iloc[:, 11].values
        n_wine = len(Y_wine)
        print("X_wine shape : ", X_wine.shape)
        
    elif (dataset == 'wave'):
        wave = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'AL_datasets/WECs_DataSet/Adelaide_Data.csv', header = None)
        X_wave = wave.iloc[0:2000, 0:48].values
        Y_wave = wave.iloc[0:2000, 48].values
        n_wave = len(Y_wave)
        print("X_wave shape : ", X_wave.shape)
        
    elif (dataset == 'superconduct'):
        superconduct = pd.read_csv(os.getcwd().removesuffix('bash_scripts') + 'AL_datasets/superconduct/train.csv')
        X_superconduct = superconduct.iloc[0:2000, 0:81].values
        Y_superconduct = superconduct.iloc[0:2000, 81].values
        n_superconduct = len(Y_superconduct)
        print("X_superconduct shape : ", X_superconduct.shape)
        
    elif (dataset == 'communities'):
        # UCI Communities and Crime Data Set
        # download from:
        # http://archive.ics.uci.edu/ml/datasets/communities+and+crime
        communities_data = np.loadtxt(os.getcwd().removesuffix('bash_scripts') + 'AL_datasets/communities/communities.data',delimiter=',',dtype=str)
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
    
    X_all = eval('X_'+dataset)
    all_inds = np.arange(eval('n_'+dataset))

    jaw_fcs_active = cal.JAWFeedbackCovariateShiftActive(muh_fun, ptrain_fn, X_all) 


#     fset_s, sset_s, jaw_fset_s, jaw_fset_nn_s = [], [], [], [] # jaw_fset_s
#     fcov_s, scov_s, jaw_fcov_s, jaw_fcov_nn_s = np.zeros([n_seed]), np.zeros([n_seed]), np.zeros([n_seed]), np.zeros([n_seed*n1])
#     ytest_s, predtest_s = np.zeros([n_seed, n1]), np.zeros([n_seed, n1])

    for seed in range(seed_initial, seed_initial + n_seed):
        ## Initial random data splits (train, validation, and pool)
        ## Note: Validation set won't change, train and pool will
        np.random.seed(seed)
        train_inds = list(np.random.choice(eval('n_'+dataset),n_train_initial,replace=False))
        val_inds = list(np.random.choice(np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds), n_val, replace=False))
        pool_inds = list(np.setdiff1d(np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds), val_inds))
        
        ## Initialize train and pool data for no sample splitting
        Xtrain = eval('X_'+dataset)[train_inds]
        ytrain = eval('Y_'+dataset)[train_inds]
        Xpool = eval('X_'+dataset)[pool_inds]
        ypool = eval('Y_'+dataset)[pool_inds]
        
        ## Create validation set (won't change)
        Xval = eval('X_'+dataset)[val_inds]
        yval = eval('Y_'+dataset)[val_inds]
        
        ## Sample splitting indices
        idx_split = list(np.random.permutation(n_train_initial))
        n_half_initial = int(np.floor(n_train_initial/4))
        train_inds_split, cal_inds_split = list(idx_split[:n_half_initial]), list(idx_split[n_half_initial:])
        ## Note: Calibration set for split won't change
        Xtrain_split = eval('X_'+dataset)[train_inds_split]
        ytrain_split = eval('Y_'+dataset)[train_inds_split]
        Xcal_split = eval('X_'+dataset)[cal_inds_split]
        ycal_split = eval('Y_'+dataset)[cal_inds_split]
        ## Pool inds for split are initially the same but will be different later
        pool_inds_split = list(np.setdiff1d(np.setdiff1d(np.arange(eval('n_'+dataset)),train_inds), val_inds))
        
        ## Initialize train and pool data for sample splitting (will change)
        Xpool_split = eval('X_'+dataset)[pool_inds_split]
        ypool_split = eval('Y_'+dataset)[pool_inds_split]
        
        
        
        ## Iterate through active learning steps
        for step in range(n_steps):
            
            ####### ******* No sample splitting ********* ########
            ## Fit Gaussian process regression and use it to select queries from pool
            gpr = muh_fun.fit(Xtrain, ytrain)
            
            ## Query point(s) for annotation from pool based on max predicted variance (max entropy)
            y_preds_pool, std_preds_pool = gpr.predict(Xpool, return_std=True) ## Predictions on pool
            var_preds_pool = std_preds_pool**2
            var_preds_pool_norm = var_preds_pool / np.sum(var_preds_pool)
            ####NOTE: Changed this from max variance to sampling in proportion to variance
            query_ann_inds = list(np.random.choice(pool_inds, n_queries_ann, replace=False, p=var_preds_pool_norm)) 
            # query_ann_inds = list(np.argpartition(var_preds_pool,-n_queries_ann)[-n_queries_ann:]) 
            
            ## Query points for coverage evaluation from validation set by sampling in proportion to variance
            y_preds_val, std_preds_val = gpr.predict(Xval, return_std=True)
            var_preds_val = std_preds_val**2
            var_preds_val_norm = var_preds_val / np.sum(var_preds_val)
            query_cov_inds = list(np.random.choice(val_inds, n_queries_cov, p=var_preds_val_norm)) 
            ## Can view these as samples from the test distribution (in terms of coverage evaluation)
            Xtest = eval('X_'+dataset)[query_cov_inds]
            ytest = eval('Y_'+dataset)[query_cov_inds]
            n_test = len(ytest)
            ytest_preds = gpr.predict(Xtest, return_std=False)
            
            ## Prepare for next active learning iteration: 
                ## Add point that was queried for annotation to the training data 
                ## & remove queried point from pooled data
            for q_ann in query_ann_inds:
                train_inds.append(q_ann) ## Add queried samples to training set
            pool_inds = list(set(all_inds) - set(train_inds))
            
            ## Update train and pool data for no sample splitting
            Xtrain = eval('X_'+dataset)[train_inds]
            ytrain = eval('Y_'+dataset)[train_inds]
            Xpool = eval('X_'+dataset)[pool_inds]
            ypool = eval('Y_'+dataset)[pool_inds]
            
            
            MSE_full = np.mean((y_preds_val - yval)**2)
            
        
        
            ####### ******* Sample splitting ********* ########
            ## Fit Gaussian process regression and use it to select queries from pool
            gpr_split = muh_fun.fit(Xtrain_split, ytrain_split)
            
            ## Query point(s) for annotation from pool based on max predicted variance (max entropy)
            y_preds_pool_split, std_preds_pool_split = gpr_split.predict(Xpool_split, return_std=True) ## Predictions on pool
            var_preds_pool_split = std_preds_pool_split**2
            var_preds_pool_norm_split = var_preds_pool_split / np.sum(var_preds_pool_split)
            ####NOTE: Changed this from max variance to sampling in proportion to variance
            query_ann_inds_split = list(np.random.choice(pool_inds_split, n_queries_ann, replace=False, p=var_preds_pool_norm_split)) 
            # query_ann_inds_split = list(np.argpartition(var_preds_pool_split,-n_queries_ann)[-n_queries_ann:]) 
            
            ## Query points for coverage evaluation from validation set by sampling in proportion to variance
            y_preds_val_split, std_preds_val_split = gpr_split.predict(Xval, return_std=True)
            var_preds_val_split = std_preds_val_split**2
            var_preds_val_norm_split = var_preds_val_split / np.sum(var_preds_val_split)
            query_cov_inds_split = list(np.random.choice(val_inds, n_queries_cov, p=var_preds_val_norm_split)) 
            ## Can view these as samples from the test distribution (in terms of coverage evaluation)
            Xtest_split = eval('X_'+dataset)[query_cov_inds_split]
            ytest_split = eval('Y_'+dataset)[query_cov_inds_split]
            ytest_preds_split = gpr_split.predict(Xtest_split, return_std=False)
            
            ## Prepare for next active learning iteration: 
                ## Add point that was queried for annotation to the training data 
                ## & remove queried point from pooled data
            for q_ann in query_ann_inds_split:
                train_inds_split.append(q_ann) ## Add queried samples to training set
            pool_inds_split = list(set(all_inds) - set(train_inds_split))
            
#             ## Update train and pool data for sample splitting
#             idx_split = list(np.random.permutation(train_inds_split + cal_inds_split))
#             n_half_ = int(np.floor(len(idx_split)/2))
#             train_inds_split, cal_inds_split = list(idx_split[:n_half_]), list(idx_split[n_half_:])
#             ## Note: Calibration set for split won't change
#             Xtrain_split = eval('X_'+dataset)[train_inds_split]
#             ytrain_split = eval('Y_'+dataset)[train_inds_split]
#             Xcal_split = eval('X_'+dataset)[cal_inds_split]
#             ycal_split = eval('Y_'+dataset)[cal_inds_split]
        
            Xtrain_split = eval('X_'+dataset)[train_inds_split]
            ytrain_split = eval('Y_'+dataset)[train_inds_split]
            Xpool_split = eval('X_'+dataset)[pool_inds_split]
            ypool_split = eval('Y_'+dataset)[pool_inds_split]
            
            MSE_split = np.mean((y_preds_val_split - yval)**2)
            
            
#             for method in ['no splitting', 'split']:
#                 if (method not in ['split', 'weighted_split']):
#                     results_by_seed.loc[len(results_by_seed)]=\
#                         [seed,step, dataset, muh, method,'NA_coverage','NA_width',MSE_full]
#                 else:
#                     results_by_seed.loc[len(results_by_seed)]=\
#                         [seed,step, dataset, muh, method,'NA_coverage','NA_width',MSE_split]
                        
            # construct confidence interval with JAW methods under feedback covariate shift
            PIs = jaw_fcs_active.compute_PIs_active(Xtrain, ytrain, Xtest, ytest, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_split, ytest_split, bandwidth = 1.0, alpha=alpha, K_vals = K_vals)
            
            
            for method in method_names:
                if (method not in ['split', 'weighted_split']):
                    coverage_by_seed = ((PIs[method]['lower'] <= ytest)&(PIs[method]['upper'] >= ytest)).mean()
                    muh_test_by_seed = ytest_preds.mean()
                    coverage_all = ((PIs[method]['lower'] <= ytest)&(PIs[method]['upper'] >= ytest))
                    muh_test_all = ytest_preds
                    ytest_method = ytest
                    MSE = MSE_full
                else:
    #                         print(len(PIs[method]))
    #                         print(PIs[method]['lower'][0:10], ytest_n1_split[0:10], PIs[method]['upper'][0:10])
                    coverage_by_seed = ((PIs[method]['lower'] <= ytest_split)&(PIs[method]['upper'] >= ytest_split)).mean()
                    muh_test_by_seed = ytest_preds_split.mean()
                    coverage_all = ((PIs[method]['lower'] <= ytest_split)&(PIs[method]['upper'] >= ytest_split))
                    muh_test_all = ytest_preds_split
                    ytest_method = ytest_split
                    MSE = MSE_split

                width_by_seed = (PIs[method]['upper'] - PIs[method]['lower']).median()
                width_all = (PIs[method]['upper'] - PIs[method]['lower'])

#                 results_by_seed.loc[len(results_by_seed)]=\
#                 [seed,fitness_str,muh,method,coverage_by_seed,width_by_seed,muh_test_by_seed]
                results_by_seed.loc[len(results_by_seed)]=\
                        [seed,step, dataset, muh, method,coverage_by_seed,width_by_seed,MSE]
                
#                 print(coverage_all)
#                 print(type(coverage_all))
#                 print(n1, len(coverage_all), len(muh_test_all))
                for test_pt in range(0, n_test):
                    results_all.loc[len(results_all) + test_pt]=[seed,step, test_pt,dataset,muh,method,\
                                                               coverage_all[test_pt],width_all[test_pt],\
                                                                 muh_test_all[test_pt], ytest_method[test_pt]]
        if (((seed+1) % 5) == 0):
            results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_ActiveLearningExpts_' + dataset + '_' + muh + '_itrain' + str(n_train_initial) + '_steps' + str(step + 1) + '_nseed' + str(n_seed) + '_iseed' + str(seed_initial) + '_PIs_results_BySeed_v2.csv',index=False)

    end_time = time.time()
    
    print("Total time (minutes) : ", (end_time - start_time)/60)

    results_by_seed.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_ActiveLearningExpts_' + dataset + '_' + muh + '_itrain' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_iseed' + str(seed_initial) + '_PIs_results_BySeed_v2.csv',index=False)
    
    results_all.to_csv(os.getcwd().removesuffix('bash_scripts') + 'results/'+ str(date.today()) + '_ActiveLearningExpts_' + dataset + '_' + muh + '_itrain' + str(n_train_initial) + '_steps' + str(n_steps) + '_nseed' + str(n_seed) + '_iseed' + str(seed_initial) + '_PIs_results_ALL_v2.csv',index=False)

# # if (n_trains[0] == 192):
# results_all.to_csv(str(date.today()) + '_' + fitness_str + '_' + muh + '_ntrain' +  str(n_train) + '_lmbda' + str(lmbda) + '_seed' + str(seed + 1) + '_PIs_results_ALL.csv',index=False)
