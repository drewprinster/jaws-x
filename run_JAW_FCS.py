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


if __name__ == "__main__":
    ### Running JAW with neural network predictor
    parser = argparse.ArgumentParser(description='Run JAW FCS experiments.')
    
    parser.add_argument('--fitness_str', type=str, default='red', help='Red or blue fluorescence experiments.')
    parser.add_argument('--n_trains', nargs='+', help='Values of n_train to try', required = True)
    parser.add_argument('--lmbdas', nargs='+', help='Values of lmbda to try', required = True)
    parser.add_argument('--n_seed', type=int, default=1000, help='Number of trials')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value corresponding to 1-alpha target coverage')
    parser.add_argument('--K_vals', nargs='+', help='Values of K to try', required = True)
    parser.add_argument('--muh', type=str, default='NN', help='Muh predictor.')
    
    ## python run_JAW_FCS.py --fitness_str red --n_trains 96 192 --lmbdas 0 2 4 6 --n_seed 1000 --K_vals 8 12 16 24 32 48
    
    args = parser.parse_args()
    fitness_str = args.fitness_str
    n_trains = [int(n_train) for n_train in args.n_trains]
    lmbdas = [int(lmbda) for lmbda in args.lmbdas]
    n_seed = args.n_seed
    n_seed = 20 ## For now override to 50 trials
    alpha = args.alpha
    K_vals = [int(K) for K in args.K_vals]
    muh = args.muh


    reload(cal)
    reload(assay)

    alpha = 0.1                           # miscoverage
#     n_trains = [96, 192, 384]    # 96, 192,          # number of training points
    ntrain2reg = {96: 10, 192: 1, 384: 1} # ridge regularization strength (gamma in code and paper)
#     n_seed = 1000                       # number of random trials  ## Drew changed this from 2000
#     lmbdas = [0, 2, 4, 6]  # 0, 2,                # lambda, inverse temperature
    y_increment = 0.02                    # grid spacing between candidate label values, \Delta in paper
    ys = np.arange(0, 2.21, y_increment)  # candidate label values, \mathcal{Y} in paper
    order = 2                             # complexity of features. 1 encodes the AA at each site,
                                          # 2 the AAs at each pair of sites,
                                          # 3 the AAs at each set of 3 sites, etc.
    n1 = 200 # Number of test points
    if (muh == 'NN'):
        muh_fun = MLPRegressor(solver='lbfgs',activation='logistic')
    elif (muh == 'RF'):
        muh_fun = RandomForestRegressor(n_estimators=20,criterion='absolute_error')
        
    method_names = ['split', 'weighted_split', 'JAW-FCS', 'JAW-SCS', 'jackknife+']

#     K_vals = [8, 12, 16, 24, 32, 48]
    K_based_method_base_names = ['CV+_K', 'wCV_FCS_K', 'wCV_SCS_K', 'JAW_FCS_KLOO_rep_K', 'JAW_SCS_KLOO_rep_K', 'JAW_FCS_KLOO_det_K', 'JAW_SCS_KLOO_det_K']
    for K in K_vals:
        method_names = np.concatenate([method_names, [K_base_name + str(K) for K_base_name in K_based_method_base_names]])

    results_by_seed = pd.DataFrame(columns = ['seed','fitness_str','muh_fun','method','coverage','width', 'muh_test'])
    results_all = pd.DataFrame(columns = ['seed','test_pt', 'fitness_str','muh_fun','method','coverage','width', 'muh_test'])

    # likelihood under training input distribution, p_X in paper (uniform distribution)
    ptrain_fn = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]])
    ptrain_fn_pointwise = lambda x: (1.0 / np.power(2, 13)) * np.ones([x.shape[0]])

    # featurize all sequences in combinatorially complete dataset
    data = assay.PoelwijkData(fitness_str, order=order)

    for t, n_train in enumerate(n_trains):

        reg = ntrain2reg[n_train]
#         fcs = cal.ConformalRidgeFeedbackCovariateShift(ptrain_fn, ys, data.X_nxp, reg)
#         jaw_fcs = cal.JAWRidgeFeedbackCovariateShift(ptrain_fn, data.X_nxp, reg)
        jaw_fcs_nn = cal.JAWFeedbackCovariateShift(muh_fun, ptrain_fn, data.X_nxp)
#         scs = cal.ConformalRidgeStandardCovariateShift(ptrain_fn, ys, data.X_nxp, reg)

        for l, lmbda in enumerate(lmbdas):

            fset_s, sset_s, jaw_fset_s, jaw_fset_nn_s = [], [], [], [] # jaw_fset_s
            fcov_s, scov_s, jaw_fcov_s, jaw_fcov_nn_s = np.zeros([n_seed]), np.zeros([n_seed]), np.zeros([n_seed]), np.zeros([n_seed*n1])
            ytest_s, predtest_s = np.zeros([n_seed, n1]), np.zeros([n_seed, n1])
            t0 = time.time()

            for seed in range(n_seed):

                # sample training and designed data
                Xtrain_nxp, ytrain_n, Xtest_n1xp, ytest_n1, pred_train_test, \
                Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_n1xp_split, ytest_n1_split, pred_cal_test_split \
                = assay.get_training_and_designed_data_my_muh(data, n_train, reg, lmbda, seed=seed, n1=n1, replace=True, muh_name=muh)
#                 Xtrain_nxp, ytrain_n, Xtest_n1xp, ytest_n1, pred_n1 = assay.get_training_and_designed_data(
#                     data, n_train, reg, lmbda, seed=seed, n1=n1, replace=True)
#                 print("Xcal_split", Xcal_split)
#                 print("ycal_split", ycal_split)
                
                ytest_s[seed] = ytest_n1
                predtest_s[seed] = pred_train_test[-n1:]

                # construct confidence interval with JAW under feedback covariate shift
                PIs = jaw_fcs_nn.compute_PIs(Xtrain_nxp, ytrain_n, Xtest_n1xp, ytest_n1, pred_train_test, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_n1xp_split, ytest_n1_split, pred_cal_test_split, lmbda, alpha=alpha, K_vals = K_vals)
                for method in method_names:
                    if (method not in ['split', 'weighted_split']):
                        coverage_by_seed = ((PIs[method]['lower'] <= ytest_n1)&(PIs[method]['upper'] >= ytest_n1)).mean()
                        muh_test_by_seed = pred_train_test[-n1:].mean()
                        coverage_all = ((PIs[method]['lower'] <= ytest_n1)&(PIs[method]['upper'] >= ytest_n1))
                        muh_test_all = pred_train_test[-n1:]
                    else:
#                         print(len(PIs[method]))
#                         print(PIs[method]['lower'][0:10], ytest_n1_split[0:10], PIs[method]['upper'][0:10])
                        coverage_by_seed = ((PIs[method]['lower'] <= ytest_n1_split)&(PIs[method]['upper'] >= ytest_n1_split)).mean()
                        muh_test_by_seed = pred_cal_test_split[-n1:].mean()
                        coverage_all = ((PIs[method]['lower'] <= ytest_n1_split)&(PIs[method]['upper'] >= ytest_n1_split))
                        muh_test_all = pred_cal_test_split[-n1:]
        
                    width_by_seed = (PIs[method]['upper'] - PIs[method]['lower']).median()
                    width_all = (PIs[method]['upper'] - PIs[method]['lower'])
                    
                    results_by_seed.loc[len(results_by_seed)]=\
                    [seed,fitness_str,muh,method,coverage_by_seed,width_by_seed,muh_test_by_seed]
                    
                    print(coverage_all)
                    print(type(coverage_all))
                    print(n1, len(coverage_all), len(muh_test_all))
                    for test_pt in range(0, n1):
                        results_all.loc[len(results_all) + test_pt]=[seed,test_pt,fitness_str,muh,method,\
                                                                   coverage_all[test_pt],width_all[test_pt],muh_test_all[test_pt]]

                if ((seed + 1) == 5 or (seed + 1) == 10 or (seed + 1) == 20 or (seed + 1) == 50):
                    results_by_seed.to_csv(str(date.today()) + '_' + fitness_str + '_' + muh + '_ntrain' +  str(n_train) + '_lmbda' + str(lmbda) + '_seed' + str(seed + 1) + '_PIs_results_BySeed_v2.csv',index=False)

results_by_seed.to_csv(str(date.today()) + '_' + fitness_str + '_' + muh + '_ntrain' +  str(n_train) + '_lmbda' + str(lmbda) + '_seed' + str(seed + 1) + '_PIs_results_BySeed.csv',index=False)

# if (n_trains[0] == 192):
results_all.to_csv(str(date.today()) + '_' + fitness_str + '_' + muh + '_ntrain' +  str(n_train) + '_lmbda' + str(lmbda) + '_seed' + str(seed + 1) + '_PIs_results_ALL.csv',index=False)
