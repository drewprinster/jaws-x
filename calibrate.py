"""
Classes for full conformal prediction for exchangeable, standard, and feedback covariate shift data,
both for black-box predictive models and computationally optimized for ridge regression, and
functions for (random, exact-coverage) split conformal prediction under standard covariate shift.
Throughout this file, variable name suffixes denote the shape of the numpy array, where
    n: number of training points, or generic number of data points
    n1: n + 1
    p: number of features
    y: number of candidate labels, |Y|
    u: number of sequences in domain, |X|
    m: number of held-out calibration points for split conformal methods
"""

import numpy as np
import time
import scipy as sc

from abc import ABC, abstractmethod

## Drew added
import math
import pandas as pd
import random

# ===== utilities for split conformal =====

def get_split_coverage(lu_nx2, y_n):
    """
    Computes empirical coverage of split conformal confidence interval
    :param lu_nx2: (n, 2) numpy array where first and second columns are lower and upper endpoints
    :param y_n: (n,) numpy array of true labels
    :return: float, empirical coverage
    """
    cov = np.sum((y_n >= lu_nx2[:, 0]) & (y_n <= lu_nx2[:, 1])) / y_n.size
    return cov

def get_randomized_staircase_coverage(C_n, y_n):
    """
    Computes empirical coverage and lengths of randomized staircase confidence sets.

    :param C_n: length-n list of outputs of get_randomized_staircase_confidence_set (i.e., list of tuples)
    :param y_n: (n,) numpy array of true labels
    :return: (n,) binary array of coverage and (n,) numpy array of lengths
    """
    def is_covered(confint_list, y):
        for confint_2 in confint_list:
            if y >= confint_2[0] and y <= confint_2[1]:
                return True
        return False
    def get_len_conf_set(confint_list):
        return np.sum([confint_2[1] - confint_2[0] for confint_2 in confint_list])

    cov_n = np.array([is_covered(confset, y) for confset, y in zip(C_n, y_n)])
    len_n = np.array([get_len_conf_set(confset) for confset in C_n])
    return cov_n, len_n

def get_randomized_staircase_confidence_set(scores_m, weights_m1, predtest, alpha: float = 0.1):
    """
    Computes the "randomized staircase" confidence set in Alg. S1.

    :param scores_m: (m,) numpy array of calibration scores
    :param weights_m1: (m + 1) numpy array of calibration weights and single test weight
    :param predtest: float, prediction on test input
    :param alpha: miscoverage level
    :return: list of tuples (l, u), where l and u are floats denoting lower and upper
        endpoints of an interval.
    """
    lb_is_set = False
    idx = np.argsort(scores_m)
    sortedscores_m1 = np.hstack([0, scores_m[idx]])
    sortedweights_m1 = np.hstack([0, weights_m1[: -1][idx]])
    C = []

    # interval that is deterministically included in the confidence set
    # (color-coded green in Fig. S1)
    cdf_m1 = np.cumsum(sortedweights_m1) # CDF up to i-th sorted calibration score
    cdf_plus_test_weight_m1 = cdf_m1 + weights_m1[-1]
    deterministic_idx = np.where(cdf_plus_test_weight_m1 < 1 - alpha)[0]
    if deterministic_idx.size:
        i_det = np.max(deterministic_idx)
        C.append((predtest - sortedscores_m1[i_det + 1], predtest + sortedscores_m1[i_det + 1]))

    # intervals that are randomly included in the confidence set
    # (color-coded teal and blue in Fig. S1)
    for i in range(i_det + 1, sortedscores_m1.size - 1):
        assert(cdf_plus_test_weight_m1[i] >= 1 - alpha)
        if cdf_plus_test_weight_m1[i] >= 1 - alpha and cdf_m1[i] < 1 - alpha:
            if not lb_is_set:
                lb_is_set = True
                LF = cdf_m1[i]
            F = (cdf_plus_test_weight_m1[i] - (1 - alpha)) / (cdf_m1[i] + weights_m1[-1] - LF)
            if sc.stats.bernoulli.rvs(1 - F):
                C.append((predtest + sortedscores_m1[i], predtest + sortedscores_m1[i + 1]))
                C.append((predtest - sortedscores_m1[i + 1], predtest - sortedscores_m1[i]))

    # halfspaces that are randomly included in the confidence set
    # (color-coded purple in Fig. S1)
    if cdf_m1[-1] < 1 - alpha:  # sum of all calibration weights
        if not lb_is_set:
            LF = cdf_m1[-1]
        F = alpha / (1 - LF)
        if sc.stats.bernoulli.rvs(1 - F):
            C.append((predtest + sortedscores_m1[-1], np.inf))
            C.append((-np.inf, predtest - sortedscores_m1[-1]))
    return C



# ========== full conformal utilities ==========

def get_weighted_quantile(quantile, w_n1xy, scores_n1xy):
    """
    Compute the quantile of weighted scores for each candidate label y

    :param quantile: float, quantile
    :param w_n1xy: (n + 1, |Y|) numpy array of weights (unnormalized)
    :param scores_n1xy: (n + 1, |Y|) numpy array of scores
    :return: (|Y|,) numpy array of quantiles
    """
    if w_n1xy.ndim == 1:
        w_n1xy = w_n1xy[:, None]
        scores_n1xy = scores_n1xy[:, None]

    # normalize probabilities
    p_n1xy = w_n1xy / np.sum(w_n1xy, axis=0)

    # sort scores and their weights accordingly
    sorter_per_y_n1xy = np.argsort(scores_n1xy, axis=0)
    sortedscores_n1xy = np.take_along_axis(scores_n1xy, sorter_per_y_n1xy, axis=0)
    sortedp_n1xy = np.take_along_axis(p_n1xy, sorter_per_y_n1xy, axis=0)

    # locate quantiles of weighted scores per y
    cdf_n1xy = np.cumsum(sortedp_n1xy, axis=0)
    qidx_y = np.sum(cdf_n1xy < quantile, axis=0)  # equivalent to [np.searchsorted(cdf_n1, q) for cdf_n1 in cdf_n1xy]
    q_y = sortedscores_n1xy[(qidx_y, range(qidx_y.size))]
    return q_y

def is_covered(y, confset, y_increment):
    """
    Return if confidence set covers true label

    :param y: true label
    :param confset: numpy array of values in confidence set
    :param y_increment: float, \Delta increment between candidate label values, 0.01 in main paper
    :return: bool
    """
    return np.any(np.abs(y - confset) < (y_increment / 2))

# ========== JAW utilities ==========

def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]
    
    return [v_sorted, w_sorted]
    

def weighted_quantile(v, w_normalized, q):
    if (len(v) != len(w_normalized)):
        raise ValueError('Error: v is length ' + str(len(v)) + ', but w_normalized is length ' + str(len(w_normalized)))
        
    if (np.sum(w_normalized) > 1.01 or np.sum(w_normalized) < 0.99):
#         print(np.sum(w_normalized))
#         print(w_normalized)
        raise ValueError('Error: w_normalized does not add to 1')
        
    if (q < 0 or 1 < q):
        raise ValueError('Error: Invalid q')

    n = len(v)
    
    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)
    
    w_sorted_cum = np.cumsum(w_sorted)
    
#     cum_w_sum = w_sorted[0]
    i = 0
    while(w_sorted_cum[i] < q):
        i += 1
#         cum_w_sum += w_sorted[i]
        
    
            
    if (q > 0.5): ## If taking upper quantile: ceil
#         print("w_sorted_cum[i]",i, v_sorted[i], w_sorted_cum[i])
        return v_sorted[i]
            
    elif (q < 0.5): ## Elif taking lower quantile:
        
        if (i > 0 and w_sorted_cum[i] == q):
            return v_sorted[i]
        elif (i > 0):
#             print("w_sorted_cum[i-1]",i-1, v_sorted[i-1], w_sorted_cum[i-1])
            return v_sorted[i-1]
        else:
            return v_sorted[0]
        
    else: ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if (w_sorted_cum[i] == 0.5):
            return v_sorted[i]
        
        elif (i > 0):
            return (v_sorted[i]*w_sorted[i] + v_sorted[i-1]*w_sorted[i-1]) / (w_sorted[i] + w_sorted[i-1])
        
        else:
            return v_sorted[0]



# ========== utilities and classes for full conformal with ridge regression ==========

def get_invcov_dot_xt(X_nxp, gamma, use_lapack: bool = True):
    """
    Compute (X^TX + \gamma I)^{-1} X^T

    :param X_nxp: (n, p) numpy array encoding sequences
    :param gamma: float, ridge regularization strength
    :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
    :return: (p, n) numpy array, (X^TX + \gamma I)^{-1} X^T
    """
    reg_pxp = gamma * np.eye(X_nxp.shape[1])
    reg_pxp[0, 0] = 0  # don't penalize intercept term
    cov_pxp = X_nxp.T.dot(X_nxp) + reg_pxp
    if use_lapack:
        # fastest way to invert PD matrices from
        # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
        zz, _ = sc.linalg.lapack.dpotrf(cov_pxp, False, False)
        invcovtri_pxp, info = sc.linalg.lapack.dpotri(zz)
        assert(info == 0)
        invcov_pxp = np.triu(invcovtri_pxp) + np.triu(invcovtri_pxp, k=1).T
    else:
        invcov_pxp = sc.linalg.pinvh(cov_pxp)
    return invcov_pxp.dot(X_nxp.T)


class ConformalRidge(ABC):
    """
    Abstract base class for full conformal with computations optimized for ridge regression.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        """
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        :param gamma: float, ridge regularization strength
        :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
        """
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
        self.ys = ys
        self.n_y = ys.size
        self.gamma = gamma
        self.use_lapack = use_lapack

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def get_insample_scores(self, Xaug_n1xp, ytrain_n):
        """
        Compute in-sample scores, i.e. residuals using model trained on all n + 1 data points (instead of LOO data)

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :return: (n + 1, |Y|) numpy array of scores
        """
        A = get_invcov_dot_xt(Xaug_n1xp, self.gamma, use_lapack=self.use_lapack)
        C = A[:, : -1].dot(ytrain_n)  # p elements
        a_n1 = C.dot(Xaug_n1xp.T)
        b_n1 = A[:, -1].dot(Xaug_n1xp.T)

        # process in-sample scores for each candidate value y
        scoresis_n1xy = np.zeros([ytrain_n.size + 1, self.n_y])
        by_n1xy = np.outer(b_n1, self.ys)
        muhatiy_n1xy = a_n1[:, None] + by_n1xy
        scoresis_n1xy[: -1] = np.abs(ytrain_n[:, None] - muhatiy_n1xy[: -1])
        scoresis_n1xy[-1] = np.abs(self.ys - muhatiy_n1xy[-1])
        return scoresis_n1xy

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, compute_lrs: bool = True):
        """
        Compute LOO scores, i.e. residuals using model trained on n data points (training + candidate test points,
        but leave i-th training point out).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        # fit n + 1 LOO models and store linear parameterizations of \mu_{-i, y}(X_i) as function of y
        n = ytrain_n.size
        ab_nx2 = np.zeros([n, 2])
        C_nxp = np.zeros([n, self.p])
        An_nxp = np.zeros([n, self.p])
        for i in range(n):
            # construct A_{-i}
            Xi_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]]) # n rows
            Ai = get_invcov_dot_xt(Xi_nxp, self.gamma, use_lapack=self.use_lapack)

            # compute linear parameterizations of \mu_{-i, y}(X_i)
            yi_ = np.hstack([ytrain_n[: i], ytrain_n[i + 1 :]])  # n - 1 elements
            Ci = Ai[:, : -1].dot(yi_) # p elements
            ai = Ci.dot(Xaug_n1xp[i])  # = Xtrain_nxp[i]
            bi = Ai[:, -1].dot(Xaug_n1xp[i])

            # store
            ab_nx2[i] = ai, bi
            C_nxp[i] = Ci
            An_nxp[i] = Ai[:, -1]

        # LOO score for i = n + 1
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)
        alast = beta_p.dot(Xaug_n1xp[-1])  # prediction a_{n + 1}. Xaug_n1xp[-1] = Xtest_p

        # process LOO scores for each candidate value y
        scoresloo_n1xy = np.zeros([n + 1, self.n_y])
        by_nxy = np.outer(ab_nx2[:, 1], self.ys)
        prediy_nxy = ab_nx2[:, 0][:, None] + by_nxy
        scoresloo_n1xy[: -1] = np.abs(ytrain_n[:, None] - prediy_nxy)
        scoresloo_n1xy[-1] = np.abs(self.ys - alast)

        # likelihood ratios for each candidate value y
        w_n1xy = None
        if compute_lrs:
            betaiy_nxpxy = C_nxp[:, :, None] + self.ys * An_nxp[:, :, None]
            # compute normalizing constant in Eq. 6 in main paper
            pred_nxyxu = np.tensordot(betaiy_nxpxy, self.Xuniv_uxp, axes=(1, 1))
            normconst_nxy = np.sum(np.exp(lmbda * pred_nxyxu), axis=2)
            ptrain_n = self.ptrain_fn(Xaug_n1xp[: -1])

            w_n1xy = np.zeros([n + 1, self.n_y])
            wi_num_nxy = np.exp(lmbda * prediy_nxy)
            w_n1xy[: -1] = wi_num_nxy / (ptrain_n[:, None] * normconst_nxy)

            # for last i = n + 1, which is constant across candidate values of y
            Z = self.get_normalizing_constant(beta_p, lmbda)
            w_n1xy[-1] = np.exp(lmbda * alast) / (self.ptrain_fn(Xaug_n1xp[-1][None, :]) * Z)
        return scoresloo_n1xy, w_n1xy

    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1, use_is_scores: bool = False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n) if use_is_scores else None

        # compute LOO scores and likelihood ratios
        scoresloo_n1xy, w_n1xy = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence sets =====

        # based on LOO score
        looq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresloo_n1xy)
        loo_cs = self.ys[scoresloo_n1xy[-1] <= looq_y]

        # based on in-sample score
        is_cs = None
        if use_is_scores:
            isq_y = get_weighted_quantile(1 - alpha, w_n1xy, scoresis_n1xy)
            is_cs = self.ys[scoresis_n1xy[-1] <= isq_y]
        return loo_cs, is_cs


### Drew modified
class JAWRidge(ABC):
    """
    Abstract base class for JAW with ridge regression mu function, based on class for full conformal
    """
    def __init__(self, ptrain_fn, Xuniv_uxp, gamma, use_lapack: bool = True):
        """
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        :param gamma: float, ridge regularization strength
        :param use_lapack: bool, whether or not to use low-level LAPACK functions for inverting covariance (fastest)
        """
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
#         self.ys = ys ## Drew: maybe don't need this
#         self.n_y = ys.size
        self.gamma = gamma
        self.use_lapack = use_lapack

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda, compute_lrs: bool = True):
        """
        Compute jackknife+ LOO scores, i.e. residuals using model trained on *n-1* data points (n-1 training points, no candidate test points).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        # Compute jackknife+ LOO residuals, test point predictions, and weights
        n = ytrain_n.size
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros(n) ## Only have one testpoint
        
        # Weights
        unnormalized_weights = np.zeros(n + 1)
        for i in range(n):
            ## Create LOO X and y data
            Xi_LOO_n_minus_1xp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 : n]]) ## LOO training data inputs
            yi_LOO_train_n = np.concatenate((ytrain_n[: i], ytrain_n[i + 1 : n])) ## LOO training data outputs
            
            ## Get LOO residuals and test point predictions
            tmp = get_invcov_dot_xt(Xi_LOO_n_minus_1xp, self.gamma, use_lapack=self.use_lapack)
            beta_p = tmp.dot(yi_LOO_train_n)
            muh_i_LOO = beta_p.dot(Xaug_n1xp[i]) ## ith LOO prediction on point i : mu_{-i}(X_i)
            resids_LOO[i] = np.abs(ytrain_n[i] - muh_i_LOO) ## ith LOO residual
            muh_LOO_vals_testpoint[i] = beta_p.dot(Xaug_n1xp[-1]) ## ith LOO prediction on test point n+1 : mu_{-i}(X_{n+1})
            
            ## Calculate unnormalized weights for the training scores 1:n
            unnormalized_weights[i] = (np.exp(lmbda * muh_i_LOO) / self.ptrain_fn(Xaug_n1xp[i][None, :])) * (np.exp(lmbda * muh_LOO_vals_testpoint[i]) / self.ptrain_fn(Xaug_n1xp[-1][None, :]))
                        
        ## Compute jackknife+ upper and lower predictive values
        unweighted_lower_vals = np.zeros(n+1)
        unweighted_upper_vals = np.zeros(n+1)
        unweighted_lower_vals[:n] = muh_LOO_vals_testpoint - resids_LOO
        unweighted_upper_vals[:n] = muh_LOO_vals_testpoint + resids_LOO
        
        
        ## Add infinity
        unweighted_lower_vals[n] = -math.inf
        unweighted_upper_vals[n] = math.inf
        
        
        ## Calculate test point unnormalized weight
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)
        muh_test = beta_p.dot(Xaug_n1xp[-1])
        unnormalized_weights[n] = (np.exp(lmbda * muh_test) / self.ptrain_fn(Xaug_n1xp[-1][None, :]))**2
        
        weights_normalized = unnormalized_weights / np.sum(unnormalized_weights)
        

        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1, use_is_scores: bool = False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])

        # ===== compute scores and weights =====

        # compute in-sample scores
        scoresis_n1xy = self.get_insample_scores(Xaug_n1xp, ytrain_n) if use_is_scores else None

        # compute LOO scores and likelihood ratios
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence sets =====
        y_lower = weighted_quantile(unweighted_lower_vals, weights_normalized, alpha)
#         print(weights_normalized)
#         print(y_lower)
        y_upper = weighted_quantile(unweighted_upper_vals, weights_normalized, 1 - alpha)
        
        return y_lower, y_upper


    
    
class ConformalRidgeExchangeable(ConformalRidge):
    """
    Class for full conformal with ridge regression, assuming exchangeable data.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=False)
        # for exchangeble data, equal weights on all data points (no need to compute likelihood ratios in line above)
        w_n1xy = np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy


class ConformalRidgeFeedbackCovariateShift(ConformalRidge):
    """
    Class for full conformal with ridge regression under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        scoresloo_n1xy, w_n1xy = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=True)
        return scoresloo_n1xy, w_n1xy
    
    
### Drew modified
class JAWRidgeFeedbackCovariateShift(JAWRidge):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, ptrain_fn, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=True)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized


class ConformalRidgeStandardCovariateShift(ConformalRidge):
    """
    Class for full conformal with ridge regression under standard covariate shift.
    """
    def __init__(self, ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack: bool = True):
        super().__init__(ptrain_fn, ys, Xuniv_uxp, gamma, use_lapack=use_lapack)

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        # fit model to training data
        tmp = get_invcov_dot_xt(Xaug_n1xp[: -1], self.gamma, use_lapack=self.use_lapack)
        beta_p = tmp.dot(ytrain_n)

        # compute normalizing constant for test covariate distribution
        Z = self.get_normalizing_constant(beta_p, lmbda)

        # get likelihood ratios for n + 1 covariates
        pred_n1 = Xaug_n1xp.dot(beta_p)
        ptest_n1 = np.exp(lmbda * pred_n1) / Z
        w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
        return w_n1

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        # LOO scores
        scoresloo_n1xy, _ = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda, compute_lrs=False)

        # compute likelihood ratios
        w_n1 = self.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
        w_n1xy = w_n1[:, None] * np.ones([Xaug_n1xp.shape[0], self.n_y])
        return scoresloo_n1xy, w_n1xy



# ========== utilities and classes for full conformal with black-box model ==========

def get_scores(model, Xaug_nxp, yaug_n, use_loo_score: bool = False):
    if use_loo_score:
        n1 = yaug_n.size  # n + 1
        scores_n1 = np.zeros([n1])

        for i in range(n1):
            Xtrain_nxp = np.vstack([Xaug_nxp[: i], Xaug_nxp[i + 1 :]])
            ytrain_n = np.hstack([yaug_n[: i], yaug_n[i + 1 :]])

            # train on LOO dataset
            model.fit(Xtrain_nxp, ytrain_n)
            pred_1 = model.predict(Xaug_nxp[i][None, :])
            scores_n1[i] = np.abs(yaug_n[i] - pred_1[0])

    else:  # in-sample score
        model.fit(Xaug_nxp, yaug_n)
        pred_n1 = model.predict(Xaug_nxp)
        scores_n1 = np.abs(yaug_n - pred_n1)
    return scores_n1


class Conformal(ABC):
    """
    Abstract base class for full conformal with black-box predictive model.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        """
        :param model: object with predict() method
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param ys: (|Y|,) numpy array of candidate labels
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        """
        self.model = model
        self.ptrain_fn = ptrain_fn
        self.ys = ys
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]
        self.n_y = ys.size

    @abstractmethod
    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda,
                           use_loo_score: bool = True, alpha: float = 0.1, print_every: int = 10, verbose: bool = True):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))

        np.set_printoptions(precision=3)
        cs, n = [], ytrain_n.size
        t0 = time.time()
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        scores_n1xy = np.zeros([n + 1, self.n_y])
        w_n1xy = np.zeros([n + 1, self.n_y])

        for y_idx, y in enumerate(self.ys):

            # get scores
            yaug_n1 = np.hstack([ytrain_n, y])
            scores_n1 = get_scores(self.model, Xaug_n1xp, yaug_n1, use_loo_score=use_loo_score)
            scores_n1xy[:, y_idx] = scores_n1

            # get likelihood ratios
            w_n1 = self.get_lrs(Xaug_n1xp, yaug_n1, lmbda)
            w_n1xy[:, y_idx] = w_n1

            # for each value of inverse temperature lambda, compute quantile of weighted scores
            q = get_weighted_quantile(1 - alpha, w_n1, scores_n1)

            # if y <= quantile, include in confidence set
            if scores_n1[-1] <= q:
                cs.append(y)

            # print progress
            if verbose and (y_idx + 1) % print_every == 0:
                print("Done with {} / {} y values ({:.1f} s)".format(
                    y_idx + 1, self.ys.size, time.time() - t0))
        return np.array(cs), scores_n1xy, w_n1xy

    
### Drew modified
class JAW_FCS(ABC):
    """
    Abstract base class for JAW with ridge regression mu function, based on class for full conformal
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        """
        :param model: object with predict() method
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        """
        self.model = model
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    def compute_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        """
        Compute jackknife+ LOO scores, i.e. residuals using model trained on *n-1* data points (n-1 training points, no candidate test points).

        :param Xaug_n1xp: (n + 1, p) numpy array encoding all n + 1 sequences (training + candidate test point)
        :param ytrain_n: (n,) numpy array of true labels for the n training points
        :param lmbda: float, inverse temperature of design algorithm in Eq. 6, {0, 2, 4, 6} in main paper
        :param compute_lrs: bool: whether or not to compute likelihood ratios (this part takes the longest,
            so set to False if only want to compute scores)
        :return: (n + 1, |Y|) numpy arrays of scores S_i(X_test, y) and weights w_i^y(X_test) in Eq. 3 in main paper
        """
        # Compute jackknife+ LOO residuals, test point predictions, and weights
        n = ytrain_n.size
        n1 = len(Xaug_n1xp) - n
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n,n1))
        
        # Weights
        unnormalized_weights_JAW_FCS = np.zeros((n + 1, n1))
        unnormalized_weights_JAW_SCS = np.zeros((n + 1, n1))
        
        for i in range(n):
            ## Create LOO X and y data
            Xi_LOO_n_minus_1xp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 : n]]) ## LOO training data inputs
            yi_LOO_train_n = np.concatenate((ytrain_n[: i], ytrain_n[i + 1 : n])) ## LOO training data outputs
            
            ## Get LOO residuals and test point predictions
            self.model.fit(Xi_LOO_n_minus_1xp, yi_LOO_train_n)
            muh_i_LOO = self.model.predict(Xaug_n1xp[i].reshape(1, -1)) ## ith LOO prediction on point i : mu_{-i}(X_i)
            resids_LOO[i] = np.abs(ytrain_n[i] - muh_i_LOO) ## ith LOO residual
            muh_LOO_vals_testpoint[i] = self.model.predict(Xaug_n1xp[-n1:]).T ## ith LOO prediction on test point n+1 : mu_{-i}(X_{n+1})
           
            ## Calculate unnormalized likelihoo-ratio weights for FCS
            unnormalized_weights_JAW_FCS[i] = (np.exp(lmbda * muh_i_LOO) / (self.ptrain_fn(Xaug_n1xp[i][None, :]))) * (np.exp(lmbda * muh_LOO_vals_testpoint[i]) / (self.ptrain_fn(Xaug_n1xp[-n1:][None, :])))
            
            
        for j in range(n1):
            ## Calculate unnormalized likelihoo-ratio weights for SCS
            unnormalized_weights_JAW_SCS[:, j] = self.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
            
            
                        
        ## Compute jackknife+ upper and lower predictive values
        unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
        unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
        
        
        ## Add infinity
        unweighted_lower_vals = np.vstack((unweighted_lower_vals, -math.inf*np.ones(n1)))
        unweighted_upper_vals = np.vstack((unweighted_upper_vals, math.inf*np.ones(n1)))
        
        
        ## Calculate test point unnormalized weight
        self.model.fit(Xaug_n1xp[: -n1], ytrain_n)
        muh_test = self.model.predict(Xaug_n1xp[-n1:])
        unnormalized_weights_JAW_FCS[n] = (np.exp(lmbda * muh_test) / self.ptrain_fn(Xaug_n1xp[-n1:][None, :]))**2
        
        weights_normalized_JAW_FCS = np.zeros((n + 1, n1))
        weights_normalized_JAW_SCS = np.zeros((n + 1, n1))
        for j in range(0, n1):
            weights_normalized_JAW_FCS[:,j] = unnormalized_weights_JAW_FCS[:,j] / np.sum(unnormalized_weights_JAW_FCS[:,j])
            weights_normalized_JAW_SCS[:,j] = unnormalized_weights_JAW_SCS[:,j] / np.sum(unnormalized_weights_JAW_SCS[:,j])
        
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized_JAW_FCS, weights_normalized_JAW_SCS

    @abstractmethod
    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        pass

    def get_confidence_set(self, Xtrain_nxp, ytrain_n, Xtest_1xp, lmbda, alpha: float = 0.1):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        n1 = len(Xtest_1xp)

        # ===== compute scores and weights =====

        # compute LOO scores and likelihood ratios
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized_JAW_FCS, weights_normalized_JAW_SCS = self.get_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)

        # ===== construct confidence intervals for FCS and SCS =====
        y_lower_JAW_FCS = np.zeros(n1)
        y_upper_JAW_FCS = np.zeros(n1)
        y_lower_JAW_SCS = np.zeros(n1)
        y_upper_JAW_SCS = np.zeros(n1)
        y_lower_Jplus = np.zeros(n1)
        y_upper_Jplus = np.zeros(n1)
        uniform_weights = np.ones(n+1) / (n+1)
        for j in range(0, n1):
            y_lower_JAW_FCS[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized_JAW_FCS[:, j], alpha)
            y_upper_JAW_FCS[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized_JAW_FCS[:, j], 1 - alpha)
            y_lower_JAW_SCS[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized_JAW_SCS[:, j], alpha)
            y_upper_JAW_SCS[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized_JAW_SCS[:, j], 1 - alpha)
            y_lower_Jplus[j] = weighted_quantile(unweighted_lower_vals[:, j], uniform_weights, alpha)
            y_upper_Jplus[j] = weighted_quantile(unweighted_upper_vals[:, j], uniform_weights, 1 - alpha)
            
        return y_lower_JAW_FCS, y_upper_JAW_FCS, y_lower_JAW_SCS, y_upper_JAW_SCS, y_lower_Jplus, y_upper_Jplus

    
        
    
    def compute_PIs(self, Xtrain_nxp, ytrain_n, Xtest_1xp, ytest_n1, pred_train_test, Xtrain_split, Xcal_split, ytrain_split, ycal_split, Xtest_n1xp_split, ytest_n1_split, pred_cal_test_split, lmbda, alpha: float = 0.1, K_vals = [8, 12, 16, 24, 32, 48], run_split=False):
        if (self.p != Xtrain_nxp.shape[1]):
            raise ValueError('Feature dimension {} differs from provided Xuniv_uxp {}'.format(
                Xtrain_nxp.shape[1], self.Xuniv_uxp.shape))
        Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        Xaug_cal_test_split = np.vstack([Xcal_split, Xtest_1xp])
        n = ytrain_n.size
        n1 = len(Xaug_n1xp) - n
        
        ###############################
        # split conformal
        ###############################
#         print("pred_cal_test_split")
#         print(pred_cal_test_split)
#         idx = np.random.permutation(n)
        n_half = int(np.floor(n/2))
#         idx_train, idx_cal = idx[:n_half], idx[n_half:]
        muh_split = self.model.fit(Xtrain_split, ytrain_split)
        muh_split_vals = self.model.predict(np.r_[Xcal_split,Xtest_n1xp_split])
#         muh_split_vals = pred_cal_test_split
        resids_split = np.abs(ycal_split-muh_split_vals[:(n-n_half)])
        muh_split_vals_testpoint = muh_split_vals[(n-n_half):]
        ind_split = (np.ceil((1-alpha)*(n-n_half+1))).astype(int)

        ###############################
        # weighted split conformal
        ###############################
        predall_n = self.model.predict(self.Xuniv_uxp)
        Z = np.sum(np.exp(lmbda * predall_n))
        
        wsplit_ptest_n1 = np.exp(lmbda * muh_split_vals) / Z
        SCS_split_weights_vec = wsplit_ptest_n1 / self.ptrain_fn(Xaug_cal_test_split)
        
        

        ## Add infty (distribution on augmented real line)
        positive_infinity = np.array([float('inf')])
        unweighted_split_vals = np.concatenate([resids_split, positive_infinity])

        wsplit_quantiles = np.zeros(n1)

        weights_normalized_wsplit = np.zeros(((n-n_half) + 1, n1))
        sum_cal_weights = np.sum(SCS_split_weights_vec[:(n-n_half)])
        for j in range(0, n1):
            for i in range(0, (n-n_half) + 1):
                if (i < (n-n_half)):
#                     i_cal = idx_cal[i]
                    weights_normalized_wsplit[i, j] = SCS_split_weights_vec[i] / (sum_cal_weights + SCS_split_weights_vec[(n-n_half) + j])
                else:
                    weights_normalized_wsplit[i, j] = SCS_split_weights_vec[(n-n_half)+j] / (sum_cal_weights + SCS_split_weights_vec[(n-n_half) + j])

        
        wsplit_quantiles_lower = np.zeros(n1)
        wsplit_quantiles_upper = np.zeros(n1)
        for j in range(0, n1):
#             C_n = get_randomized_staircase_confidence_set(resids_split, weights_normalized_wsplit[:, j], pred_cal_test_split[(n-n_half) + j])
#             print("C_n", C_n)
#             wsplit_quantiles_lower[j] = C_n[0][0]
#             wsplit_quantiles_upper[j] = C_n[0][1]
#             wsplit_quantiles[j] = get_randomized_staircase_coverage(C_n, ytest_n1_split)
            wsplit_quantiles[j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)
        
        
        ###############################
        # JAW FCS & SCS
        ###############################
        
        # Compute jackknife+ LOO residuals, test point predictions, and weights
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n,n1))
        
        # Weights
        unnormalized_weights_JAW_FCS = np.zeros((n + 1, n1))
        unnormalized_weights_JAW_SCS = np.zeros((n + 1, n1))

        
        for i in range(n):
            ## Create LOO X and y data
            Xi_LOO_n_minus_1xp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 : n]]) ## LOO training data inputs
            yi_LOO_train_n = np.concatenate((ytrain_n[: i], ytrain_n[i + 1 : n])) ## LOO training data outputs
            
            ## Get LOO residuals and test point predictions
            self.model.fit(Xi_LOO_n_minus_1xp, yi_LOO_train_n)
            muh_i_LOO = self.model.predict(Xaug_n1xp[i].reshape(1, -1)) ## ith LOO prediction on point i : mu_{-i}(X_i)
            resids_LOO[i] = np.abs(ytrain_n[i] - muh_i_LOO) ## ith LOO residual
            muh_LOO_vals_testpoint[i] = self.model.predict(Xaug_n1xp[-n1:]).T ## ith LOO prediction on test point n+1 : mu_{-i}(X_{n+1})
           
            ## Calculate unnormalized likelihoo-ratio weights for FCS
            unnormalized_weights_JAW_FCS[i] = (np.exp(lmbda * muh_i_LOO) / (self.ptrain_fn(Xaug_n1xp[i][None, :]))) * (np.exp(lmbda * muh_LOO_vals_testpoint[i]) / (self.ptrain_fn(Xaug_n1xp[-n1:][None, :])))
            
        ## Calculate FCS test point unnormalized weight
        self.model.fit(Xaug_n1xp[: -n1], ytrain_n)
        muh_test = self.model.predict(Xaug_n1xp[-n1:]) ## did have np.exp(lmbda * pred_train_test[-n1:])
        unnormalized_weights_JAW_FCS[n] = (np.exp(lmbda * muh_test) / self.ptrain_fn(Xaug_n1xp[-n1:][None, :]))**2
            
        ## Standard covariate shift weights
        muh_train_test = self.model.predict(Xaug_n1xp)
        ptest_train_test = np.exp(lmbda * muh_train_test) # pred_train_test
#         SCS_weights_vec = self.get_lrs(Xaug_n1xp, ytrain_n, lmbda)
        SCS_weights_vec = ptest_train_test / self.ptrain_fn(Xaug_n1xp)
        weights_normalized_JAW_SCS = np.zeros((n + 1, n1))
        sum_train_weights = np.sum(SCS_weights_vec[0:n])
        for j in range(0, n1):
            for i in range(0, n + 1):
                if (i < n):
                    weights_normalized_JAW_SCS[i, j] = SCS_weights_vec[i] / (sum_train_weights + SCS_weights_vec[n + j])
                else:
                    weights_normalized_JAW_SCS[i, j] = SCS_weights_vec[n+j] / (sum_train_weights + SCS_weights_vec[n + j])

                        
        ## Compute jackknife+ upper and lower predictive values
        unweighted_lower_vals = (muh_LOO_vals_testpoint.T - resids_LOO).T
        unweighted_upper_vals = (muh_LOO_vals_testpoint.T + resids_LOO).T
        

        ## Add infinity
        unweighted_lower_vals = np.vstack((unweighted_lower_vals, -math.inf*np.ones(n1)))
        unweighted_upper_vals = np.vstack((unweighted_upper_vals, math.inf*np.ones(n1)))
        

        
        weights_normalized_JAW_FCS = np.zeros((n + 1, n1))
        for j in range(0, n1):
            weights_normalized_JAW_FCS[:,j] = unnormalized_weights_JAW_FCS[:,j] / np.sum(unnormalized_weights_JAW_FCS[:,j])
        
        # ===== construct confidence intervals for FCS and SCS =====
        y_lower_JAW_FCS = np.zeros(n1)
        y_upper_JAW_FCS = np.zeros(n1)
        y_lower_JAW_SCS = np.zeros(n1)
        y_upper_JAW_SCS = np.zeros(n1)
        y_lower_Jplus = np.zeros(n1)
        y_upper_Jplus = np.zeros(n1)
        uniform_weights = np.ones(n+1) / (n+1)
        for j in range(0, n1):
            y_lower_JAW_FCS[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized_JAW_FCS[:, j], alpha)
            y_upper_JAW_FCS[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized_JAW_FCS[:, j], 1 - alpha)
            y_lower_JAW_SCS[j] = weighted_quantile(unweighted_lower_vals[:, j], weights_normalized_JAW_SCS[:, j], alpha)
            y_upper_JAW_SCS[j] = weighted_quantile(unweighted_upper_vals[:, j], weights_normalized_JAW_SCS[:, j], 1 - alpha)
            y_lower_Jplus[j] = weighted_quantile(unweighted_lower_vals[:, j], uniform_weights, alpha)
            y_upper_Jplus[j] = weighted_quantile(unweighted_upper_vals[:, j], uniform_weights, 1 - alpha)
            
            
        ## Add PIs for initially non JAW or K-dependent methods
        PIs_dict = {'JAW-FCS' : pd.DataFrame(np.c_[y_lower_JAW_FCS, \
                        y_upper_JAW_FCS],\
                           columns = ['lower','upper']),\
                'JAW-SCS' : pd.DataFrame(np.c_[y_lower_JAW_SCS, \
                        y_upper_JAW_SCS],\
                           columns = ['lower','upper']),\
                'jackknife+' : pd.DataFrame(np.c_[y_lower_Jplus, \
                        y_upper_Jplus],\
                           columns = ['lower','upper']),\
                'split' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - np.sort(resids_split)[ind_split-1], \
                           muh_split_vals_testpoint + np.sort(resids_split)[ind_split-1]],\
                            columns = ['lower','upper']),\
                'weighted_split' : pd.DataFrame(\
                    np.c_[muh_split_vals_testpoint - wsplit_quantiles, \
                           muh_split_vals_testpoint + wsplit_quantiles],\
                            columns = ['lower','upper'])}
        
        
        ###############################
        # For each value of K in K_vals
        ###############################

        for K in K_vals:
            ###############################
            # CV+
            ###############################
            
            ## CV+
            n_K = np.floor(n/K).astype(int)
            base_inds_to_delete = np.arange(n_K).astype(int)
            resids_LKO = np.zeros(n)
            muh_LKO_vals_testpoint = np.zeros((n,n1))
            muh_vals_LKO_all = np.zeros(n)
            
            ## weights for wCV_FCS
            unnormalized_weights_wCV_FCS = np.zeros((n + 1, n1))
            
            for i in range(K):
                inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
                self.model.fit(np.delete(Xtrain_nxp,inds_to_delete,0),np.delete(ytrain_n,inds_to_delete))
                muh_vals_LKO = self.model.predict(np.r_[Xtrain_nxp[inds_to_delete],Xaug_n1xp[-n1:]])
                resids_LKO[inds_to_delete] = np.abs(ytrain_n[inds_to_delete] - muh_vals_LKO[:n_K])
                for inner_K in range(n_K):
#                     muh_vals_LKO_all[inds_to_delete[inner_K]] = self.model.predict(Xaug_n1xp[]
                    muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
                    muh_vals_LKO_all[inds_to_delete[inner_K]] = muh_vals_LKO[inner_K]
            
                ## Calculate unnormalized likelihoo-ratio weights for FCS
            for i in range(0, n):
                unnormalized_weights_wCV_FCS[i] = (np.exp(lmbda * muh_vals_LKO_all[i]) / (self.ptrain_fn(Xaug_n1xp[i][None, :]))) * (np.exp(lmbda * muh_LKO_vals_testpoint[i]) / (self.ptrain_fn(Xaug_n1xp[-n1:][None, :])))
            
            ## Calculate FCS test point unnormalized weight
            unnormalized_weights_wCV_FCS[n] = (np.exp(lmbda * muh_test) / self.ptrain_fn(Xaug_n1xp[-n1:][None, :]))**2
            
#             print(np.sum(unnormalized_weights_wCV_FCS))
            ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)
    
            weights_normalized_wCV_FCS = np.zeros((n + 1, n1))
            for j in range(0, n1):
                weights_normalized_wCV_FCS[:,j] = unnormalized_weights_wCV_FCS[:,j] / np.sum(unnormalized_weights_wCV_FCS[:,j])
        

        
#             print("weights_normalized_wCV_FCS")
#             print(weights_normalized_wCV_FCS)

            ###############################
            # JAW : 
            # wCV+: 
            ###############################

            unweighted_upper_vals_CV = (muh_LKO_vals_testpoint.T + resids_LKO).T
            unweighted_lower_vals_CV = (muh_LKO_vals_testpoint.T - resids_LKO).T

            ## Add infty (distribution on augmented real line)

            unweighted_upper_vals_CV = np.vstack((unweighted_upper_vals_CV, positive_infinity*np.ones(n1)))
            unweighted_lower_vals_CV = np.vstack((unweighted_lower_vals_CV, -positive_infinity*np.ones(n1)))

            ## Get normalized weights:
            
            ## FCS
#             y_upper_JAW_FCS = np.zeros(n1)
#             y_lower_JAW_FCS = np.zeros(n1)

            y_upper_wCV_FCS = np.zeros(n1)
            y_lower_wCV_FCS = np.zeros(n1)

            y_upper_JAW_FCS_KLOO_rep = np.zeros(n1)
            y_lower_JAW_FCS_KLOO_rep = np.zeros(n1)
            
            y_upper_JAW_FCS_KLOO_det = np.zeros(n1)
            y_lower_JAW_FCS_KLOO_det = np.zeros(n1)
            
            ## SCS
#             y_upper_JAW_SCS = np.zeros(n1)
#             y_lower_JAW_SCS = np.zeros(n1)

            y_upper_wCV_SCS = np.zeros(n1)
            y_lower_wCV_SCS = np.zeros(n1)

            y_upper_JAW_SCS_KLOO_rep = np.zeros(n1)
            y_lower_JAW_SCS_KLOO_rep = np.zeros(n1)
            
            y_upper_JAW_SCS_KLOO_det = np.zeros(n1)
            y_lower_JAW_SCS_KLOO_det = np.zeros(n1)
            

            for j in range(0, n1):
                y_upper_wCV_FCS[j] = weighted_quantile(unweighted_upper_vals_CV[:, j], weights_normalized_wCV_FCS[:, j], 1 - alpha)
                y_lower_wCV_FCS[j] = weighted_quantile(unweighted_lower_vals_CV[:, j], weights_normalized_wCV_FCS[:, j], alpha)
                y_upper_wCV_SCS[j] = weighted_quantile(unweighted_upper_vals_CV[:, j], weights_normalized_JAW_SCS[:, j], 1 - alpha)
                y_lower_wCV_SCS[j] = weighted_quantile(unweighted_lower_vals_CV[:, j], weights_normalized_JAW_SCS[:, j], alpha)

                ## K LOO sampling with replacement
                JAW_FCS_KLOO_inds = []
                JAW_SCS_KLOO_inds = []
                weights_normalized_FCS_j_cum = np.cumsum(weights_normalized_JAW_FCS[:, j])
                weights_normalized_SCS_j_cum = np.cumsum(weights_normalized_JAW_SCS[:, j])
                while (len(set(JAW_FCS_KLOO_inds)) < K):
                    JAW_FCS_KLOO_inds.append(random.choices(list(range(0, n+1)), cum_weights = weights_normalized_FCS_j_cum)[0])
                while (len(set(JAW_SCS_KLOO_inds)) < K):
                    JAW_SCS_KLOO_inds.append(random.choices(list(range(0, n+1)), cum_weights = weights_normalized_SCS_j_cum)[0])
                    
                JAW_FCS_KLOO_inds = np.array(JAW_FCS_KLOO_inds)
                JAW_SCS_KLOO_inds = np.array(JAW_SCS_KLOO_inds)

                upper_vals_JAW_FCS_KLOO_all = unweighted_upper_vals[:, j][JAW_FCS_KLOO_inds]
                lower_vals_JAW_FCS_KLOO_all = unweighted_lower_vals[:, j][JAW_FCS_KLOO_inds]
                
                upper_vals_JAW_SCS_KLOO_all = unweighted_upper_vals[:, j][JAW_SCS_KLOO_inds]
                lower_vals_JAW_SCS_KLOO_all = unweighted_lower_vals[:, j][JAW_SCS_KLOO_inds]

                upper_vals_JAW_FCS_KLOO_unique = np.unique(unweighted_upper_vals[:, j][JAW_FCS_KLOO_inds], return_counts=True)
                lower_vals_JAW_FCS_KLOO_unique = np.unique(unweighted_lower_vals[:, j][JAW_FCS_KLOO_inds], return_counts=True)
                
                upper_vals_JAW_SCS_KLOO_unique = np.unique(unweighted_upper_vals[:, j][JAW_SCS_KLOO_inds], return_counts=True)
                lower_vals_JAW_SCS_KLOO_unique = np.unique(unweighted_lower_vals[:, j][JAW_SCS_KLOO_inds], return_counts=True)

                y_upper_JAW_FCS_KLOO_rep[j] = weighted_quantile(upper_vals_JAW_FCS_KLOO_unique[0], upper_vals_JAW_FCS_KLOO_unique[1]/np.sum(upper_vals_JAW_FCS_KLOO_unique[1]), 1 - alpha)
                y_lower_JAW_FCS_KLOO_rep[j] = weighted_quantile(lower_vals_JAW_FCS_KLOO_unique[0], lower_vals_JAW_FCS_KLOO_unique[1]/np.sum(lower_vals_JAW_FCS_KLOO_unique[1]), alpha)
                
                y_upper_JAW_SCS_KLOO_rep[j] = weighted_quantile(upper_vals_JAW_SCS_KLOO_unique[0], upper_vals_JAW_SCS_KLOO_unique[1]/np.sum(upper_vals_JAW_SCS_KLOO_unique[1]), 1 - alpha)
                y_lower_JAW_SCS_KLOO_rep[j] = weighted_quantile(lower_vals_JAW_SCS_KLOO_unique[0], lower_vals_JAW_SCS_KLOO_unique[1]/np.sum(lower_vals_JAW_SCS_KLOO_unique[1]), alpha)
                
                
                ## K LOO deterministic
                FCS_inds_K_largest = np.concatenate((np.argpartition(weights_normalized_JAW_FCS[:n, j], -K)[-K:], [n]))
                FCS_K_LOO_det_norm_weights = weights_normalized_JAW_FCS[:, j][FCS_inds_K_largest]/np.sum(weights_normalized_JAW_FCS[:, j][FCS_inds_K_largest])
                
#                 print("FCS_K_LOO_det_norm_weights!!!!")
#                 print(FCS_K_LOO_det_norm_weights)
#                 print(FCS_inds_K_largest)
                y_upper_JAW_FCS_KLOO_det[j] = weighted_quantile(unweighted_upper_vals[:, j][FCS_inds_K_largest], FCS_K_LOO_det_norm_weights, 1 - alpha)
                y_lower_JAW_FCS_KLOO_det[j] = weighted_quantile(unweighted_lower_vals[:, j][FCS_inds_K_largest], FCS_K_LOO_det_norm_weights, alpha)
                
                SCS_inds_K_largest = np.concatenate((np.argpartition(weights_normalized_JAW_SCS[:n, j], -K)[-K:], [n]))
                SCS_K_LOO_det_norm_weights = weights_normalized_JAW_SCS[:, j][SCS_inds_K_largest]/np.sum(weights_normalized_JAW_SCS[:, j][SCS_inds_K_largest])
#                 print("SCS_K_LOO_det_norm_weights!!!!")
#                 print(SCS_K_LOO_det_norm_weights)
#                 print(SCS_inds_K_largest)
                y_upper_JAW_SCS_KLOO_det[j] = weighted_quantile(unweighted_upper_vals[:, j][SCS_inds_K_largest], SCS_K_LOO_det_norm_weights, 1 - alpha)
                y_lower_JAW_SCS_KLOO_det[j] = weighted_quantile(unweighted_lower_vals[:, j][SCS_inds_K_largest], SCS_K_LOO_det_norm_weights, alpha)
                

            ## Add PIs for each K and method
            PIs_dict['CV+_K' + str(K)] = pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - resids_LKO,axis=1).T[-ind_Kq], \
                        np.sort(muh_LKO_vals_testpoint.T + resids_LKO,axis=1).T[ind_Kq-1]],\
                           columns = ['lower','upper'])
            
            PIs_dict['wCV_FCS_K' + str(K)] = pd.DataFrame(\
                    np.c_[y_lower_wCV_FCS, \
                        y_upper_wCV_FCS],\
                           columns = ['lower','upper'])
            
            PIs_dict['wCV_SCS_K' + str(K)] = pd.DataFrame(\
                    np.c_[y_lower_wCV_SCS, \
                        y_upper_wCV_SCS],\
                           columns = ['lower','upper'])
            
            PIs_dict['JAW_FCS_KLOO_rep_K' + str(K)] = pd.DataFrame(\
                    np.c_[y_lower_JAW_FCS_KLOO_rep, \
                        y_upper_JAW_FCS_KLOO_rep],\
                           columns = ['lower','upper'])
            
            PIs_dict['JAW_SCS_KLOO_rep_K' + str(K)] = pd.DataFrame(\
                    np.c_[y_lower_JAW_SCS_KLOO_rep, \
                        y_upper_JAW_SCS_KLOO_rep],\
                           columns = ['lower','upper'])
            
            PIs_dict['JAW_FCS_KLOO_det_K' + str(K)] = pd.DataFrame(\
                    np.c_[y_lower_JAW_FCS_KLOO_det, \
                        y_upper_JAW_FCS_KLOO_det],\
                           columns = ['lower','upper'])
            
            PIs_dict['JAW_SCS_KLOO_det_K' + str(K)] = pd.DataFrame(\
                    np.c_[y_lower_JAW_SCS_KLOO_det, \
                        y_upper_JAW_SCS_KLOO_det],\
                           columns = ['lower','upper'])
            
#             PIs_dict['muh_test'] = pd.DataFrame(muh_test,
#                            columns = ['muh_test'])
            
        
        return PIs_dict
    
    

class ConformalExchangeable(Conformal):
    """
    Full conformal with black-box predictive model, assuming exchangeable data.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        return np.ones([Xaug_n1xp.shape[0]])


class ConformalFeedbackCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # compute weights for each value of lambda, the inverse temperature
        w_n1 = np.zeros([yaug_n1.size])
        for i in range(yaug_n1.size):

            # fit LOO model
            Xtr_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]])
            ytr_n = np.hstack([yaug_n1[: i], yaug_n1[i + 1 :]])
            self.model.fit(Xtr_nxp, ytr_n)

            # compute normalizing constant
            predall_n = self.model.predict(self.Xuniv_uxp)
            Z = np.sum(np.exp(lmbda * predall_n))

            # compute likelihood ratios
            testpred = self.model.predict(Xaug_n1xp[i][None, :])
            ptest = np.exp(lmbda * testpred) / Z
            w_n1[i] = ptest / self.ptrain_fn(Xaug_n1xp[i][None, :])
        return w_n1


### Drew modified
class JAWFeedbackCovariateShift(JAW_FCS):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda, split=False, idx=None): ### THIS IS GOING TO HAVE TO TAKE AS ARGUMENTS THIS SPLITS THEMSELVES
        n = len(ytrain_n)
        n1 = len(Xaug_n1xp) - n
        
        if (split == True):
            n_half = int(np.floor(n/2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:]
            self.model.fit(Xaug_n1xp[idx_train],ytrain_n[idx_train])
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        else:
            self.model.fit(Xaug_n1xp[: -n1], ytrain_n)  # Xtrain_nxp, ytrain_n
            # get likelihood ratios
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        
        
### Drew modified
class SplitFeedbackCovariateShift(JAW_FCS):
    """
    Class for JAW with ridge regression under feedback covariate shift
    """
    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)

    def get_loo_scores_and_lrs(self, Xaug_n1xp, ytrain_n, lmbda):
        unweighted_lower_vals, unweighted_upper_vals, weights_normalized = self.compute_loo_scores_and_lrs(Xaug_n1xp, ytrain_n, lmbda)
        return unweighted_lower_vals, unweighted_upper_vals, weights_normalized

    def get_lrs(self, Xaug_n1xp, ytrain_n, lmbda, split=False, idx=None): ### THIS IS GOING TO HAVE TO TAKE AS ARGUMENTS THIS SPLITS THEMSELVES
        n = len(ytrain_n)
        n1 = len(Xaug_n1xp) - n
        
        if (split == True):
            n_half = int(np.floor(n/2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:]
            self.model.fit(Xaug_n1xp[idx_train],ytrain_n[idx_train])
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1

        else:
            self.model.fit(Xaug_n1xp[: -n1], ytrain_n)  # Xtrain_nxp, ytrain_n
            # get likelihood ratios
            pred_n1 = self.model.predict(Xaug_n1xp)
            ptest_n1 = np.exp(lmbda * pred_n1)
            w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
            return w_n1


class ConformalFeedbackCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under feedback covariate shift via Eq. 6 in main paper.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # compute weights for each value of lambda, the inverse temperature
        w_n1 = np.zeros([yaug_n1.size])
        for i in range(yaug_n1.size):

            # fit LOO model
            Xtr_nxp = np.vstack([Xaug_n1xp[: i], Xaug_n1xp[i + 1 :]])
            ytr_n = np.hstack([yaug_n1[: i], yaug_n1[i + 1 :]])
            self.model.fit(Xtr_nxp, ytr_n)

            # compute normalizing constant
            predall_n = self.model.predict(self.Xuniv_uxp)
            Z = np.sum(np.exp(lmbda * predall_n))

            # compute likelihood ratios
            testpred = self.model.predict(Xaug_n1xp[i][None, :])
            ptest = np.exp(lmbda * testpred) / Z
            w_n1[i] = ptest / self.ptrain_fn(Xaug_n1xp[i][None, :])
        return w_n1


class ConformalStandardCovariateShift(Conformal):
    """
    Full conformal with black-box predictive model under standard covariate shift.
    """
    def __init__(self, model, ptrain_fn, ys, Xuniv_uxp):
        super().__init__(model, ptrain_fn, ys, Xuniv_uxp)

    def get_lrs(self, Xaug_n1xp, yaug_n1, lmbda):
        # get normalization constant for test covariate distribution
        self.model.fit(Xaug_n1xp[: -1], yaug_n1[: -1])  # Xtrain_nxp, ytrain_n
        predall_u = self.model.predict(self.Xuniv_uxp)
        Z = np.sum(np.exp(lmbda * predall_u))

        # get likelihood ratios
        pred_n1 = self.model.predict(Xaug_n1xp)
        ptest_n1 = np.exp(lmbda * pred_n1) / Z
        w_n1 = ptest_n1 / self.ptrain_fn(Xaug_n1xp)
        return w_n1

    