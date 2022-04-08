from scripts.util import print_omni_dict
# from scripts.viz import *

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import wilcoxon, ranksums, ttest_rel, ttest_ind, normaltest, bartlett
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold, GridSearchCV

def permutation_test(x, y, paired=False, n_perm=int(1e5), plot_me=False, tails=2):
    ''' performs a permutation test:
    paired... [True\False], whether the samples are paired
    n_perm... number of permutations defaults to 100k since numba makes it very fast :)
    plot_me... plot permuted values and true value
    tails... =2 (two-tailed), =1 (two-tailed & x>y) or =-1 (two-tailed and x<y)
    Author: Lukas Hecker, 03/2020
    '''
    if type(x) == list or type(y) == list:
        x = np.array(x)
        y = np.array(y)
    d_perm = np.zeros(shape=(n_perm))
    d_true = np.mean(x) - np.mean(y)
    if paired and len(x) == len(y):

        # assert len(x) == len(y), f"x and y are not of equal length, choose paired=False instead!\nx:{len(x)}, y:{len(y)}"
        
        signs = np.ones_like(x)
        signs[0:int(np.round(len(x)/2))] = signs[0:int(np.round(len(x)/2))] * -1
        diff = x-y
        d_perm = perm_pair(diff, n_perm, signs, d_perm)
        
    else:
        all_in_one_pot = np.array(list(x) + list(y))
        d_perm = perm_nopair(all_in_one_pot, n_perm, d_perm)
        
        
    # Calculate p as the proportion of permuted group differences that are abs() larger than the true difference:
    if tails == 2:
        p = len(np.where(np.abs(d_perm)>=np.abs(d_true))[0]) / n_perm
    elif tails == -1:
        p = len(np.where(d_perm<=d_true)[0]) / n_perm
    elif tails == 1:
        p = len(np.where(d_perm>=d_true)[0]) / n_perm
    else:
        raise NameError('tails should be either 2 (two-tailed), 1 (two-tailed & x>y) or -1 (two-tailed and y>x)')

    # Clip such that the lowest possible p-value is dependent onthe number of permutations
    p = np.clip(p, 1. / n_perm, None)

    if plot_me:
        plt.figure()
        plt.hist(d_perm)
        plt.plot([d_true, d_true], [plt.ylim()[0], plt.ylim()[1]])
        plt.title(f'd_true={d_true}, p={p}')

    return p

@jit(nopython=True)
def perm_nopair(all_in_one_pot, n_perm, d_perm):
    for i in range(n_perm):
        np.random.shuffle(all_in_one_pot)
        d_perm[i] = np.mean(all_in_one_pot[0:int(len(all_in_one_pot)/2)]) - np.mean(all_in_one_pot[int(len(all_in_one_pot)/2):])
    return d_perm

@jit(nopython=True)
def perm_pair(diff, n_perm, signs, d_perm):
    for i in range(n_perm):
        np.random.shuffle(signs)
        d_perm[i] = np.mean(diff * signs)
    return d_perm

def omni_stat(x, y, paired=False, tails=2, n_perm=int(1e5), verbose=0, effort=0):
    if paired:
        assert len(x) == len(y), 'Length of x and y is unequal. Choose paired=False instead!'
    p_perm = permutation_test(x, y, paired=paired, tails=tails, n_perm=n_perm, plot_me=False)
    # Test for normal distribution
    if len(x) >= 8 or len(y) >= 8:

        _, normality_x = normaltest(x)
        _, normality_y = normaltest(y)
    else:
        normality_x = normality_y = None
    
    if normality_x is not None:
        if normality_x > 0.05 and normality_y > 0.05:
            isnormal = ''
            normal = True
        else:
            isnormal = 'not '
            normal = False
    else:
        isnormal = 'maybe '
        normal = False
    
    # Test for homoscedasticity (equal variances):
    _, eq_var_p = bartlett(x, y)

    if eq_var_p > 0.05:
        isequal = ''
        equal = True
    else:
        isequal = 'not '
        equal = False
    if paired:
        if tails == -1:
            alternative = 'less'
        elif tails == 1:
            alternative = 'greater'
        elif tails == 2:
            alternative = 'two-sided'
        else:
            raise ValueError('tails must be -1, 1 or 2!')

        t_ttest, p_ttest = ttest_rel(x, y)
        t_wilc, p_wilc = wilcoxon(x, y, alternative=alternative)
    else:
        t_welch, p_welch = ttest_ind(x, y, equal_var=False)
        t_ttest, p_ttest = ttest_ind(x, y)
        t_wilc, p_wilc = ranksums(x, y)
    dec = decision_tree(equal, normal)
    # Cohens d
    d = cohens_d(x, y)
    # SVM discriminability
    loo = 0  # classification_test(x, y, effort=effort, cv='auto')

    omni_dict =  {'Is normal': normal,
            'Equal Variance': equal,
            'Rec': dec,
            'Mean Diff': np.mean(x) - np.mean(y),
            '': '', 
            'p_perm': p_perm, 
            ' ': '', 
            'p_ttest': p_ttest,
            't_ttest': t_ttest, 
            '  ': '', 
            'p_wilc': p_wilc, 
            't_wilc': t_wilc, 
            '   ': '', 
            'p_welch': p_welch,
            't_welch': t_welch,
            '    ': '', 
            'Cohens d': d,
            'SVM acc': loo}
    if verbose == 1:
        print_omni_dict(omni_dict)
    return omni_dict


def cohens_d(x, y):
    n1 = len(x)
    n2 = len(y)
    s = np.sqrt( ((n1 - 1) * variance(x) + (n2 - 1) * variance(y)) / (n1 + n2 - 2) )
    d = (np.mean(x) - np.mean(y)) / s
    return d

def variance(x):
    return np.sum((x - np.mean(x))**2) / (len(x)-1)

def decision_tree(equal, normal):
    if equal and normal:
        dec = ['t-tst']
    elif equal and (not normal):
        dec = ['wilc', 'perm']
    elif (not equal) and normal:
        dec = ['welch']
    elif (not equal) and (not normal):
        dec = ['perm']
    return dec


def classification_test(a, b, effort=0, cv='auto'):

    # Organize Variables and crossvalidation
    X = np.concatenate([a, b], axis=0)
    y = np.concatenate([np.zeros_like(a), np.ones_like(b)], axis=0).astype(int)
    # Crossvalidation
    if cv=='loo' or (cv=='auto' and len(X)<100):
        # print('Decided for leave one out cv')
        cv='loo'
        loo = LeaveOneOut()
        split = loo.split(X)
    elif cv=='stratkfold' or (cv=='auto' and len(X)>=100):
        # print('Decided for stratified k-fold cv')
        cv='stratkfold'
        loo = StratifiedKFold()
        split = loo.split(X, y)
    else:
        print('crossval selection did not work')
        return


    loo.get_n_splits(X)
    accuracies = []

    # Grid for parameter search
    if effort==0:
        param_grid = []
    elif effort==1:
        param_grid = {'C': [0.1, 10], 'gamma': [1, 0.01, 0.001],'kernel': ['rbf', 'linear']}
    elif effort==2:
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    else:
        print(f'effort ({effort}) must be 0, 1 or 2')
        return

    
    cnt = 0
    for train_index, test_index in split:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC()

        if effort != 0:
            # Do hyperparameter tuning
            grid = GridSearchCV(clf, param_grid, refit=True, verbose=0)
        else:
            # No hyperparameter tuning
            grid = clf

        if cv == 'loo':
            grid.fit(X_train.reshape(-1, 1), y_train)
            preds = grid.predict(X_test.reshape(1, -1))
        elif cv == 'stratkfold':
            grid.fit(X_train.reshape(-1, 1), y_train)
            preds = grid.predict(X_test.reshape(-1, 1))

        # Determine accuracy for current fold
        comp = preds == y_test

        accuracies.append(len(np.where(comp)[0]) / len(comp))

        cnt += 1
    acc = np.mean(accuracies)  # len(np.argwhere(accuracies)) / len(accuracies)
    return acc

def get_p_img(metric):
    n_freq, n_time = metric[0].shape[1:3]
    p_img = np.zeros((n_freq, n_time))
    for f in range(n_freq):
        for t in range(n_time):
            p_img[f, t] = ranksums(metric[0][:, f, t], metric[1][:, f, t])[1]
    return p_img

def p_img_mask(p_img, alpha_crit=0.05, alpha=0.75):
    p_mask = (p_img<alpha_crit).astype(float)
    p_mask[p_mask==0] = alpha
    return p_mask