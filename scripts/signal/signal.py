import numpy as np
import string
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def get_slope(x, y, order=1):
    ''' Fit a polynomial function of nth order to the data specified in x and y.
    Parameters:
    -----------
    x : list, x values
    y : list, y values
    order : order of the polynomial
    Returns:
    '''
    assert len(x) == len(y), 'x and y must be of same length.'
    assert type(oder) == int, 'order must be an integer.'

    fitparams = np.polyfit(x, y, 1)
    return fitparams[:-1]


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interp_nans(y):
    if type(y) == list:
        y = np.array(y)
        
    nans, x= nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    
    if any(np.isnan(y)):
        print('interpolation did not work!')
    return y

def bootstrap_correlation(corrFunction, X, Y, numberOfBootstraps=int(1e4)):
    r_base, _ = corrFunction(X, Y)
    corrCoefs = []
    for i in range(numberOfBootstraps):
        indices = np.random.choice(np.arange(len(Y)), len(Y), replace=True)
        r, _ = corrFunction(X[indices], Y[indices])
        corrCoefs.append( r )
    lower, upper = np.percentile(corrCoefs, (2.5, 97.5))


    return r_base, upper, lower

def group_correlation_permutation_test(corrFunction, X, Y, groupIndices, numberOfPermutations=int(1e4),
        plot=False):
    numberOfSubjects = (len(groupIndices[0]), len(groupIndices[1]))
    corrObservedAll = corrFunction(X, Y)[0]

    corrObservedGroup1 = corrFunction(X[groupIndices[0]], Y[groupIndices[0]])[0]
    corrObservedGroup2 = corrFunction(X[groupIndices[1]], Y[groupIndices[1]])[0]

    corrObservedDiff = corrObservedGroup1 - corrObservedGroup2

    permutedCorrelationDiffs = np.zeros((numberOfPermutations))
    allIndices = np.arange(len(X))
    for i in range(numberOfPermutations):
        # Shuffle indices
        np.random.shuffle(allIndices)
        permutedGroupIndices = (allIndices[:numberOfSubjects[0]], allIndices[-numberOfSubjects[1]:])
        permutedCorrelationsGroup1 = corrFunction(X[permutedGroupIndices[0]], Y[permutedGroupIndices[0]])[0]
        permutedCorrelationsGroup2 = corrFunction(X[permutedGroupIndices[1]], Y[permutedGroupIndices[1]])[0]
        permutedCorrelationDiffs[i] = permutedCorrelationsGroup1 - permutedCorrelationsGroup2
    
    samplesHigherThanObserved = len(np.where(np.abs(np.array(permutedCorrelationDiffs))>np.abs(corrObservedDiff))[0])

    p_observed = samplesHigherThanObserved / numberOfPermutations
    
    if plot:
        plt.figure()
        sns.distplot(permutedCorrelationDiffs)
        plt.title(f'observed diff = {corrObservedDiff:.2f} which is at p={p_observed:.4f}')
        maxval = np.max(sns.distplot(permutedCorrelationDiffs).get_lines()[0].get_data()[1])
        plt.plot([corrObservedDiff, corrObservedDiff], [0, maxval], 'r')
        plt.show()



    return p_observed