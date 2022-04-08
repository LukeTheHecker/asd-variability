''' Scripts for the PRESENCE project that will be migrated into object oriented code.'''
import numpy as np
import matplotlib.pyplot as plt
import string
from numpy.core.numeric import full
from scipy.stats import pearsonr
from scipy.signal import hilbert, detrend, welch

from pactools import Comodulogram
from pactools.dar_model import DAR # pip install pactools

from scripts.signal import interp_nans
from joblib import Parallel, delayed

from itertools import combinations
from tqdm import tqdm

# Run length encoding:

def start_run(x, idx):
    ''' Loops through time series x starting from given idx as long as the values are the same.
    Returns the length of the run and the index of the first element unequal to x[idx]

    Paramters:
    ----------
    x :  list/ndarray, 1D time series vector
    idx : int, starting index 

    Return:
    -------
    runlen : int, length of the run (== number of iterations in loop)
    stop_idx : index of first element where x[idx] != x[stop_idx]
    '''
    # Tests
    assert idx <= len(x), 'Index is higher than time series x. '
    # initialize runlen
    runlen = 0

    for i in range(idx, len(x)):
        if x[i] == x[idx]:
            runlen += 1
        else:
            stop_idx = i
            return runlen, stop_idx

    stop_idx = i+1
    return runlen, stop_idx

def discretize_time_series(x, gridsize):
    ''' Transform a time series x into discretized values.

    Parameters:
    -----------
    x : list/ndarray, 1D time series vector
    gridsize : number of discrete steps

    Return:
    -------
    discretized : discretized time series
    '''
    # Tests
    assert type(x)==list or type(x)==np.ndarray, 'X is wrong data type. It should be list or numpy ndarray'
    assert type(gridsize)==int, 'gridsize must be of type integer'

    lower = np.min(x)
    upper = np.max(x)
    span  = upper - lower
    section_height = span/gridsize
    valranges = []
    for i in range(gridsize):
        valranges.append([i*(section_height) + lower, lower + (section_height) * (i+1)])

    discretized = np.zeros((len(x)))
    for i, val in enumerate(x):
        for grid_idx, valrange in enumerate(valranges):
            # print(f'val={val}')
            if val >= valrange[0] and val <= valrange[1]:
                discretized[i] = grid_idx+1
                break
    return discretized

def run_length_encoding(x, gridsize=2, plotme=False):
    ''' Run length encoding for time series data.

    Parameters:
    -----------
    x : list/ndarray, 1D time series data
    gridsize : int,  number of discretizations of the data. The lower the gridsize,
        the higher the runlength will be.
    
    Return:
    -------
    avg_run_len : float, the mean run length
    '''
    # Tests
    assert type(x)==list or type(x)==np.ndarray, 'X is wrong data type. It should be list or numpy ndarray'
    assert type(gridsize)==int, 'gridsize must be of type integer'
    assert type(plotme)==bool, 'plotme can be either True or False.'

    abc = string.ascii_lowercase
    
    # Calculate a discretized time series
    discretized = discretize_time_series(x, gridsize)
    # Assign labels to each discrete step
    labels = list(abc[int(np.min(discretized))-1:int(np.max(discretized))])
    print(labels)
    # Encode run length
    
    # encoding = ''
    list_of_runs = []
    idx = 0
    while True:
        state = discretized[idx]
        runlen, idx = start_run(discretized, idx)
        list_of_runs.append(runlen)
        # encoding += (labels[int(state-1)] + str(runlen))
        if idx == len(discretized):
            break
    # Plot
    if plotme:
        plt.figure()
        plt.plot(discretized)

    return np.mean(list_of_runs)  #, discretized, encoding 

# Autocorrelation function width:

def autocorrelation(x, corrfun=pearsonr):
    ''' Calculate the autocorrelation of a time series x.
    Paramters:
    ----------
    x : list/ndarray, 1D vector of the time series
    Return:
    -------
    ac : autocorrelation signal
    '''
    assert type(x)==list or type(x)==np.ndarray, 'Input x must be list or numpy array'

    if np.mod(len(x), 2) == 0:
        # if not divisible by 2
        x = np.append(x, x[-1])
        extra = 1
    else:
        extra = 0
    # loop through each lag
    ac = []
    mid_idx = (len(x)-1) / 2
    lags = np.array(np.arange(len(x)) - mid_idx, dtype=int)
    for lag in lags:
        if lag > 0:
            x_tmp = x[lag:]
            x_lag = x[0:-lag]
        elif lag < 0:
            x_tmp = x[0:lag-1]
            x_lag = x[-lag+1:]
        else:
            x_tmp = x
            x_lag = x
        ac.append(corrfun(x_tmp, x_lag)[0])
    return ac[:-extra]

def full_width_half_maximum(x):
    ''' Full width at half maximum calculation.
    Paramters:
    ----------
    x : list/ndarray, time series data, probability distribution
        or anything where fwhm makes sense
    Return:
    -------
    width : float, 
    '''
    half_maximum = np.max(x) / 2
    # Subtract halt maximum so we can search for zero crossings
    x -= half_maximum
    mid_idx = int(round(len(x) / 2))

    # plt.figure()
    # plt.plot(x)

    zero_crossing = np.where(np.diff(np.sign(x)))[0] 
    zero_crossing -= mid_idx  
    if len(zero_crossing) > 2:
        lower_bound = np.abs(np.min(np.abs(zero_crossing[np.where(zero_crossing<0)[0]])))
        upper_bound = np.abs(np.min(np.abs(zero_crossing[np.where(zero_crossing>0)[0]])))
    elif len(zero_crossing) == 1:
        lower_bound = upper_bound = np.abs(zero_crossing)
    elif len(zero_crossing) == 2:
        print(f'zero_crossing={zero_crossing}')
        lower_bound, upper_bound = np.abs(zero_crossing)
    else:
        lower_bound = upper_bound = np.nan
    return lower_bound + upper_bound

def get_acfw(x):
    ac = autocorrelation(x)
    fwhm = full_width_half_maximum(ac)
    return fwhm

# Phase Amplitude Coupling:

def get_pac(x, srate, plot=True):
    ''' Calculate phase-amplitude coupling of a given time series x.
    Parameters:
    -----------
    x : list, time series data
    srate : int, sampling frequency 

    '''
    
    dar = DAR(ordar=20, ordriv=2, criterion='bic')
    low_fq_range = np.linspace(0.01, 10, 30)
    high_fq_range= np.linspace(10, 30, 20)
    low_fq_width = 1.0  # Hz
    estimator = Comodulogram(fs=srate, low_fq_range=low_fq_range,
                         low_fq_width=low_fq_width, method=dar,
                         high_fq_range=high_fq_range, progress_bar=True, 
                         n_jobs=-1)
    estimator.fit(x)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        estimator.plot(axs=[ax])
        ax.set_title('Comodulogram')
        plt.show()
        
    return estimator.comod_

# Phase Synchrony

def phase_synchrony(y1, y2):
    ''' Calculate phase synchrony between two signals.
    Parameters:
    -----------
    y1 : float, bandpass-filtered time series data
    y2 : float, bandpass-filtered time series data of same length as y1

    Return:
    -------
    mean_angle : float, phase synchrony index
    '''
    # Get Phase for each signal
    phase1 = np.angle(hilbert(y1))
    phase2 = np.angle(hilbert(y2))
    # Get phase difference
    phasediff = phase1 - phase2
    # Calculate length of average vector

    mean_angle = 0
    for i, th in enumerate(phasediff):
        mean_angle += np.e**(1j*th)

    mean_angle = np.abs(mean_angle / len(phasediff))

    return mean_angle

#average of phase synchrony for all electrode pairs:

def average_phase_synchrony(electrodeslist):
    ''' Calculate the average phase synchrony of all electrode pairs.
        Input:
        -----------
        electrodeslist: List of data arrays for each electrode. 
        Structure of array like parameters in phase_synchrony. 
        
        Output:
        --------
        avg_ps: Float, Average phase synchrony index of all pairs of electrodes
    '''
    
    #empty dictionary where key is index of electrode list of each pair
    phasesync_dict = {}

    #get all possible pairs of electrodes
    perm = combinations(range(len(electrodeslist)), 2) 

    #loop through all pairs
    for i, j in list(perm):
        #key of dictionary is index of electrode in list
        key = f'{i}, {j}'
        #calculate the phase synchrony for pair
        phasesync_dict[key] = phase_synchrony(electrodeslist[i], electrodeslist[j])

    #put all phase synchrony values into list
    values = [val for key, val in phasesync_dict.items()] 
    #get average phase synchrony 
    avg_ps = np.mean(values)
    
    return avg_ps

# Useful window function:

def windowed(x, fun, winsize, *args, padding=True, overlap=0.5, parallel=False, nancrit=0.10, **kwargs):
    ''' Apply a function on time series x using multiple windows of certain size.
    If nans are found within a window the metric will NOT be computed on said window and the result will be nan then.
    Parameters:
    -----------
    x : float, time series data
    fun : function, the function that will be applied numerous times to the data
    winsize : int, size of the window
    padding : bool, if True zero padding will be applied such that the output is as long as x.
    overlap : float, if float between 0 and 1, the windows will overlap by proportion of winsize. 
        If None, windows will have no overlap.
    nancrit : float, proportion of nans that are acceptable for interpolation
    Return:
    -------
    out : float, result of the windowed operation
    '''
    
    # Tests
    assert type(winsize) == int, 'winsize must be a positive integer'
    assert overlap is None or type(overlap) == float, 'overlap must be float or None'

    
    x = np.atleast_2d(np.array(x))


    # If winsize is even: make it odd:
    if np.mod(winsize, 2) == 0:
        winsize += 1

    # Get half the window size for later
    halfWindow = int((winsize-1)/2)
    
    # Output vector
    try:
        outLen = len(fun(np.random.randn(100), *args, **kwargs))
    except:
        outLen = 1

    # out = np.empty(x[0, :].shape)
    # out[:] = np.NaN
    print(x.shape[1])
    nans = [np.nan]*outLen
    out = [nans for _ in range(x.shape[1])]

    # Calculate indices of the data 
    # at which the function will be applied:
    if overlap is not None:
        # Calculate the step size based on overlapping proportion
        # Assure that the step size is at least 1 and maximal the winsize
        step = np.clip(int(winsize-round(winsize*overlap)), 1, winsize)
        indicesOfCalculation = np.arange(halfWindow, x.shape[1]-halfWindow, step).astype(int)
    else:
        indicesOfCalculation = np.arange(halfWindow, x.shape[1]-halfWindow, winsize).astype(int)
    
    # Since sometimes we have np.nans in the data we need to exclude those segments here:
    idxWhereNan = []  # indices of windows that contain at least one nan
    idxWhereInterpolated = []
    for i, index in reversed(list(enumerate(indicesOfCalculation))):
        # Get data of current window
        x_tmp = x[:, index-halfWindow:index+halfWindow]
        # Get number of nans within that window
        n_nans = np.sum([np.sum(np.isnan(arr)) for arr in x_tmp])
        # Calculate the proportion of nans in the window
        porportion_of_nans = n_nans / x_tmp.size
        if n_nans > 0 and porportion_of_nans > nancrit:
            indicesOfCalculation = np.delete(indicesOfCalculation, i)
            idxWhereNan.append(i)
        elif n_nans > 0 and porportion_of_nans <= nancrit:
            # print(f'{i}: only {n_nans} of {x_tmp.size} points (->{porportion_of_nans*100:.1f} %) are nans. Interpolating')
            idxWhereInterpolated.append(i)
            for j in range(x.shape[0]):
                x[j, index-halfWindow:index+halfWindow] = interp_nans(x[j, index-halfWindow:index+halfWindow])
                if any(np.isnan(x[j, index-halfWindow:index+halfWindow])):
                    msg = f'interpolation did not work on index {index}'
                    raise NameError(msg)



    print(f'The function {fun.__name__} will be applied for {len(indicesOfCalculation)} windows of size {winsize}, averaged across {x.shape[0]} channels.\n{len(idxWhereNan)} windows are left out because of nans.\n{len(idxWhereInterpolated)} windows were interpolated due to few nans.')

    # Perform Calculation in parallel
    if parallel:
        if fun.__name__ != 'average_phase_synchrony':
            result = []
            # loop through channels
            for ch in range(x.shape[0]):
                tmp = Parallel(n_jobs=-1, backend='loky')(delayed(fun)
                    (x[ch, index-halfWindow:index+halfWindow], *args, **kwargs) for index in tqdm(indicesOfCalculation) )
                result.append(tmp)
            result = np.stack(result, axis=0)
            print(f'result of all indices and channels has shape {result.shape}')
            result = np.median(result, axis=0)
        else:
            result = np.stack(Parallel(n_jobs=-1, backend='loky')(delayed(fun)
                (x[:, index-halfWindow:index+halfWindow], *args, **kwargs) for index in tqdm(indicesOfCalculation) ))
        for i, idx in enumerate(indicesOfCalculation):
            out[idx] = result[i]
    else: # Not parallel
        if fun.__name__ != 'average_phase_synchrony':
            for index in tqdm(indicesOfCalculation):
                tmp_result = []
                # Loop through channels
                for ch_idx in range(x.shape[0]):
                    if not any(np.isnan(x[ch_idx, index-halfWindow:index+halfWindow])):
                        tmp_result.append( fun(x[ch_idx, index-halfWindow:index+halfWindow], *args, **kwargs) )
                    else:
                        
                        nanArr = np.empty(outLen)
                        nanArr[:] = np.nan
                        tmp_result.append(nanArr)

                out[index] =  np.median(tmp_result, axis=0)
        else:
            for index in tqdm(indicesOfCalculation):
                x_tmp = x[:, index-halfWindow:index+halfWindow]

                if np.any(np.isnan(x_tmp)):
                    out[index] = np.nan
                else:
                    out[index] = fun(x_tmp, *args, **kwargs)

    out = np.array(out).T
    # Remove edges where no calculation was possible
    if not padding:
        out = out[halfWindow:len(x)-halfWindow]
    
    # Interpolate missing values
    print('interping...')
    if len(np.squeeze(out).shape) > 1:
        for i, scalePack in enumerate(out):
            scalePack[scalePack==np.inf] = np.nan
            scalePack = interp_nans(scalePack)
            out[i, :] = scalePack
    else:
        print(list(out)[0][0])
        out = np.array([i[0] if type(i)==list else i for i in list(out)])
        out = interp_nans(out)
    print('\t...done!')
    return out

# Some features

def detrended_std(x):
    x_detrended = detrend(x, type='linear')
    dstd = np.std(x_detrended)
    return dstd

# Get Segments - might be useful later when we go object-oriented:

def get_segments(data, segments, selection):
    ''' Extract selected segments of data, e.g. of prodomal sections.
    Parameters:
    -----------
    data : list/ndarray, 2D data matrix of shape channels x time points.
    segments : dict, dictionary that contains different states and 
        corresponding time ranges.
    selection : the desired selection in the segments dictionary.
    
    Example:

    segments = {
    'Prodrome': ((100, 500), (1100, 2500), (3400, 6400)),
    'Seizure': ((500, 700), (2900, 3000))
        }
    selection = 'Prodrome'
    prodromes = get_segments(y, segments, selection)
    '''
    
    # Input checks
    assert selection in segments, 'selected state in argument selection can not be found in segments'
    assert type(data) == list or type(data) == np.ndarray, 'data must be of type list or np.ndarray'
    assert type(segments) == dict, 'argument segments must be of type dict '
    assert len(data.shape) == 2, 'data must be a 2D matrix of chans x time points.' 

    out_data = []
    time_spans = segments[selection]
    for time_span in time_spans:
        pnt_range = range(time_span[0], time_span[1])
        out_data.append(data[:, pnt_range])
    return out_data

def freq_band_power(data, sfreq, freqband):
    freqs, psd = welch(data, sfreq, nperseg=len(data))

    idx_lower, idx_upper = [np.argmin(np.abs(freqs-freq)) for freq in freqband]
    psd_band = np.mean(psd[idx_lower:idx_upper])
    return psd_band