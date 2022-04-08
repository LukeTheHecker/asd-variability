from mne import baseline
import numpy as np
import logging
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy.stats import median_absolute_deviation as mad
from scipy.stats.stats import zscore
import json

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV



from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization #pip install bayesian-optimization

from varname import nameof  # pip install python-varname
from tqdm import tqdm

from scipy.stats import pearsonr
import seaborn as sns

def corr_bootstrap(vec_a, vec_b, corr=None, n_bootstraps=1000, interval=0.95, plot=False):
    if corr is None:
        corr = pearsonr

    lower_perc, upper_perc = [(1-interval)/2, interval + (1-interval)/2]
    r = corr(vec_a, vec_b)[0]

    idc = np.arange(len(vec_a))
    bootstrapped_r = np.zeros(n_bootstraps)
    leave_out_idc = np.random.choice(idc, size=n_bootstraps)
    for i, idx in enumerate(leave_out_idc):
        bootstrapped_r[i] = corr(np.delete(vec_a, idx), np.delete(vec_b, idx))[0]
    lower_limit, upper_limit = np.percentile(bootstrapped_r, [lower_perc*100, upper_perc*100])
    if plot:
        plt.figure()
        sns.distplot(bootstrapped_r)
        ylim=plt.ylim()
        plt.plot([r,r], ylim, color='red')
        plt.title("Bootstrapped correlation")
    return r, lower_limit, upper_limit




def prints(var):
    print(f'{nameof(var)}: {var}')


def euclidean_distance(x, y):
    ''' Calculate euclidean distance between two vectors x and y'''
    return np.sqrt(np.sum((x-y)**2))

def expo(x, a, b, c):
        return a * np.exp(-b * x) + c

def hyperbole(x, a, b, c):
        return a / (x + b) + c

def nearest(arr, val):
    if type(arr) == list:
        arr = np.array(arr)

    assert len(arr) > 1, 'arr contains only 1 value'
    
    diff = arr - val
    return np.argmin(np.abs(diff))

def gof(f, x, y, popt):
    ''' Goodness of fit for function f on points x with fit parameters popt.
        popt is yielded by scipy.optimize.curve_fit
    '''
    residuals = y - f(x, *popt)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    r_square = 1 - (SS_res / SS_tot)
    return r_square


def custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(levelname)s: "
                    "%(message)s")
    log_format = logging.Formatter(format_string)

    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

#edited code from: https://stackoverflow.com/questions/54591352/python-logging-new-log-file-each-loop-iteration/56409646#56409646 (fourth answer)
def log(filename, level=None, format=None):
    '''
    Function to create new log files for each iteration.
    Input: filename, level, format. 
    if no level is set, lowest level will be set. if no format is set, simple message set.
    Output: log object
    '''
    #setting default variables
    if level is None:
        level = logging.DEBUG()

    if format is None:
        format = "%(message)s"

    # https://stackoverflow.com/a/12158233/1995261
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #creating log object
    logger = logging.basicConfig(filename=filename, level=level, format=format,filemode = 'w') #filemode = 'a' is default

    return logger



def concatenate_epochs(epochs_list, with_data=True, add_offset=True):
    """Auxiliary function for concatenating epochs. Ruthlessly stolen from python package MNE. """
    if not isinstance(epochs_list, (list, tuple)):
        raise TypeError('epochs_list must be a list or tuple, got %s'
                        % (type(epochs_list),))
    # for ei, epochs in enumerate(epochs_list):
        # if not isinstance(epochs, epochs_var):
        #     raise TypeError('epochs_list[%d] must be an instance of epochs_var, '
        #                     'got %s' % (ei, type(epochs)))
    out = epochs_list[0]
    data = [out.get_data()] if with_data else None
    events = [out.events]
    metadata = [out.metadata]
    baseline, tmin, tmax = out.baseline, out.tmin, out.tmax
    info = deepcopy(out.info)
    verbose = out.verbose
    drop_log = list(deepcopy(out.drop_log))
    event_id = deepcopy(out.event_id)
    selection = out.selection
    # offset is the last epoch + tmax + 10 second
    events_offset = (np.max(out.events[:, 0]) +
                     int((10 + tmax) * epochs_list[0].info['sfreq']))
    for ii, epochs in enumerate(epochs_list[1:]):
        _compare_epochs_infos(epochs.info, info, ii)
        if not np.allclose(epochs.times, epochs_list[0].times):
            raise ValueError('Epochs must have same times')

        if epochs.baseline != baseline:
            raise ValueError('Baseline must be same for all epochs')

        # compare event_id
        common_keys = list(set(event_id).intersection(set(epochs.event_id)))
        for key in common_keys:
            if not event_id[key] == epochs.event_id[key]:
                msg = ('event_id values must be the same for identical keys '
                       'for all concatenated epochs. Key "{}" maps to {} in '
                       'some epochs and to {} in others.')
                raise ValueError(msg.format(key, event_id[key],
                                            epochs.event_id[key]))

        if with_data:
            data.append(epochs.get_data())
        evs = epochs.events.copy()
        # add offset
        if add_offset:
            evs[:, 0] += events_offset
        # Update offset for the next iteration.
        # offset is the last epoch + tmax + 10 second
        events_offset += (np.max(epochs.events[:, 0]) +
                          int((10 + tmax) * epochs.info['sfreq']))
        events.append(evs)
        selection = np.concatenate((selection, epochs.selection))
        drop_log.extend(epochs.drop_log)
        event_id.update(epochs.event_id)
        metadata.append(epochs.metadata)
    events = np.concatenate(events, axis=0)

    # Create metadata object (or make it None)
    n_have = sum(this_meta is not None for this_meta in metadata)
    if n_have == 0:
        metadata = None
    elif n_have != len(metadata):
        raise ValueError('%d of %d epochs instances have metadata, either '
                         'all or none must have metadata'
                         % (n_have, len(metadata)))
    else:
        pd = True  # _check_pandas_installed(strict=False)
        if pd is not False:
            metadata = pd.concat(metadata)
        else:  # dict of dicts
            metadata = sum(metadata, list())
    if with_data:
        data = np.concatenate(data, axis=0)
    for epoch in epochs_list:
        epoch.drop_bad()
    epochs = epochs_list[0]
    epochs.info = info
    epochs._data = data
    epochs.events = events
    epochs.event_id = event_id
    # epochs.tmin = tmin
    # epochs.tmax = tmax
    # epochs.metadata = metadata
    # epochs.baseline = baseline
    epochs.selection = selection
    epochs.drop_log = drop_log
    # return (info, data, events, event_id, tmin, tmax, metadata, baseline,
    #         selection, drop_log, verbose)
    return epochs

def _compare_epochs_infos(info1, info2, ind):
    """Compare infos."""
    info1._check_consistency()
    info2._check_consistency()
    if info1['nchan'] != info2['nchan']:
        raise ValueError('epochs[%d][\'info\'][\'nchan\'] must match' % ind)
    if info1['bads'] != info2['bads']:
        raise ValueError('epochs[%d][\'info\'][\'bads\'] must match' % ind)
    if info1['sfreq'] != info2['sfreq']:
        raise ValueError('epochs[%d][\'info\'][\'sfreq\'] must match' % ind)
    if set(info1['ch_names']) != set(info2['ch_names']):
        raise ValueError('epochs[%d][\'info\'][\'ch_names\'] must match' % ind)
    if len(info2['projs']) != len(info1['projs']):
        raise ValueError('SSP projectors in epochs files must be the same')
    if any(not _proj_equal(p1, p2) for p1, p2 in
           zip(info2['projs'], info1['projs'])):
        raise ValueError('SSP projectors in epochs files must be the same')
    if (info1['dev_head_t'] is None) != (info2['dev_head_t'] is None) or \
            (info1['dev_head_t'] is not None and not
             np.allclose(info1['dev_head_t']['trans'],
                         info2['dev_head_t']['trans'], rtol=1e-6)):
        raise ValueError('epochs[%d][\'info\'][\'dev_head_t\'] must match. '
                         'The epochs probably come from different runs, and '
                         'are therefore associated with different head '
                         'positions. Manually change info[\'dev_head_t\'] to '
                         'avoid this message but beware that this means the '
                         'MEG sensors will not be properly spatially aligned. '
                         'See mne.preprocessing.maxwell_filter to realign the '
                         'runs to a common head position.' % ind)

def tolerant_mean(arrs, method='tol'):
    ''' method performs mean across traces with differnt number of points based on either of two methods:
    method : str,   'lcd' -> lowest common denominator, i.e. mean is calculated across the lowest number of points
                    'tol' -> tolerant mean, i.e. mean across the highest number of points
    '''
    if method=='tol':
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens),len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[:len(l),idx] = l
        return arr.mean(axis = -1), arr.std(axis=-1)
    elif method == 'lcd':
        min_len = np.min([len(i) for i in arrs])
        pruned_array = np.zeros((len(arrs), min_len,))
        for i, arr in enumerate(arrs):
            pruned_array[i,:] = arr[:min_len]
        return np.mean(pruned_array, axis=0), np.std(pruned_array, axis=0)

def hyperbole_fit(x, a, b, c):
        return a / (x + b) + c   

def multipage(filename, figs=None, dpi=300, png=False):
    ''' Saves all open (or list of) figures to filename.pdf with dpi''' 
    pp = PdfPages(filename)
    path = os.path.dirname(filename)
    fn = os.path.basename(filename)[:-4]
    

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        print(f'saving fig {fig}\n')
        fig.savefig(pp, format='pdf', dpi=dpi)
        if png:
            fig.savefig(f'{path}\\{i}_{fn}.png', dpi=600)
    pp.close()

def print_dict(dct):
    ''' Print out a dictionary '''
    strings = ['Stats\n']
    for key, val in dct.items():
        try:
            keyvalpair = f'{key}: {val:.4}'
        except:
            keyvalpair = f'{key}: {val}'
        print(keyvalpair)
        strings.append(keyvalpair+'\n')
    return ''.join(strings)

def print_omni_dict(dct):
    strings = ['\nStats\n']
    for key, val in dct.items():
        if val != '':
            try:
                keyvalpair = f'{key}: {val:.4}'
            except:
                keyvalpair = f'{key}: {val}'
            print(keyvalpair)
            strings.append(keyvalpair+'\n')
        else:
            strings.append('\n')
    print('\n')
    return ''.join(strings)

def make_Xy(fits_separate, sda = None):
    '''Function that outputs data arrays for the separate fits and/or the sda values. Needed for classification.
    Input: List of fits (and list of sda if needed).
    Output: arrays with data where rows = participants and columns = features'''
    #make X array for fits_separate
    #get lengths of lists for array shapes
    no_of_controls = len(fits_separate[0][0])
    no_of_asd = len(fits_separate[0][1])

    #add together asd and controls
    rows =  no_of_controls + no_of_asd
    #get total number of parameters 
    columnsfits = len(fits_separate)
    #empty array to put list values in to 
    Xfits = np.zeros([rows,columnsfits])

    #counter to index columns of empty array
    columnindex = -1
    #loop through outer most list
    for parameter in fits_separate:
        #counter to index rows of empty array
        rowindex = -1
        #add 1 to column index every time parameter gets looped through to get [0,1,2]
        columnindex += 1
        #loop through group in each parameter
        for group in parameter:
            #loop through values in group to access fits
            for value in group:
                #add 1 to every row index, every time value gets looped through in parameter list to get 0-33 index
                rowindex += 1
                #replace 0 with value that is in list
                Xfits[rowindex,columnindex] = value
    
    #make X array for sda
    if sda != None:
        #get lengths of sda arrays to shape Xsda array
        sdarows = sda[0].shape[0] + sda[1].shape[0]
        sdacolumns = 1
        #empty array to put sda values in to
        Xsda = np.zeros([sdarows, sdacolumns])
        #counter to index rows for empty array
        sdarowindex = -1
        #loop through sda list
        for sdagroup in sda:
            #loop through values in group 
            for sdavalue in sdagroup:
                #add 1 to index
                sdarowindex += 1
                #replace 0 with sdavalue
                Xsda[sdarowindex,0] = sdavalue
    
    #make label array for classification
    #controls labelled as 0
    y_controls = np.zeros([no_of_controls,1]) 
    #asd labelled as 1
    y_asd = np.ones([no_of_asd,1])
    #make into 1 dimesional array
    y = np.append(y_controls,y_asd)
    
    #function should return different variables depending on input
    if sda != None:
        return Xsda, y
    elif sda == None:
        return Xfits, y

def bayesian_opt(X,y):
    '''bayesian optimizer. 
    input: data array and label array
    output: gives you best auc, c, and gamma.'''
    X = X
    y = y
    def estimator(C, gamma):
        '''creates model. needs to be defined within other function because of data (X).
        input: c and gamma in hparams dictionary.
        output: returns the mean auc aka target'''
        # initialize model
        model = SVC(kernel = 'rbf', C=C, gamma=gamma)
        #scaler for data
        scaler = StandardScaler()
        #make pipeline inorder to have scaling be part of cv
        pipeline = make_pipeline(scaler,model)
        # set in cross-validation
        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.125, random_state=70) 
        #set cross_val_score 
        result = cross_val_score(pipeline,X,y,cv=cv,scoring='roc_auc')
        
        # result is mean of aucs
        return np.mean(result)

    #c and gamma ranges
    hparams = {'C': (0.01,1000),'gamma': (0.01,1000)}

    # give model and hyperparameter to optmizer
    svc_bayesopt = BayesianOptimization(estimator, hparams)
    
    #do optimization
    svc_bayesopt.maximize(init_points=5, n_iter=30, acq='ei') #initiation points and number of iterations acq is aquisition method. more iterations bc of rbf kernel
    
    return svc_bayesopt.max #returns best auc and its params as dict


def near(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def argnear(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def running_mean(x, N):
    N = int(round(N))
    return np.convolve(x, np.ones((N,))/N, mode='same')

def bad_segments_to_nan(data, std_seizure, std_factor=3, segmentSize=100):
    ''' This function detects bad segments in data and sets them to np.nan.
    Parameters:
    -----------
    data : ndarray, two dimensional matrix of multichanneldata (chan x time points)
    std_seizure : float, the standard deviation of the flattened eeg signal during a seizure
    std_factor : int/float, standard deviation criterion to reject segments automatically
    prob_bad_chans : float, porportion of channels that need to be bad in order to reject segment
    Return:
    -------
    data_cleaned : ndarray, cleaned matrix of same dimension as data
    '''
    assert len(data.shape) == 2, 'data must have two dimensions'
    assert type(std_factor)==int or type(std_factor)==float, 'std_factor must be of type integer or float.'

    n_timePoints = data.shape[1]
    n_channels = data.shape[0]
    n_segs = int(n_timePoints/segmentSize)

    data_cleaned = data.copy()
    

    for i in tqdm(range(n_segs)):
        data_seg = data[:, i*segmentSize:(i+1)*segmentSize]
        if np.std(data_seg.flatten()) > std_seizure * std_factor:
            data_cleaned[:, i*segmentSize:(i+1)*segmentSize] = np.nan
            

    return data_cleaned

def bad_segment_detection(data_seg, globalSTD, std, prob_bad_chans, n_channels):
    ''' This function takes 
    '''
    StdSegment = np.max(np.abs( data_seg ), axis=1)
    return sum( StdSegment > globalSTD*std ) / n_channels >= prob_bad_chans

def continuous_data_cleaning(data, sfreq, std_seizure, std_factor=3, segmentDur=60):
    ''' Wrapper to clean a set of data
    Paramters:
    ----------
    data : ndarray, two dimensional matrix of multichanneldata (chan x time points)
    sfreq : int, sampling frequency
    std : int/float, standard deviation criterion to reject segments automatically
    prob_bad_chans : float, porportion of channels that need to be bad in order to reject segment
    segmentDur : int, duration (in seconds) of a single segment that, if it contains artefacts, will be set to all "np.nan"s
    
    '''
    segmentSize = int(segmentDur * sfreq)
    print(f'segmentSize={segmentSize}')
    data_cleaned = bad_segments_to_nan(data, std_seizure, std_factor=std_factor, segmentSize=segmentSize)

    n_nans = np.count_nonzero(np.isnan(data_cleaned[1, :]))
    print(f'{(n_nans/(1000/sfreq)) / 60:.1f} out of {(data_cleaned.shape[1]/(1000/sfreq)) / 60:.1f} minutes of data are bad')

    return data_cleaned


def get_sd_threshold(raw, labels, plotme=False):
    data = [list() for _ in range(len(labels))]

    for i, label in enumerate(labels):
        idx_of_segments = np.where(raw._annotations.description == label)[0]
        for j, idx in enumerate(idx_of_segments):
            onset_s = raw._annotations.onset[idx]
            onset_idx = argnear(raw.times, onset_s)
            offset_s = raw._annotations.onset[idx] + raw._annotations.duration[idx]
            offset_idx = argnear(raw.times, offset_s)

            if j == 0:
                data[i] = raw._data[:, onset_idx:offset_idx]
            else:
                data[i]= np.concatenate((data[i], raw._data[:, onset_idx:offset_idx]), axis=1)
    
    # Plotting
    if plotme:

        bins = 1000
        xlim = [np.min([np.min(data[0]), np.min(data[1])]), np.max([np.max(data[0]), np.max(data[1])])]
        plt.figure()
        plt.subplot(211)
        plt.hist(data[0].flatten(), bins=bins)
        plt.xlim(xlim)
        plt.subplot(212)
        plt.hist(data[1].flatten(), bins=bins)
        plt.xlim(xlim)
        
    return [np.std(d.flatten()) for d in data]

def scale_to_freq(scales, sr):
    ''' Convert scale (from multiscale entropy) to the maximum
    frequency captured.
    Parameters:
    -----------
    scales : list, list of scales
    sr : int, sampling rate
    
    Return:
    -------
    freqs : list, list of max frequencies that can be captured.'''
    freqs = []
    for scale in scales:
        freqs.append( (sr/scale) / 2 )
    return freqs


def zscore_blc(x, baselineRange=[0, 10]):
    x = np.array(x)
    x_baselineCorrected = x / np.mean(x[baselineRange[0]:baselineRange[1]])

    x = zscore(x_baselineCorrected)
    
    return x_baselineCorrected

def extract_sequential_itv(meta, idx):
    '''Retrieve some attrubute from the meta class and return the  '''
    RawData = meta.retrieve_attr('sequential_variability')
    data = [list(), list()]
    maxtrials = 9999
    numberOfSubjects = []
    for i, group in enumerate(RawData):
        numberOfSubjects.append(len(group))
        for sub in group:
            subjectData = sub[idx]['data']
            data[i].append(subjectData)
            maxtrials = np.min([ len(subjectData), maxtrials ])

    data = np.concatenate(data, axis=0)
    data = [i[:maxtrials] for i in data]
    labels = np.zeros(len(data)).astype(int)
    labels[-numberOfSubjects[1]:] = 1
    return data, labels

def extract_scd_itv(meta, idx, itvkey='data'):
    '''Retrieve some attrubute from the meta class and return the  '''
    RawData = meta.retrieve_attr('scd_itv')
    data = [list(), list()]
    maxtrials = 9999
    numberOfSubjects = []
    for i, group in enumerate(RawData):
        numberOfSubjects.append(len(group))
        for sub in group:
            subjectData = sub[idx][itvkey]
            data[i].append(subjectData)

    data = np.concatenate(data, axis=0)
    labels = np.zeros(len(data)).astype(int)
    labels[-numberOfSubjects[1]:] = 1
    return data, labels

def get_AQ_EQ(fn, all_data_epoch):
    with open(fn, 'r') as fp:
        aq_eq = json.load(fp)

    keylist = []
    for key, _ in all_data_epoch.items():
        for key2, _ in all_data_epoch[key].items():
            keylist.append(key2)

    controlkeys = [key for key in aq_eq['control'].keys()]
    asdkeys = [key for key in aq_eq['asd'].keys()]


    AQ = []
    EQ = []
    for key in keylist:
        if key in controlkeys:
            vals = aq_eq['control'][key]
        elif key in asdkeys:
            vals = aq_eq['asd'][key]
        else:
            continue
        AQ.append(vals[0])
        EQ.append(vals[1])

    AQ, EQ = (np.array(AQ), np.array(EQ))
    return AQ, EQ

def get_best_svm(X, y, scaler=StandardScaler(), cv=LeaveOneOut):
    if cv is None:
        cv = LeaveOneOut()
    if scaler is None:
        scaler = StandardScaler()
    # define parameter ranges
    params = dict()
    params['C'] = (1e-3, 10.0, 'log-uniform')
    params['gamma'] = (1e-5, 10.0, 'log-uniform')
    params['degree'] = (1,7)
    params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # define the search
    optimized_clf = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv, verbose=0)
    optimized_clf.fit(scaler.fit_transform(X), y)
    # report the best result
    print(f'optimized accuracy: {100*optimized_clf.best_score_:.2f} %')
    print(f'best params: {optimized_clf.best_params_}')
    return optimized_clf.best_params_

def smooth_image(img, n=5):
    img_shape = img.shape
    img_smooth = deepcopy(img)
    lower = 0
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            span_i = [np.clip(i-n, a_min=lower, a_max=img_shape[0]), np.clip(i+n, a_min=lower, a_max=img_shape[0])]
            span_j = [np.clip(j-n, a_min=lower, a_max=img_shape[1]), np.clip(j+n, a_min=lower, a_max=img_shape[1])]
            img_smooth[i, j] = np.mean(img[span_i[0]:span_i[1], span_j[0]:span_j[1]])
    return img_smooth

