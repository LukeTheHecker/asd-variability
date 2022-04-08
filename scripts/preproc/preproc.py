from scripts.signal import *
from scripts.stat import *
from scripts.util import *
from scripts.viz import *
from scripts.classes import epochs_var

from pyentrp.entropy import multiscale_entropy as mse
from pyentrp.entropy import multiscale_permutation_entropy as pmse
from tqdm import tqdm
import pandas as pd
import numpy as np
import mne
from scipy.stats import pearsonr
from scipy.stats import median_absolute_deviation as mad
from scipy.signal import find_peaks, welch
from autoreject import Ransac  # noqa
import logging

def get_events(onoff, annotations, srate, round_to, triglens, triglabels, mode='checkers'):
    ''' Extract Events from annotations using trigger lengths '''

    a = annotations[0]
    e = np.zeros((a.shape[0], a.shape[1]+1))
    e[:,0:3] = a
    sample_factor = 1000 / srate
    if mode=='checkers':
        # how it used to be:
        for i in range(e.shape[0]):
            if e[i, 2] == onoff[0]:
                j = i + 1
                while j < e.shape[0]:
                    if e[j, 2] == onoff[1]:
                        if round_to == 1:
                            e[i, 3] = (e[j , 0] - e[i,0]) * sample_factor
                        else:
                            e[i, 3] = (np.round((e[j,0] - e[i,0])/round_to)*round_to) * sample_factor

                        if e[i, 3] in triglens:
                            idx = triglens.index(e[i, 3])
                            e[i,2] = triglabels[idx]
                        break
                    j+=1

    elif mode=='lattices':
        # Iterate through all events "e"
        for i in range(e.shape[0]-1):
            # If On-trigger found
            if e[i, 2] == onoff[0] and e[i+1, 2] == 1:
                j = i + 1
                # serch for Off-trigger
                while j < e.shape[0]-1:
                    # If Off-trigger found
                    if e[j, 2] == onoff[1] and e[j+1, 2] == 2:
                        # Store the trigger length in ms at On-trigger 
                        if round_to == 1:
                            e[i, 3] = (e[j , 0] - e[i,0]) * sample_factor
                        else:
                            e[i, 3] = (np.round((e[j,0] - e[i,0])/round_to)*round_to) * sample_factor
                        # If trigger length is of interest
                        if e[i, 3] in triglens:
                            # get index in list of trigger lengths
                            idx = triglens.index(e[i, 3])
                            k = i - 1
                            # go back to stimulus onset in order to place the stimulus code correctly
                            while True:
                                if e[k, 2] == onoff[0] and e[k+1, 2] == 1:
                                    e[k, 2] = triglabels[idx]
                                    break
                                k -= 1
                            else:
                                print(f'{e[i, 3]} not in triglens')
                        break
                    j += 1
    return np.ndarray.astype(e[:,0:3],'int32')

def load_prepare_data(pth, refelec, filt, perform_ICA=True, logger=None, rm_bad_chans=False, n_jobs=-1):
    projection = False
    # Load
    raw = mne.io.read_raw_brainvision(pth, preload=True, verbose=0)
    # filter
    if logger is not None:
        logger.info(f'Filtering at {filt[0]} to {filt[1]} Hz')
    raw.filter(filt[0], filt[1], n_jobs=n_jobs, verbose=0)
    # Read montage using older mne version (0.19)
    # montage = mne.channels.read_montage(kind='standard_1020', ch_names=raw.ch_names,
    #                                     transform=True)
    # raw.set_montage(montage)

    # Read montage using newer mne version (0.20)
    raw.set_montage('standard_1020')

    # Detect Bad electrodes
    # chan_list using older version of mne (0.19)
    # chan_list = [[montage.ch_names[i], montage.pos[i, :]] for i in range(len(montage.ch_names))]
    
    # chan_list using newer version of mne (0.20)
    

    if rm_bad_chans:
        chan_list = get_chan_pos_list(raw)
        raw = detect_bad_chans(raw, chan_list)
        bads = raw.info['bads']
        if logger is not None:
            logger.info(f'Bad channels: {bads}')
        # Remove Bad electrodes
        if not not bads:
            raw.interpolate_bads()
            logger.info(f'Bad channels removed & replaced by interpolation.')

    # Remove VEOG
    try:
        mne.io.Raw.drop_channels(raw, "VEOG")
        if logger is not None:
            logger.info(f'removing VEOG')
    except:
        print('channel already dropped')

    # # Common Average Reference
    # if 'average' in refelec:
    #     projection = True
    # else:
    #     projection = False
    if logger is not None:
        logger.info(f'Re-reference: {refelec}')
    raw.set_eeg_reference(refelec, projection=projection, verbose=0)  # set EEG average reference

    

    if perform_ICA:
        raw_ica = deepcopy(raw.copy())
        if raw_ica.info['highpass'] < 0.9:
            raw_ica.filter(1., None, n_jobs=n_jobs)

        method = 'fastica'  # Common Method
        # Initialize & fit
        ica = mne.preprocessing.ICA(n_components=25, method=method)
        ica.fit(raw_ica)

        eog_epochs = mne.preprocessing.create_eog_epochs(raw_ica, ch_name='Fp1')
        eog_inds, _ = ica.find_bads_eog(eog_epochs, ch_name='Fp1')  # find via correlation
        ica.exclude = eog_inds
        raw = ica.apply(raw)
        logger.info(f'ICA removed {len(eog_inds)} components ({eog_inds})')
        
    return raw

def epoch_clean(raw, baseline, trial_time, onoff, art_thresh, srate, round_to, triglens, 
                triglabels, isi=False, mode='checkers', logger=None, 
                stim_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                csd=False, use_ransac=False):
    # define artifacts
    if art_thresh is not None:
        reject = dict(eeg=float(art_thresh) * 1e-6)
    else:
        reject = None
    annotations = mne.events_from_annotations(raw)

    if len(triglens) == len(triglabels):
        event = get_events(onoff, annotations, srate, round_to, triglens, triglabels, mode=mode)

    
    if isi:
        isi_ls = []
        sample_factor = 1000. / srate
        for i in range(event.shape[0]):
            if event[i, 2] == triglabels[0]:
                for j in range(i+1, event.shape[0]):
                    if event[j, 2] == 3 and event[j+1, 2] == 1:
                        isi_ls.append((event[j, 0]-event[i, 0]) * sample_factor)
                        break
        print(f'isi: {np.mean(isi_ls)} +- {np.std(isi_ls)}')
        print(f'isi: {np.mean(isi_ls)} +- {np.std(isi_ls)}')
        print(f'isi: {np.mean(isi_ls)} +- {np.std(isi_ls)}')

    # Pick only stimulus onsets:
    event_id = {}
    for i, tl in enumerate(np.unique(triglabels)):
        event_id[stim_names[i]] = tl
    # RANSAC
    if use_ransac:
        epochs = raw_to_epochs(raw)   # we have to transform Raw to Epochs bc. the bcr only works with Epochs (Raw is 2D (channels x time), Epochs is 3D (trials x channels x time))
        ransac = Ransac(verbose=0, n_jobs=-1)
        epochs_clean = ransac.fit_transform(epochs)
        raw._data = np.squeeze(epochs_clean.get_data())  # after cleaning up, we transform Epochs back into Raw by "removing" the empty dimension (here: trials. We only have one trial though. trials = 1. Kann man eigentlich weglassen, deshalb "empty")
        logger.info(f'Interpolated channel indices: {ransac.bad_chs_ }')
        # epochs = ransac.fit_transform(epochs)

    # Create Epochs
    
    if logger is not None:
        logger.info(f'Epoching Settings: tmin={trial_time[0]}, tmax={trial_time[1]}, baseline={baseline}, rejection crit={art_thresh}')
    # epochs = mne.Epochs(raw, event, event_id=event_id, tmin=trial_time[0], tmax=trial_time[1], 
    #                     proj=True, baseline=baseline,  
    #                     preload=True, reject=reject)
    
    epochs = epochs_var(raw, event, event_id=event_id, tmin=trial_time[0], tmax=trial_time[1], 
                        proj=True, baseline=baseline,  
                        preload=True, reject=reject)
    if csd:
        epochs = mne.preprocessing.compute_current_source_density(epochs)
    
    if logger is not None:
        logger.info(f'Number of trials per condition:')
        for i in range(len(np.unique(triglabels))):
            logger.info(f'{stim_names[i]}: {epochs[stim_names[i]]._data.shape[0]}')
    if isi:
        return epochs , event, annotations[0], isi_ls
    else:
        return epochs

def detect_bad_chans(raw, chan_list_ordererd, crit=0.1, radius=0.075, zthresh=100):
    ''' Detect bad channels and return list of strings 
    radius...in meters within which neighboring electrodes are defined
    crit... neighbor correlation criterion. If maximum correlation with any neighbor is below this crit the channels is marked as bad
    '''
    
    
    # Handle exceptions
    assert len(chan_list_ordererd) == len(raw.ch_names), 'Chan list has different number of channels than there are channels in raw'

    # Correlation Criterion "CC"
    bad_chans_cc = bad_chan_cc(raw, chan_list_ordererd, crit, radius)

    # Deviation Criterion "DC"
    bad_chans_dc = bad_chan_dc(raw, chan_list_ordererd, zthresh)
    
    # Summarize
    bad_chans = np.unique(np.concatenate((bad_chans_cc, bad_chans_dc)))
    
    # Bad Channel info
    bads_info = []
    for i in bad_chans:
        if i in bad_chans_cc and i in bad_chans_dc:
            bads_info.append('Corr & Deviation')
        elif i in bad_chans_cc and not i in bad_chans_dc:
            bads_info.append('Corr')
        else:
            bads_info.append('Deviation')

    raw.info['bads'] = bad_chans
    raw.info['bads_info'] = bads_info
    return raw

def bad_chan_dc(raw, chan_list_ordererd, zthresh):
    bad_chans = []
    
    total_median = np.median(raw._data)
    total_mad = mad(raw._data)

    for i in range(len(raw.ch_names)):
        data = raw._data[i, :]
        # Robust Z-Scaling
        zdata = (data - total_median) / total_mad
        # Median Z-score
        zscore = np.median(np.abs(zdata))
        # If high amplitudes compared to rest:
        if zscore > zthresh:
            bad_chans.append(raw.ch_names[i])

    return bad_chans

def bad_chan_cc(raw, chan_list_ordererd, crit, radius):
    ''' This function checks the correlation criterion (cc) as a base for rejecting bad channels: 
        If a channel is not sufficiently correlated with its close neighbors, it is probably bad.'''
    bad_chans = []
    # Loop through each channel
    for i in range(len(raw.ch_names)):
        # print(f'Current electrode: {raw.ch_names[i]}')
        # Get position of current channel
        pos = chan_list_ordererd[i][1]
        # Find neighbors
        distlist = np.zeros((len(raw.ch_names)-1))
        pos_list = [ls[1] for j, ls in enumerate(chan_list_ordererd) if j != i]
        pos_list = np.array(pos_list)

        idx_list = [j for j in range(len(chan_list_ordererd)) if j != i]

        for j in range(pos_list.shape[0]):
            distlist[j] = euclidean_distance(pos_list[j, :], pos)
        neighbor_indices = np.argwhere(distlist<=radius)
        neighbor_indices = [k[0] for k in neighbor_indices]
        neighbor_indices = [idx_list[k] for k in neighbor_indices]
        # ...
        # Perform Correlations
        corrs = []
        for k, index in enumerate(neighbor_indices):
            corrs.append(pearsonr(raw._data[i, :], raw._data[index, :])[0])
            # print(f'index: {index}, Correlation between {raw.ch_names[i]} and {raw.ch_names[index]} = {corrs[k]:.4f}')
        
        if np.max(np.abs(corrs)) < crit:
            print(f'{raw.ch_names[i]} is bad!')
            bad_chans.append(raw.ch_names[i])

    return bad_chans

def get_chan_pos_list(raw, montage_type='standard_1020'):
    ''' Returns a list of channel names + respective position in 
    the same order as they occur in the raw structure'''
    if type(montage_type) == str:
        montage = mne.channels.make_standard_montage(montage_type)
    else:
        montage = montage_type
    n_chan = len(montage.ch_names)
    chan_assign = {}
    for i in range(3, n_chan+3):
        chan_assign[montage.dig[i]['ident']] = montage.ch_names[i-3]
    # print(chan_assign)
    ###
    chan_list = [list() for i in range(len(raw.ch_names))]

    for i in range(3, len(raw.info['dig'])):
        # print(raw.info['dig'][i])
        current_identitiy = raw.info['dig'][i]['ident']
        # print(f'Current Identitiy: {current_identitiy}')
        chan_list[i-3].append(chan_assign[current_identitiy])
        # print(f'Channel name: {chan_assign[current_identitiy]}')
        chan_list[i-3].append(raw.info['dig'][i]['r'])
        pos = raw.info['dig'][i]['r']
        # print(f'Channel position: {pos}\n')

    chan_list_ordererd = chan_list.copy()
    # Loop through channel names in raw data
    for j, chan in enumerate(raw.ch_names):
        # Find the correct row in chan_list
        for row in chan_list:
            if row[0] == chan:
                # And copy that row to the correctly ordered list
                chan_list_ordererd[j] = row
    # print(chan_list_ordererd)
    return chan_list_ordererd

def make_montage(csvname, rawdata):
    '''Function to create 64-channel montage.
    Input: excel spreadsheet name with all coordinate info in meters including reference points, raw data object.
    Output: montage object'''
    
    # import excel spreadsheet as dataframe
    csv = pd.read_csv(csvname, sep=';')
    # access cartesian coordinates of electrodes
    cart_csv = csv.iloc[:-4, :]
    print(f'cart_csv: {cart_csv}')
    # put coordinates into dictionary for mne function
    dig_mont = {}
    for i in range(len(cart_csv)):
        dig_mont[cart_csv.iloc[i][0]] = cart_csv.iloc[i][1:]
    keys = [i for i in dig_mont.keys()]
    for i in range(len(dig_mont)):
        dig_mont[keys[i]] = dig_mont[keys[i]].values.astype(float) / 1000

    # list of electrode names
    ch_names=list(dig_mont.keys())

    # import reference points from excel for mne function
    reference_points = csv.iloc[64:, 1:]
    print(f'reference_points: {reference_points}')

    # put cartesian coordinates of reference into arrays
    inion = np.array(reference_points.iloc[1],dtype=float) / 1e3
    nasion = np.array(reference_points.iloc[0],dtype=float) / 1e3
    lpa = np.array(reference_points.iloc[2],dtype=float) / 1e3
    rpa = np.array(reference_points.iloc[3],dtype=float) / 1e3
    # remove extra channels except stim
    rawdata.pick_types(rawdata, eeg=True, exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'])
    
    # make dictionary of old and new electrode names for mne function
    mapping = {element1: element2 for element1, element2 in zip(rawdata.ch_names, ch_names)}
    # rename channels
    rawdata.rename_channels(mapping)

    # make montage with info collected above
    montage = mne.channels.make_dig_montage(dig_mont, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown')
    
    return montage

def remove_blinks(epoch, raw):
    epoch_ica = deepcopy(epoch)
    epoch_ica.filter(1., None, verbose=0)
    raw_ica = deepcopy(raw.copy())
    raw_ica.filter(1., None, verbose=0)
    
    random_state = 23  # just so we get the same results.
    n_components = 25
    method = 'fastica'  # Common Method
    # Initialize & fit
    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state, verbose=0)
    ica.fit(raw_ica)
    # Find artefactual components based on Fp1 activity
    eog_epochs = mne.preprocessing.create_eog_epochs(raw_ica, ch_name='Fp1')
    print(f'what happens now?')
    eog_inds, _ = ica.find_bads_eog(eog_epochs, ch_name='Fp1')  # find via correlation
    ica.exclude = eog_inds
    print(f'\n\ndeleting index {eog_inds}\n\n')
    epoch_clean = epoch.copy()
    ica.apply(epoch_clean)
    return epoch_clean, ica

def get_isolated_component(idx, epoch, ica):
    ica_copy = deepcopy(ica.copy())
    epoch_copy = deepcopy(epoch.copy())

    indices_to_delete = np.arange(ica_copy.n_components_)
    # Select all components except the desired one
    indices_to_delete = np.delete(indices_to_delete, idx)
    print('keeping only idx ', idx)
    ica_copy.include = [idx]
    ica_copy.exclude = indices_to_delete

    ica_copy.apply(epoch_copy)
    return epoch_copy

def calc_itv(epoch, cond=None, thresh=100e-6, norm_type=None, bad_policy='keep',
    relative=False, prestim_range=(-0.2, 0), verbose=0):
    ''' Calculate the inter-trial variability for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    norm_type : normalize the itv by the root mean square (rms) of the subjects ERP.
    bad_policy : str, defines the handling of bad trials. 
        'keep'-> keep bad trials, replaced as np.nan
        'dump'-> delete bad trials

    Return:
    -------
    itv : numpy.ndarray, inter-trial variability
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)

    if norm_type == 'rms':
        norm = np.sqrt(np.nanmean(np.square(data)))
        data /= norm
    elif norm_type == 'md':
        norm = np.nanmean(np.abs(data - np.nanmedian(data)))
        data /= norm
    elif norm_type == 'z':
        norm = np.nanmean(np.abs(data - np.nanmedian(data)))
        data = (data - np.nanmean(data)) / np.nanstd(data)
    
    


    
    itv = np.nanstd(data, axis=0)
    
    if relative:
        pnt_range = np.arange(*[np.argmin(np.abs(epoch.times - t)) 
                    for t in prestim_range])
        itv = np.stack([ch / np.mean(ch[pnt_range]) for ch in itv], axis=0)
        
    return itv
    
def calc_etv(epoch, cond=None, winsize=10, thresh=100e-6, rms_norm=False, md_norm=False, 
    verbose=0, bad_policy='keep', relative=False, cv=False):
    ''' Calculate the evolving (windowed) inter-trial variability for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    cond : str, defines the condition to use from the epoch (epoch[cond])
    winsize : int, number of trials per window
    thresh : float/int, defines the artifact rejection criterion in Volt
    rms_norm : normalize the itv by the root mean square (rms) of the subjects ERP.
    verbose : int/bool, verbosity of the function
    bad_policy : str, defines the handling of bad trials. 
        'keep'-> keep bad trials, replaced as np.nan
        'dump'-> delete bad trials

    Return:
    -------
    itv : numpy.ndarray, inter-trial variability
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()
    
    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    
    if rms_norm:
        data /= np.sqrt(np.nanmean(np.square(data)))
    elif md_norm:
        norm = np.nanmean(np.abs(data - np.nanmedian(data)))
        data /= norm

    n_tr, n_elec, n_time = data.shape

    windows = np.lib.stride_tricks.sliding_window_view(np.arange(n_tr), (winsize))
    n_windows = windows.shape[0]
    

    etv = np.zeros((n_windows, n_elec, n_time))
    for i, window in enumerate(windows):
        data_slice = data[window, :, :]
        etv[i] = np.nanstd(data_slice, axis=0)

    if relative:
        etv = np.stack([ch / ch[0] for ch in etv], axis=0)
    return etv

def get_itc(epoch, ch_name='Oz', freqrange=[3, 30], cond=None, thresh=100e-6, 
    verbose=0):

    epoch = deepcopy(epoch.copy())

    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy='dump', thresh=thresh, verbose=verbose)
    epoch.data = data
    
    # Channel handling
    if ch_name is None:
        ch_indices = None
    else:
        if type(ch_name) == str:
            ch_name = [ch_name]
    
        ch_indices = [epoch.ch_names.index(ch) for ch in ch_name]

    freqs = np.logspace(*np.log10(freqrange), num=50)
    n_cycles = freqs / 3. 
    kwargs = dict(n_jobs=-1,
                # return_itc=True,
                # average=True,
                # picks=ch_indices,
                use_fft=True,
                output='itc')
    # _, itc = mne.time_frequency.tfr_morlet(epoch, freqs, n_cycles, **kwargs)
    itc = mne.time_frequency.tfr_array_morlet(data[:, ch_indices, :], epoch.info['sfreq'], freqs, n_cycles=n_cycles, **kwargs)

    
    return np.mean(itc, axis=0)

def get_power(epoch, ch_name='Oz', freqrange=[3, 30], cond=None, thresh=100e-6, 
    verbose=0, induced=True):

    epoch = deepcopy(epoch.copy())

    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy='dump', thresh=thresh, verbose=verbose)
    epoch.data = data
    
    # Channel handling
    if ch_name is None:
        ch_indices = None
    else:
        if type(ch_name) == str:
            ch_name = [ch_name]
    
        ch_indices = [epoch.ch_names.index(ch) for ch in ch_name]

    freqs = np.logspace(*np.log10(freqrange), num=50)
    n_cycles = freqs / 3. 
    kwargs = dict(n_jobs=-1,
            # return_itc=False,
            # average=False,
            # picks=ch_indices,
            use_fft=True,
            output='power')
    # power = mne.time_frequency.tfr_morlet(epoch, freqs, n_cycles, **kwargs)
    if induced:
        power = mne.time_frequency.tfr_array_morlet(data[:, ch_indices, :], epoch.info['sfreq'], freqs, n_cycles=n_cycles, **kwargs)
    else:
        ERP = np.expand_dims(np.mean(data[:, ch_indices, :], axis=0), axis=0)
        power = mne.time_frequency.tfr_array_morlet(ERP, epoch.info['sfreq'], freqs, n_cycles=n_cycles, **kwargs)
        

    return power

def get_itpv(epoch, ch_name='Oz', freqrange=[3, 30], cond=None, thresh=100e-6, 
    verbose=0):

    epoch = deepcopy(epoch.copy())

    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy='dump', thresh=thresh, verbose=verbose)
    epoch.data = data
    
    # Channel handling
    if ch_name is None:
        ch_indices = None
    else:
        if type(ch_name) == str:
            ch_name = [ch_name]
    
        ch_indices = [epoch.ch_names.index(ch) for ch in ch_name]

    freqs = np.logspace(*np.log10(freqrange), num=50)
    n_cycles = freqs / 3. 
    kwargs = dict(n_jobs=-1,
            return_itc=False,
            average=False,
            picks=ch_indices,
            use_fft=True)
    power = mne.time_frequency.tfr_morlet(epoch, freqs, n_cycles, **kwargs)
    
    return np.mean(np.squeeze(np.std(power.data, axis=0)), axis=0)

def handle_bad_trials(data, bad_policy='keep', thresh=100e-6, verbose=0):
    data = deepcopy(data)

    bad_trials = find_bad_trials(data, thresh=thresh, verbose=verbose)
    if bad_policy=='keep':
        data[bad_trials, :, :] = np.nan
    else:
        data = np.delete(data, bad_trials, axis=0)
    return data

def calc_pmse(epoch, cond=None, sample_length=3, delay=2, scale=30, verbose=0):
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    sf = epoch.info['sfreq']

    bad_trials = find_bad_trials(data, thresh=100e-6, verbose=verbose)
    data[bad_trials, :, :] = np.nan
    data_clean = np.delete(data, bad_trials, axis=0)
    data_clean = np.concatenate(data_clean, axis=1)
    multiscale_entropy = np.array([pmse(d, sample_length, delay, scale=scale) for d in tqdm(data_clean)])
    return multiscale_entropy    
    
def find_bad_trials(data, thresh=200e-6, verbose=0):
    bad_trials = []
    for trial in range(data.shape[0]):
        if np.max(np.abs(data[trial])) > thresh:
            bad_trials.append(trial)
    if verbose==1:
        print(f'bad_trials: {bad_trials}')
    return bad_trials


def var_quench(epoch, cond=None, thresh=100e-6, bad_policy='keep',
    verbose=0, cv=False, prestim_range=(-0.2, 0), poststim_range=(0.15, 0.5),
    itv_based=True):
    ''' Calculate the inter-trial variability for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    bad_policy : str, defines the handling of bad trials. 
        'keep'-> keep bad trials, replaced as np.nan
        'dump'-> delete bad trials
    verbose : int/bool, verbosity of the program
    itv_based : bool, defines how the quenching is defined. 
        True -> Ratio of itv of prestim to itv of poststim interval 
        False -> Ratio of intra-trial prestim variability to intra-trial poststim variability
    cv : bool, coefficient of variation; normalizes the quenching by 
        area under the amplitudes
    Return:
    -------
    quench : float, variabilty quenching
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    
    if prestim_range  is not None and poststim_range is not None:
        times = epoch.times
        prestim_range = np.where((times>prestim_range[0]) & (times<=prestim_range[1]))[0]
        poststim_range = np.where((times>poststim_range[0]) & (times<=poststim_range[1]))[0]
    else:
        prestim_range = np.where(epoch.times < 0)[0]
        poststim_range = np.where(epoch.times >= 0)[0]
    
    if itv_based:
        itv_prestim = np.nanmean(np.nanstd(data[:, :, prestim_range], axis=0), axis=-1)
        itv_poststim = np.nanmean(np.nanstd(data[:, :, poststim_range], axis=0), axis=-1)

        if cv:
            erp = np.nanmean(data, axis=0)
            norm_itv_prestim = itv_prestim / np.trapz(np.abs(erp[:, prestim_range]), axis=-1)
            norm_itv_poststim = itv_poststim / np.trapz(np.abs(erp[:, poststim_range]), axis=-1)
            quench = norm_itv_prestim / norm_itv_poststim
        else:
            quench = itv_prestim / itv_poststim
    else:
        
        prestim_intra_trial_variance = np.nanmean(data[:, :, prestim_range], axis=2)
        poststim_intra_trial_variance = np.nanmean(data[:, :, poststim_range], axis=2)

        quench = np.nanmean(np.divide(prestim_intra_trial_variance, poststim_intra_trial_variance), axis=0)
        
    return quench


def pm_method(data, n_perm=5):
    data = first_dim_even(data)
    n_tr = data.shape[0]
    sign_flip = np.ones((n_tr))
    sign_flip[:int(n_tr/2)]  *= -1
    noise_ERP = np.zeros((n_perm, data.shape[1], data.shape[2]))
    for i in range(n_perm):
        np.random.shuffle(sign_flip)
        noise_ERP[i] = np.nanmean(np.stack([trial * sign for trial, sign in zip(data, sign_flip)], axis=0), axis=0)
    return np.mean(noise_ERP, axis=0)

def subtract_ERP_method(data):
    erp = np.nanmean(data, axis=0)
        
    data_subtracted = np.nanmean([trial - erp for trial in data], axis=0)
    
    return data_subtracted

def first_dim_even(data):
    if np.mod(data.shape[0], 2) != 0:
        idx = np.random.choice(np.arange(data.shape[0]))
        data = np.delete(data, idx, axis=0)
    return data


def get_highest_peak(peaks, data):
    if len(peaks)==1:
        return peaks[0]
    elif len(peaks) == 0:
        return 0
    else:
        peak_amplitudes = [data[peak] if not not peak else 0 for peak in peaks]
        return peaks[np.argmax(peak_amplitudes)]


def p100_milne(epoch, time_range=(0.1, 0.17), cond=None, 
    thresh=100e-6, verbose=0, bad_policy='keep', baseline=(-0.1,0), ch_name=None):
    ''' Calculate the evolving (windowed) inter-trial variability 
    for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    time_range : tuple, specifies the beginning and end in seconds of the peak search 
    ch_name : str, the channel of interest
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    verbose : int/bool, verbosity of the function
    bad_policy : str, defines the handling of bad trials. 
        'keep'-> keep bad trials, replaced as np.nan
        'dump'-> delete bad trials

    Return:
    -------
    cv : float, coefficient of variation at p100
    '''
    epoch = deepcopy(epoch.copy())
    
    if baseline is not None:
        epoch = epoch.apply_baseline(baseline=baseline)
    epoch = epoch.crop(*time_range)
    # Find channels
    if ch_name is None:
        epoch_csd = mne.preprocessing.compute_current_source_density(epoch.copy())
        csd_data = epoch_csd.get_data().mean(axis=0,)
        channel_peak_list = []
        latency_list = []
        for ch_idx, ch_name in enumerate(epoch_csd.ch_names):
            peaks, _ = find_peaks(csd_data[ch_idx])
            peak = get_highest_peak(peaks, csd_data[ch_idx])
            channel_peak_list.append(csd_data[ch_idx, peak])
            latency_list.append(epoch.times[peak])
        ch_idx = np.argmax(channel_peak_list)
        ch_name = epoch_csd.ch_names[ch_idx]
        latency = latency_list[ch_idx]

        # fig = epoch.average().plot_topomap(latency)
        

        print('BEST CHANNEL: ', ch_name)
        
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()
    
    # Handle channel selection
    if type(ch_name) == list:
        ch_idc = np.array([epoch.ch_names.index(ch) for ch in ch_name])
    elif type(ch_name) == str:
        ch_idc = epoch.ch_names.index(ch_name)
    else:
        raise ValueError("ch_name must be list of names or single channel name. Got {} instead".format(ch_name))

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    if isinstance(ch_idc, (list, np.ndarray)):
        data = np.mean(data[:, ch_idc, :], axis=1)
    else:
        data = data[:, ch_idc, :]

    erp = np.nanmedian(data, axis=0)
    # erp = np.nanmean(data, axis=0)
    
    # peaks, _ = find_peaks(erp)
    # erp_peak = get_highest_peak(peaks, erp)
    # p100_erp = erp[erp_peak]

    trial_peak_amplitudes = []
    # plt.figure()
    for trial in data:
        # trial_peak_amplitudes.append(np.max(trial))
        peaks, _ = find_peaks(trial)
        highest_peak = get_highest_peak(peaks, trial)
        trial_peak_amplitudes.append(trial[highest_peak])

    
    # plt.figure()
    # plt.plot(np.nanmean(data, axis=0))

    median = np.nanmedian(trial_peak_amplitudes)
    peak_variability = np.nanmedian(np.abs(trial_peak_amplitudes-median))
    cv = peak_variability / median
    
    return  cv


def spontaneous_activity(epoch, ch_name=None, cond=None, 
    thresh=100e-6, verbose=0, baseline=(-0.1, 0), norm_type='z',
    bad_policy='keep'):

    epoch = deepcopy(epoch.copy())
    if baseline is not None:
        epoch = epoch.apply_baseline(baseline=baseline)

    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    ch_idx = epoch.ch_names.index(ch_name)
    
    if norm_type == 'z':
        data = (data-np.nanmean(data))/np.nanstd(data)
    elif norm_type == 'erp':
        erp = np.nanmean(data, axis=0)
        data = data / np.nanmean(np.abs(erp))

    erp = np.nanmean(data, axis=0)
    for i_tr, trial in enumerate(data):
        data[i_tr] = trial - erp

    data = np.abs(data)
    erp_spontaneous = np.nanmean(data, axis=0)
    
    
    return erp_spontaneous[ch_idx]


def cross_trial_amplitude_var(epoch, ch_name=None, cond=None, 
    thresh=100e-6, verbose=0, baseline=(-0.1, 0), z_norm=True, erp_norm=False,
    bad_policy='keep'):

    epoch = deepcopy(epoch.copy())
    if baseline is not None:
        epoch = epoch.apply_baseline(baseline=baseline)

    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    if z_norm:
        data = (data-np.nanmean(data))/np.nanstd(data)
    if erp_norm:
        erp = np.nanmean(data, axis=0)
        data = data / np.nanmean(np.abs(erp))

    # Handle channel selection
    if type(ch_name) == list:
        ch_idc = np.array([epoch.ch_names.index(ch) for ch in ch_name])
    elif type(ch_name) == str:
        ch_idc = epoch.ch_names.index(ch_name)
    else:
        raise ValueError("ch_name must be list of names or single channel name. Got {} instead".format(ch_name))

    if isinstance(ch_idc, (list, np.ndarray)):
        data = np.nanmean(data[:, ch_idc, :], axis=1)
    else:
        data = data[:, ch_idc, :]
    variability = np.zeros(data.shape[-1])
    for i in range(len(variability)):
        # variability[i] = np.nanmedian(np.abs(data[:, i] - np.nanmedian(data[:, i])))
        variability[i] = np.nanstd(data[:, i])
    return variability



def p100_cv(epoch, time_range=(0.08, 0.17), ch_name=None, cond=None, 
    thresh=100e-6, verbose=0, bad_policy='keep', plotme=False,
    cv_norm=True, z_norm=False):
    ''' Calculate the evolving (windowed) inter-trial variability 
    for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    time_range : tuple, specifies the beginning and end in seconds of the peak search 
    ch_name : str, the channel of interest
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    verbose : int/bool, verbosity of the function
    bad_policy : str, defines the handling of bad trials. 
        'keep'-> keep bad trials, replaced as np.nan
        'dump'-> delete bad trials

    Return:
    -------
    cv : float, coefficient of variation at p100
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    if z_norm:
        data = (data-np.nanmean(data)) / np.nanstd(data)

    erp = np.nanmean(data, axis=0)
    pnt_range = np.arange(np.argmin(np.abs(epoch.times-time_range[0])), 
                    np.argmin(np.abs(epoch.times-time_range[1])))
     
    if ch_name is not None and type(ch_name) != list:
        # print(type(ch_name))
        ch_idx = epoch.ch_names.index(ch_name)
        peaks, _ = find_peaks(erp[ch_idx, pnt_range])
        peak = get_highest_peak(peaks, erp[ch_idx, pnt_range])
    else:
        if type(ch_name) == list:
            ch_idc = np.array([epoch.ch_names.index(ch) for ch in ch_name])
            print(ch_name, ch_idc)
            erp = erp[ch_idc]
        # find peak per channel
        peaks = [find_peaks(chan_erp[pnt_range])[0] for chan_erp in erp]
        # Get only the largest ones
        peaks = [get_highest_peak(peak, dat) if len(peak)>0 else None for peak, dat in zip(peaks, erp[:, pnt_range])]
        ch_idx = np.argmax( [data[peak] if peak is not None else 0 for peak, data in zip(peaks, erp[:, pnt_range])] )
        peak = peaks[ch_idx]
        if verbose==1:
            print(ch_idx)
            print(f'Selected channel {np.array(epoch.ch_names)[ch_idc][ch_idx]}')
        if peak is None:
            return np.nan
    if plotme:
        plt.figure()
        plt.plot(epoch.times, erp[ch_idx, :])
        plt.scatter(epoch.times[peak+pnt_range[0]], erp[ch_idx, peak+pnt_range[0]])

    # Calculate coefficient of variartion
    if cv_norm:
        cv = np.nanstd(data[:, ch_idx, peak+pnt_range[0]], axis=0) / erp[ch_idx, peak+pnt_range[0]]
    else:
        cv = np.nanstd(data[:, ch_idx, peak+pnt_range[0]], axis=0)

    return np.abs(cv)

def p100(epoch, time_range=(0.08, 0.17), ch_name=None, cond=None, 
    thresh=100e-6, verbose=0, bad_policy='keep', plotme=False,
    z_norm=False):
    ''' Calculate the evolving (windowed) inter-trial variability 
    for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    time_range : tuple, specifies the beginning and end in seconds of the peak search 
    ch_name : str, the channel of interest
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    verbose : int/bool, verbosity of the function
    bad_policy : str, defines the handling of bad trials. 
        'keep'-> keep bad trials, replaced as np.nan
        'dump'-> delete bad trials

    Return:
    -------
    cv : float, coefficient of variation at p100
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()

    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    if z_norm:
        data = (data-np.nanmean(data)) / np.nanstd(data)

    erp = np.nanmean(data, axis=0)
    pnt_range = np.arange(np.argmin(np.abs(epoch.times-time_range[0])), 
                    np.argmin(np.abs(epoch.times-time_range[1])))
     
    if ch_name is not None and type(ch_name) != list:
        # print(type(ch_name))
        ch_idx = epoch.ch_names.index(ch_name)
        peaks, _ = find_peaks(erp[ch_idx, pnt_range])
        peak = get_highest_peak(peaks, erp[ch_idx, pnt_range])
    else:
        if type(ch_name) == list:
            ch_idc = np.array([epoch.ch_names.index(ch) for ch in ch_name])
            print(ch_name, ch_idc)
            erp = erp[ch_idc]
        # find peak per channel
        peaks = [find_peaks(chan_erp[pnt_range])[0] for chan_erp in erp]
        # Get only the largest ones
        peaks = [get_highest_peak(peak, dat) if len(peak)>0 else None for peak, dat in zip(peaks, erp[:, pnt_range])]
        ch_idx = np.argmax( [data[peak] if peak is not None else 0 for peak, data in zip(peaks, erp[:, pnt_range])] )
        peak = peaks[ch_idx]
        if verbose==1:
            print(ch_idx)
            print(f'Selected channel {np.array(epoch.ch_names)[ch_idc][ch_idx]}')
        if peak is None:
            return np.nan
    if plotme:
        plt.figure()
        plt.plot(epoch.times, erp[ch_idx, :])
        plt.scatter(epoch.times[peak+pnt_range[0]], erp[ch_idx, peak+pnt_range[0]])

    
    return erp[ch_idx, peak+pnt_range[0]]

def rms_epoch(epoch, cond=None, bad_policy='dump', thresh=100e-6, verbose=0):
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()
    
    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    
    n_tr, n_elec, n_time = data.shape

    rms_tr = np.stack([rms(x, axis=1) for x in data], axis=1)
    return rms_tr

def rms(x, axis=None):
    return np.sqrt(np.nanmean(np.array(x)**2, axis=axis))

def calc_raw_rms(epoch, conds=None, ch_picks=None):

    if ch_picks is None:
        ch_indices = np.arange(len(epoch.ch_names))
    else:
        ch_indices = [epoch.ch_names.index(ch_pick) for ch_pick in ch_picks]
    
    if conds is not None:
        data = epoch[conds].get_data()[:, ch_indices, :]
    else:
        data = epoch.get_data()[:, ch_indices, :]

    data = handle_bad_trials(data, bad_policy='dump', thresh=100e-6, verbose=0)
    data = zeroing(data)
    return rms(data)

def calc_erp_rms(epoch, conds=None, ch_picks=None):

    if ch_picks is None:
        ch_indices = np.arange(len(epoch.ch_names))
    else:
        ch_indices = [epoch.ch_names.index(ch_pick) for ch_pick in ch_picks]

    if conds is not None:
        data = epoch[conds].get_data()[:, ch_indices, :]
    else:
        data = epoch.get_data()[:, ch_indices, :]

    data = handle_bad_trials(data, bad_policy='dump', thresh=100e-6, verbose=0)
    data = zeroing(data)
    return rms(np.nanmean(data, axis=0))

def zeroing(x):
    x = deepcopy(x)
    if len(x.shape) == 3:
        for i in range(x.shape[1]):
            x[:, i, :] -= np.mean(x[:, i, :])
    elif len(x.shape) == 2:
        for i in range(x.shape[0]):
            x[i, :] -= np.mean(x[i, :])
    return x


def calc_n_tr(epoch, conds=None, ch_picks=None):
    
    if ch_picks is None:
        ch_indices = np.arange(len(epoch.ch_names))
    else:
        ch_indices = [epoch.ch_names.index(ch_pick) for ch_pick in ch_picks]

    if conds is not None:
        data = epoch[conds].get_data()[:, ch_indices, :]
    else:
        data = epoch.get_data()[:, ch_indices, :]
    
    data = handle_bad_trials(data, bad_policy='dump', thresh=100e-6, verbose=0)

    return data.shape[0]

def get_beta_exponent(data, sfreq=500, freqband='all', plotme=False):
    freqs, psd = welch(data, sfreq, nperseg=len(data))

    if type(freqband) == str:
        if freqband == 'all':
            idx_lower, idx_upper = [0, len(freqs)]
    elif type(freqband) == list:
        idx_lower, idx_upper = [np.argmin(np.abs(freqs-freq)) for freq in freqband]
    
    # Check if idx_lower starts with 0 Hz (not allowed)
    if freqs[idx_lower] == 0:
        idx_lower += 1
        
    beta = np.polyfit(np.log(freqs[idx_lower:idx_upper]), np.log(psd[idx_lower:idx_upper]), 1)[0]

    if plotme:
        plt.figure()
        plt.loglog(freqs[idx_lower:idx_upper], psd[idx_lower:idx_upper])

    return np.abs(beta)


def topographic_consistency(epoch, cond=None, thresh=100e-6, winsize=0.1, 
    bad_policy='dump', time_resolved=True, mode='cosine', verbose=0):
    ''' Calculate the inter-trial variability for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    winsize : float/int, size of sliding window in seconds
    mode : str, 'cosine': use cosine similarity among scalp maps between trials (Schurger et al., 2015)
        'gfp': use global field power for topographic similarity (Bender et al., 2015)
    Return:
    -------
    directed_coherence : float, directed_coherence, i.e. how coherent trials are among each other (without amplitudes involved)
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()
    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    n_tr, n_elec, n_time = data.shape
    # Extract time windows

    winsize_pnts = int(np.clip(epoch.info['sfreq'] * winsize, a_min=1, a_max=np.inf))

    if np.mod(winsize_pnts, 2) == 0:
        winsize_pnts += 1

    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=winsize_pnts, axis=-1)

    topo_sim = np.zeros((windows.shape[2]))
    if mode == 'cosine':
        for i in range(windows.shape[2]):
            window = windows[:, :, i, :]
            window_avg = np.mean(window, axis=-1).T
            window_avg_normed = window_avg / np.linalg.norm(window_avg, axis=0)
            cov = np.matmul(window_avg_normed.T, window_avg_normed)
            topo_sim[i] = np.mean(cov[np.tril_indices(cov.shape[0])])
    elif mode == 'gfp':
        data_norm = deepcopy(data)
        for i_tr in range(n_tr):
            for i_time in range(n_time):
                data_norm[i_tr, :, i_time] = data[i_tr, :, i_time] / np.std(data[i_tr, :, i_time])
        
        topo_sim = np.std(np.mean(data_norm, axis=0), axis=0)
    else:
        msg = f'mode {mode} is not known. Use either <cosine> or <gfp> instead.'
        raise ValueError(msg)
        
    return topo_sim

def gain_consistency(epoch, cond=None, thresh=100e-6, winsize=0.1, 
    bad_policy='dump', verbose=0):
    ''' Calculate the inter-trial variability for one participant.
    Parameters:
    -----------
    epoch : mne.Epochs or similar, object containing the trials of one or many conditions
    cond : str, defines the condition to use from the epoch (epoch[cond])
    thresh : float/int, defines the artifact rejection criterion in Volt
    winsize : float/int, size of sliding window in seconds

    Return:
    -------
    directed_coherence : float, directed_coherence, i.e. how coherent trials are among each other (without amplitudes involved)
    '''
    epoch = deepcopy(epoch.copy())
    if cond is not None:
        data = epoch[cond].get_data()
    else:
        data = epoch.get_data()
    
    data = handle_bad_trials(data, bad_policy=bad_policy, thresh=thresh, verbose=verbose)
    n_tr, n_elec, n_time = data.shape
    # Extract time windows

    winsize_pnts = int(np.clip(epoch.info['sfreq'] * winsize, a_min=1, a_max=np.inf))
    
    if np.mod(winsize_pnts, 2) == 0:
        winsize_pnts += 1

    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=winsize_pnts, axis=-1)

    scale_norms_mean = np.zeros((windows.shape[2]))
    for i in range(windows.shape[2]):
        window = windows[:, :, i, :]
        window_avg = np.mean(window, axis=-1).T
        norms = np.linalg.norm(window_avg, axis=0)
        scale_norms_mean[i] = np.std(norms) #/ np.mean(norms)

    return scale_norms_mean

def raw_to_epochs(raw):
    ''' Convert raw data object to an epochs data object
    Parameters
    ----------
    raw : mne.io.Raw, the raw data object from MNE
    
    Return
    ------
    epochs : mne.Epochs, the epochs data object from MNE
    '''
    epochs = mne.EpochsArray(np.expand_dims(raw._data, axis=0), raw.info)
    return epochs