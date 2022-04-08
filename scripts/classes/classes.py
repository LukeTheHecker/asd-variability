
from scripts.stat.stat import variance
from scripts.util.util import print_dict, smooth_image
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import Epochs
from mne.io import Raw
from numba import jit
from copy import Error, deepcopy

from scipy.optimize import curve_fit
from scipy.stats import sem, ttest_ind
from scripts.util import nearest, hyperbole, gof, tolerant_mean, make_Xy, bayesian_opt, prints
from scripts.viz import plot_hyperbole, scatter_plot, plot_roc, plot_two_with_error
from scripts.stat import permutation_test, omni_stat, ranksums

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from scipy.stats import median_absolute_deviation as mad

from nolds import dfa, sampen
from pyentrp.entropy import multiscale_entropy, composite_multiscale_entropy

from joblib import Parallel, delayed

from itertools import combinations

class Base:
    ''' Base class with basic extended functionality inherited to e.g. epochs_var and raw_var.''' 


    def status(self, attribute):
        ''' Ask for status of an attribute.
        Parameters:
        -----------
        attribute : str, name of the attribute (e.g. 'scd_itv', 'chaos', etc)
        
        Return:
        -------
        Nothing, just printouts.
        '''
        
        # Exception handling
        if not hasattr(self, attribute):
            msg = f'attribute {attribute} not found.'
            raise AttributeError(msg)

        print('ITV is present with the following settings:')
        if getattr(self, attribute) == []:
            msg = f'attribute {attribute} is empty. (== [])'
            raise AttributeError(msg)
        for i in range(len(getattr(self, attribute))):
            print(f'\n\nIndex {i}:\n')
            # print(self.itv[i]['info'])
            print_dict(self.itv[i]['info'])
        
    def pop(self, index, attribute):
        ''' Delete entry of the list self.itv. To see all entries type self.status_itv() 
        Parameters:
        -----------
        index : intor str, index in the attribute list to choose from. index can be 'all' 
            to delete all indices. Use meta.check_common_itv(attribute) to find desired index.
        attribute : attribute to delete index from
        '''
        
        # Exception handling
        if not hasattr(self, attribute):
            msg = f'Attribute {attribute} not in instance.'
            raise AttributeError(msg)

        # Delete index
        if type(index) == str:
            if index == 'all':
                setattr(self, attribute, [])
        elif type(index) == int:
            assert type(index) == int or type(index) == float, 'index needs to be an integer, a whole number float or a string that says all'
            data = getattr(self, attribute)
            data.pop(index)
            setattr(self, attribute, data)
        else:
            msg = f'Wrong data type {type(attribute)}. Use integer or str <all> to delete all indices.'
            raise TypeError(msg)
        return 'pop'

class BaseMeta:
    ''' Basic Functions that are useful for the meta classes (e.g. meta_var) in which
    multiple "Epochs" or "epochs_var" objects are handled.''' 
    def _consistency_check(self):
        ''' Consistency check of all attributes'''
        self._test_attribute_equality('event_id')
        self._test_attribute_equality('ch_names')
        self._test_attribute_equality('times')
        self._test_attribute_equality('info')
        print('Consistency Check passed.')

    def _test_attribute_equality(self, attribute):
        ''' Tests whether the attribute is equal in all Epochs.
        Parameters:
        -----------
        attribute : str, the name of the Attribute (see https://mne.tools/dev/generated/mne.Epochs.html for a list of all attributes)
        '''
        failure = False
        attributes = deepcopy(self.retrieve_attr(attribute))
        # If attribute is info pop some of the individual keys
        if attribute == 'info':
            for group in attributes:
                for sub in group:
                    if 'bads' in sub:
                        sub.pop('bads')  
                    if 'meas_date' in sub:
                        sub.pop('meas_date')
                    if 'chs' in sub:
                        sub.pop('chs')
                    if 'bads_info' in sub:
                        sub.pop('bads_info')
                    if 'projs' in sub:
                        sub.pop('projs')
        attributes_concat = []
        for ls in attributes:
            attributes_concat += ls
        equal = [i == attributes_concat[0] for i in attributes_concat]

        # Handle excpetion when single attribute is a list
        if type(equal[0]) == bool:
            if not all(equal):
                failure = True
        elif type(equal[0]) == list:
            if not all([all(i) for i in equal]):
                failure = True

        if failure:
            msg = f'attribute {attribute} is not equal in all Epochs.'
            raise ElementsNotEqualError(msg)

    def retrieve_attr(self, attribute):
        ''' Loop through all_epochs dictionary and get individual attributes.
        Parameters:
        -----------
        attribute : str, name of some attribute
        
        Return:
        -------
        data : list, list of attributes in lists of groups
        '''
        # Retrieve some attribute or the result of some function1
        keys = [key for key in self.all_epochs.keys()]
        data = [list(), list()]
        # Loop Groups
        for i, group in enumerate(keys):
            subkeys = [key for key in self.all_epochs[group].keys()]
            # Loop Participants in Groups
            for j, sub in enumerate(subkeys):
                # Check if attribute is there
                if not hasattr(self.all_epochs[group][sub], attribute):
                    msg = (f'Subject {sub} in {group} does not have the attribute {attribute}')
                    raise AttributeError(msg)
                component = getattr(self.all_epochs[group][sub], attribute)

                data[i].append(component)
        return data
    
    # def call_fun(self, method, *args, **kwargs):
    #     ''' Loop through all_epochs dictionary and call a given method.
    #     Parameters:
    #     -----------
    #     method : str, name of some method
        
    #     Return:
    #     -------
    #     data : list, list of attributes in lists of groups
    #     '''
    #     # Call method for each data set in 'all_epochs'
    #     keys = [key for key in self.all_epochs.keys()]
    #     retr = [list(), list()]
    #     # Loop Groups
    #     for i, group in enumerate(keys):
    #         print(f'\nProcessing Group: {group}\n')
    #         subkeys = [key for key in self.all_epochs[group].keys()]
    #         # Loop Participants in Groups
    #         # Parallel
    #         np.stack(Parallel(n_jobs=-1, backend=backend)(delayed(vec_to_sevelev_newlayout)(i) for i in x_noise))
            
    #         for j, sub in enumerate(subkeys):
    #             print(f'Processing subject {j+1}/{len(subkeys)}')
    #             if not method in dir(self.all_epochs[group][sub]):
    #                 msg = (f'Subject {sub} in {group} does not have method {method}')
    #                 raise NotImplementedError(msg)

    #             data = getattr(self.all_epochs[group][sub], method)(*args, **kwargs)
    #             retr[i].append(data)
    #     return retr


    def call_fun(self, method, *args, **kwargs):
        backend='loky'
        # Call method for each data set in 'all_epochs'
        keys = [key for key in self.all_epochs.keys()]
        retr = [list(), list()]
        if 'get_' in method:
            nameOfAttribute = method[4:]
            parallel = True
        else:
            nameOfAttribute = ''
            parallel = False

        print('#########################')
        print(f'Calling {method}() on each subject.\n')

        # Loop Groups
        for i, group in enumerate(keys):
            print(f'\nProcessing Group: {group}\n')
            subkeys = [key for key in self.all_epochs[group].keys()]
            # Loop Participants in Groups
            #======================#
            # ...in parallel
            if parallel:
                retr[i].append(np.stack(Parallel(n_jobs=-1, backend=backend)(delayed(getattr(self.all_epochs[group][sub], method))(*args, **kwargs) for sub in subkeys)))
                retr[i] = retr[i][0]
                # Since this parallelization has a problem with storing the calculated attributes in the epochs_var.attr we have to do that quickly:
                for s, sub in enumerate(subkeys):
                    currentAttribute = getattr(self.all_epochs[group][sub], nameOfAttribute)
                    newAttribute = retr[i][s]
                    currentAttribute.append(newAttribute)
                    setattr(self.all_epochs[group][sub], nameOfAttribute, currentAttribute)
            else:
                #======================#
                # ...sequentially:
                for j, sub in enumerate(subkeys):
                    if not method in dir(self.all_epochs[group][sub]):
                        msg = (f'Subject {sub} in {group} does not have method {method}')
                        raise NotImplementedError(msg)

                    data = getattr(self.all_epochs[group][sub], method)(*args, **kwargs)
                    retr[i].append(data)
        return retr

    def call_external_fun(self, fun, *args, parallel=True, **kwargs):
        backend='loky'
        # Call method for each data set in 'all_epochs'
        keys = [key for key in self.all_epochs.keys()]
        retr = [list(), list()]


        print('#########################')
        print(f'Calling {fun.__name__}() on each subject.\n')

        # Loop Groups
        for i, group in enumerate(keys):
            print(f'\nProcessing Group: {group}\n')
            subkeys = [key for key in self.all_epochs[group].keys()]
            # Loop Participants in Groups
            #======================#
            # ...in parallel
            if parallel:
                retr[i].append(np.stack(Parallel(n_jobs=-1, backend=backend)(delayed(fun)(self.all_epochs[group][sub], *args, **kwargs) for sub in subkeys)))
                retr[i] = retr[i][0]
            else:
                #======================#
                # ...sequentially:
                for j, sub in enumerate(subkeys):
                    data = fun(self.all_epochs[group][sub], *args, **kwargs)
                    retr[i].append(data)
        return retr

    def clean_itv(self):
        ''' This function will tidy up all the itv calculations within subjects that are not shared among >>all<< subjects, leaving only common ITVs.
            This function helps to keep the data organized'''
            # TODO: Implement!
        assert 1==0, 'This function is not properly implemented yet!'

        kgroup = [key for key in self.all_epochs.keys()]
        ksub = [key for key in self.all_epochs[kgroup[0]].keys()]  # just temporary
        # The template contains ITV of one subject which will be compared with all the others.
        template_itv = self.all_epochs[kgroup[0]][ksub[0]].itv

        info = template_itv[0]['info']

        template_itv_info = []
        for i in template_itv:
            tmp_info = i.copy()
            tmp_info.pop('n_tr', None)
            template_itv_info.append(tmp_info)

        # template_itv_info = [i['info'][:end_of_info] for i in template_itv]

        
        common_itv_info = template_itv_info.copy()
        for i, group in enumerate(kgroup):
            ksub = [key for key in self.all_epochs[group].keys()]
            for j, sub in enumerate(ksub):
                tmp_itv = self.all_epochs[group][sub].itv
                tmp_itv_info = [i['info'][:end_of_info] for i in tmp_itv]

                info_not_there = []
                for k, info in enumerate(template_itv_info):
                    # info_not_there.append(not (info in tmp_itv_info))
                    info_not_there.append(info != tmp_itv_info)

                idx_where_not_present = np.where(info_not_there)[0]
                for idx in sorted(idx_where_not_present, reverse=True):
                    common_itv_info.pop(idx)
                    template_itv_info.pop(idx)
        
        # Print Common ITVs:
        if len(common_itv_info) > 0:
            print('ITV is present with the following settings:')
            for i in range(len(common_itv_info)):
                print(f'\n\nIndex {i}:\n')
                # print(common_itv_info[i])
                print_dict(common_itv_info[i])
        else: 
            # print(f'common_itv_info: {common_itv_info}')
            # print(f'template_itv_info: {template_itv_info}')
            print('common_itv_info: \n')
            print_dict(common_itv_info)
            print('\ntemplate_itv_info:\n')
            print_dict(template_itv_info)

    def check_common_itv(self, attr):
        ''' This function loops through all subjects to check whether there are common ITVs (i.e. with same settings).
        Parameters:
        -----------
        attr : str, e.g. scd_itv
        
        Return:
        -------
        No value returned. Creates printout of common attributes
        '''
        assert type(attr) == str, 'attr must be a string, e.g. scd_itv'

        kgroup = [key for key in self.all_epochs.keys()]
        ksub = [key for key in self.all_epochs[kgroup[0]].keys()]  # just temporary

        # The template contains ITV of one subject which will be compared with all the others.
        template_itv = getattr(self.all_epochs[kgroup[0]][ksub[0]], attr)

        info = template_itv[0]['info']

        template_itv_info = []
        for i in template_itv:
            tmp_info = i['info']
            tmp_info.pop('n_tr', None)
            template_itv_info.append(tmp_info)


        
        common_itv_info = template_itv_info.copy()
        for i, group in enumerate(kgroup):
            ksub = [key for key in self.all_epochs[group].keys()]
            for j, sub in enumerate(ksub):
                if i==0 and j==0:
                    # Skip very first sub since it is the template
                    continue
                tmp_itv = getattr(self.all_epochs[group][sub], attr)
                tmp_itv_info = []
                for m in tmp_itv:
                    tmp_info = m['info']
                    tmp_info.pop('n_tr', None)
                    tmp_itv_info.append(tmp_info)

                info_not_there = []
                for k, info in enumerate(template_itv_info):
                    info_not_there.append(not(info in tmp_itv_info))

                idx_where_not_present = np.where(info_not_there)[0]
                for idx in sorted(idx_where_not_present, reverse=True):
                    common_itv_info.pop(idx)
                    template_itv_info.pop(idx)
        
        # Print Common ITVs:
        if len(common_itv_info) > 0:
            print('ITV is present with the following settings:')
            for i in range(len(common_itv_info)):
                print(f'\n\nIndex {i}:\n')
                print_dict(common_itv_info[i])
        else: 
            print('No Common ITVs')

class epochs_var(Epochs, Base):
    '''Child of mne.Epochs, extended with functions for variability analysis
    
    '''
    def __init__(self, *args, **kwargs):
        ''' Call Parent __init__ and some extra things'''
        super(epochs_var, self).__init__(*args, **kwargs)

        self.itv = []  # Intertrial Variance using fit method
        self.scd_itv = []  # Intertrial Variance using smallest common denominator
        self.sequential_variability = []  # Variability evolving over ordered trials
        self.chaos = []

    def get_scd_itv(self, condition, time_range, n_tr=None, time_resolved=False, 
                                ch_name='O1', k=100, variance_metric=np.std):
        ''' Compare Intertrial Variability based on RMS over given number of trials.
        Use the get_scd_itv method in the meta_var class for a smallest common denominator
        analysis.

        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        time_range : list, [low, high] in seconds
        time_resolved : bool, False: averaged over time_range; True: variability for 
            each time point
        ch_name : str, channel name or 'all' to calculate for all channels
        n_tr : int/None, if None the smalles common denominator is chosen, 
            otherwise the given integer is chosen
        k : int, number of resampling
        variance_metric : function, e.g. np.std, scipy.stats.median_absolute_deviation, scipy.stats.variation
        '''
        obj = self.__getitem__(condition)
        data = obj.get_data()
        pnt_range = [nearest(self.times, time_range[0]), nearest(self.times, time_range[1])]
        ch_idx = self.ch_names.index(ch_name)
        fits = (None, None)

        info = dict(condition=condition, time_range=time_range, time_resolved=time_resolved, ch_name=ch_name, iterations=k, n_tr=n_tr, variance_metric=variance_metric.__name__)
        if variance_metric == dfa:
            print(f'metric is {variance_metric}, thus order of trials is not shuffled/ resampled.')
            keep_order = True
            sd = np.array([variance_metric(data[:n_tr, ch_idx, i]) for i in np.arange(pnt_range[0], pnt_range[1])])
            if not time_resolved:
                sd = np.mean(sd)
        else:
            keep_order = False

        if not keep_order:
            # If number of trials is None or equal or larger than the number of available trials: Take all of them
            if n_tr is None or (n_tr >= data.shape[0]):
                # if subject has exactly n_tr trials
                sd = np.array([variance_metric(data[:, ch_idx, i]) for i in np.arange(pnt_range[0], pnt_range[1])])
                # sd = np.std(data[:, ch_idx, pnt_range[0]:pnt_range[1]], axis=0)
                if not time_resolved:
                    sd = np.mean(sd)
                else:
                    x = np.arange(len(sd))
                    fits = tuple(np.polyfit(x, sd, 1))
            else:
                # Else: resample!
                sd_resample = []
                tr_arr = np.arange(data.shape[0])
                # Iterate over k for resampling
                for i in range(k):
                    # Draw random selection of all trials
                    selection = np.random.choice(tr_arr, size=n_tr)
                    # Standard Deviation across trials
                    sd = np.array([variance_metric(data[selection, ch_idx, i]) for i in np.arange(pnt_range[0], pnt_range[1])])
                    # sd = np.std(data[selection, ch_idx, pnt_range[0]:pnt_range[1]], axis=0)
                    if not time_resolved:
                        sd = np.mean(sd)
                    sd_resample.append(sd)

                if time_resolved: 
                    # Calculate mean across iterations and retain time axis
                    sd = np.mean(sd_resample, axis=0)
                    x = np.arange(len(sd))
                    fits = tuple(np.polyfit(x, sd, 1))
                else: 
                    # Mean across everything
                    sd = np.mean(sd_resample)
                

        pack = dict(data=sd, trials=n_tr, info=info, fits=fits)
        self.scd_itv.append(pack)

        return pack

    def get_itv(self, condition, time_range, sign=False, std='trials', 
                            ch_name='O1', n_tr=None, k=1000, 
                            reduce_iters=True, resample_freq=100):
        '''Calculates the inter-trial variance of increasing number 
        of trials for a given (set of) electrode(s)

        Parameters
        ----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        time_range : time region of interest to calculate intertrial_variability 
            from
        sign : [True/False] whether to use the signflip method to reduce 
            variability to noise
        std : ['trials'/'time'] calculate standard deviation across trials OR 
            first calculate ERP to calculate Standard Deviation over time
        time_resolved : [True/False] Whether to return the Standard Deviation 
            for each time point individually (True). If False, the Mean Standard 
            Deviation is calculated and returned per number of trials.
        ch_name : [str/None] If None, all channels will be selected. If ch_name 
            is a string of a channel name (e.g. 'Cz') only this channel will be 
            used to calculate the intertrial variability
        n_tr : Specify number of trials if required.
        k : Number of repetitions for each number of trials computed
        reduce_iters : Reduces Computation time by using a higher number of 
            iterations 'k' for low number of trials and fewer iterations for high 
            number of trials. The reduced iters are calculated as an hyperbole with 
            damping factor of 50
        '''    
        
        # Create the sub-object using the condition in order to obtain the data requested.
        obj = self.__getitem__(condition)
        # Resample data to 100hz due to long computation times:
        if resample_freq is not None:
            obj.resample(resample_freq)
        data = obj.get_data()

        # Exception Handling
        if ch_name == None:
            ch_idx = None
        else:
            ch_idx = self.ch_names.index(ch_name)
        if type(n_tr) != int or type(n_tr) != float:
            n_tr = data.shape[0]
        if ch_idx == None:
            n_elec = np.arange(0, data.shape[1])
        else:
            n_elec = [ch_idx]

        # Create Info String (contains all necessary itv properties)
        info = dict(condition=condition, time_range=time_range, sign=sign, std=std, ch_name=ch_name, iterations=k, reduce_iters=reduce_iters, n_tr=n_tr, resample_freq=resample_freq)

        # Definitions
        trial_idc = np.arange(0, n_tr)
        
        # old_self = deepcopy(self)
        
        
        time_pnt = [nearest(obj.times, time_range[0]), nearest(obj.times, time_range[1])]
        # [np.argwhere(self.times==time_range[0])[0][0], np.argwhere(self.times==time_range[1])[0][0]] #changing ms to index

        num_trials = np.arange(4, n_tr, 2)
        # Reduce number of iterations based on the number of trials. 
        # Rationale: The more trials, the less iterations are required.
        if reduce_iters:
            damp = 50
            k = (1/(num_trials + damp)) * (damp + 4) * k
            k = [int(round(i)) for i in k]
        else: 
            k = [k] * np.max(num_trials)

        variance_decrement_time = np.zeros(shape=(len(num_trials)), dtype=np.float)

        fits = np.zeros( (3,)) # number of electrodes X number of fit parameters

        
        variance_decrement_time, variance_decrement_time_sd = np.squeeze(self.iter_signmethod(data, k, num_trials, ch_idx, trial_idc, time_pnt, std, variance_decrement_time, sign))

        # Fit Curve
 
        try:
            fits[:], pcov = curve_fit(hyperbole, num_trials, variance_decrement_time[:])
        except:
            print('Fits dont make sense, continuing anyway.') 
        gofs = gof(hyperbole, num_trials, variance_decrement_time[:], fits[ :])
        fits = np.squeeze(fits)
        gofs = np.squeeze(gofs)

        # self = old_self
        pack = dict(itv=variance_decrement_time, itv_sd=variance_decrement_time_sd, trials=num_trials, fits=fits, gofs=gofs, info=info)
        self.itv.append(pack)
        return pack

    @staticmethod
    # @jit(nopython=True, fastmath=True)
    def iter_signmethod(data, k, num_trials, ch_idx, trial_idc, time_pnt, std, variance_decrement_time, sign):
        ''' Just an outsourced function for get_variance_decrement_effect_signmethod. Reson for outsourcing and weird code is fast numba compilation (the @jit thing above the function).'''
        # Loop different amounts of trials
        variance_decrement_time_sd = np.zeros_like(variance_decrement_time)
        if len(variance_decrement_time.shape) == 1:
            for i in range(len(num_trials)):
                # here we will store the k permutations of selecting trials and calculating the measure of interest
                selection = np.zeros(shape=(k[i]))
                # k-fold Cross validation:
                for j in range(k[i]):
                    
                    # Select some random trials
                    choice = np.random.choice(trial_idc, num_trials[i], replace=False)
                    # And the actual data based on these trials
                    tmp_data = data[choice, ch_idx, time_pnt[0]:time_pnt[1]]
                    if sign:
                        # Flip sign of every second trial:
                        for ii in range(tmp_data.shape[0]):
                            if np.mod(ii, 2) == 0:
                                tmp_data[ii,] = np.multiply(tmp_data[ii,], -1)  # pointwise multiplication
                    if std=='time':
                        # Calculate standard deviation over the averaged trails (ERP)
                        noise_avg = np.sum(tmp_data, axis=0) / tmp_data.shape[0]  # weird implementation for numba
                        selection[j] = np.sqrt(np.mean(np.square(noise_avg)))  # root mean square (RMS)
                    elif std=='trials':
                        # Calculate standard deviation over trials at each time point and average those
                        std_over_trials = np.zeros((tmp_data.shape[1],))
                        for n in range(tmp_data.shape[1]):
                            std_over_trials[n] = np.std(tmp_data[:, n])

                        selection[j] = np.mean(std_over_trials)

                    variance_decrement_time[i] = np.mean(selection)
                    variance_decrement_time_sd[i] = np.std(selection)

        return variance_decrement_time, variance_decrement_time_sd
    
    def get_sequential_variability(self, condition, time_range, ch_name='O1', winsize=10, plotme=False, 
            time_resolved=False, normalize=None, variance_metric=np.std):
        ''' Variability of EEG Epochs over the course of one block.
        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        time_range: list, [lower time limit, upper time limit], e.g. [0.04, 0.2] in seconds.
        ch_name : str, name of channel
        winsize : int, size of sliding window to calculate standard deviation from
        plotme : bool, plot the result
        variance_metric : function, e.g. np.std, scipy.stats.median_absolute_deviation, dfa
        normalize : None/ func, can be either None or some function f(x) that returns a normalized vector x_norm
        '''

        scale = 1e6  # to scale up from volts to microvolts
        # Time in seconds to index
        pnt_range = range(np.where(np.isclose(self.times, time_range[0]))[0][0], np.where(np.isclose(self.times, time_range[1]))[0][0])
        # Channel Index
        ch_idx = self.ch_names.index(ch_name)
        # Retrieve condition
        obj = self.__getitem__(condition)
        data = obj.get_data()[:, ch_idx, pnt_range]
        n_tr = data.shape[0]
        win_var = []
        # loop through all sliding windows
        for i in range(n_tr-winsize):
            rng = range(i, i+winsize)
            win_var_tmp = np.array([variance_metric(data[rng, i]) * scale for i in range(data.shape[1])])
            # win_var_tmp = np.std(data[rng, :], axis=0) * scale
            if time_resolved:
                win_var.append(win_var_tmp)
            else:
                win_var.append(np.mean(win_var_tmp))
        win_var = np.asarray(win_var)
        # Normalize
        if normalize is not None:
            if time_resolved:
                win_var = normalize(win_var)
            else:
                win_var = normalize(win_var)  # normalize
        # Fit first degree polynomial to data and retain the slope and offset
        try:
            slope, offset = np.polyfit(np.arange(len(win_var)), win_var, 1)
        except:
            slope = 0
            offset = 0

        if plotme:
            if time_resolved:
                
                plt.figure()
                plt.imshow(win_var, extent=[time_range[0], time_range[1], 0, n_tr-winsize], aspect='auto')
                cb = plt.colorbar()
                cb.set_label('Standard Deviation [\u03BCV]')
                plt.xlabel('Time [s]')
                plt.ylabel('Trial No.')
                plt.title('Sequential Variability')
            else:
                plt.figure()
                plt.plot(win_var)
                plt.xlabel('Ordered number of trials N')
                plt.ylabel(f'Standard Deviation over 10 trials [mirovolt]')
                plt.title(f'Sequential Variability at {time_range[0]:.2f} - {time_range[1]:.2f} s ({ch_name})')

        # Info on settings
        info = dict(condition=condition, time_range=time_range, time_resolved=time_resolved, ch_name=ch_name, winsize=winsize, n_tr=n_tr, variance_metric=variance_metric.__name__)
        pack = dict(data=win_var, slope=slope, offset=offset, trials=n_tr, info=info)
        self.sequential_variability.append(pack)
        
        return pack

    def get_chaos(self, condition, ch_name='O1', resample_freq=100, time_range=None):
        ''' Variability of EEG Epochs over the course of one block.
        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        ch_name : str, name of channel
        '''
        scale = 1e6  # to scale up from volts to microvolts
        # Channel Index
        ch_idx = self.ch_names.index(ch_name)
        # Retrieve condition
        obj = self.__getitem__(condition)
        # Resampling
        if resample_freq is not None:
            print('Resampling')
            obj.resample(resample_freq)
        if time_range is None:
            data = np.array(obj.get_data()[:, ch_idx, :] * scale)
        else:
            pnt_range = [nearest(obj.times, time_range[0]), nearest(obj.times, time_range[1])]
            data = np.array(obj.get_data()[:, ch_idx, pnt_range[0]:pnt_range[1]] * scale)

        data = data.flatten()

        info = {
            'condition': condition,
            'ch_name': ch_name,
            'resample_freq': resample_freq
        }
        chaos = {}
        ## Calculate Detrended Fluctuation Analysis
        print('DFA')
        chaos['dfa'] = dfa(data)
        ## Calculate Sample Entropy
        print('Multiscale Sample Entropy')
        sample_length = 2
        maxscale = 100
        chaos['mse'] = multiscale_entropy(data, sample_length=sample_length, maxscale=maxscale)
        # Sample Entropy on cumulative sum of the data
        # chaos['mse_walk'] = multiscale_entropy(np.cumsum(data), sample_length=sample_length, maxscale=maxscale)

        pack = dict(data=chaos, info=info)
        self.chaos.append(pack)

        return pack

    def plot_itv(self, instance):
        
        # Catch Error if instance is not in self.itv:
        if instance not in np.arange(0, len(self.itv)):
            print(f'instance {instance} not in data. Please choose an itv below:')
            self.status_itv()
            return
        
        itv = self.itv[instance]
        a, b, c = itv['fits']
        gof = itv['gofs']
        plt.figure()
        plt.fill_between(itv['trials'], itv['itv']*1e6 + itv['itv_sd']*1e6, itv['itv']*1e6 - itv['itv_sd']*1e6, alpha=0.3, color='red')
        plt.plot(itv['trials'], itv['itv']*1e6, label='Intertrial Variability per No. trials', color='red')
        plt.plot(itv['trials'], hyperbole(itv['trials'], a, b, c)*1e6, label='Fitted Hyperbole')
        plt.title(f'fits: a={a*1e6:.3f}, b={b:.3f}, c={c*1e6:.3f}, gof: {gof:.2f}')
        plt.xlabel('Number of trials')
        plt.ylabel('SD across trials [\u03BCV]')
        plt.legend(loc='lower right')

class continuous_var(Raw, Base):
    def __init__(self, *args, **kwargs):
        ''' Call Parent __init__ and some extra things'''
        super(epochs_var, self).__init__(*args, **kwargs)

class meta_var(BaseMeta):
    ''' Meta Class of many epochs_var instances. Work in progress, can be used already!
    '''
    def __init__(self, all_epochs):
        assert type(all_epochs) == dict, 'all_epochs needs to be of type dict'
        self.all_epochs = all_epochs
        self._consistency_check()
        print('initialize meta class')
 
    def get_scd_itv(self, condition, time_range, time_resolved=False, 
                                ch_name='O1', n_tr=9999, k=100, 
                                variance_metric=np.std, parallel=False):
        ''' Compare Intertrial Variability based on RMS at smallest common denominator
        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        time_range : list, [low, high] in seconds
        time_resolved : bool, False: averaged over time_range; True: variability for 
            each time point
        ch_name : str, channel name or 'all' to calculate for all channels
        n_tr : int/None, if None the smalles common denominator is chosen, 
            otherwise the given integer is chosen
        k : int, number of resampling
        '''

        n_tr = self.common_trials(condition, n_tr)
        retr = self.call_fun('get_scd_itv', condition, time_range, time_resolved=time_resolved,
                                ch_name=ch_name, n_tr=n_tr, k=k, variance_metric=variance_metric)

        return retr

    def common_trials(self, condition, n_tr):
        ''' Calculate the highest number of common trials over all participants.
        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest 
        n_tr : int, number of desired trials. If too large (e.g. type 9999) 
            the number of available trials is chosen.
        
        Return:
        -------
        n_tr : int, number of common trials
        ''' 
        assert type(condition) == str, 'condition must be a string, e.g. <SF>'
        assert type(n_tr) == int, 'n_tr must be the number of trials of type int'

        keys = [key for key in self.all_epochs.keys()]
        # Loop Groupds
        for i, group in enumerate(keys):
            subkeys = [key for key in self.all_epochs[group].keys()]
            # Loop Participants in Groups
            for j, sub in enumerate(subkeys):
                obj = self.all_epochs[group][sub].__getitem__(condition)
                data = obj.get_data()
                n_tr = np.min([n_tr, data.shape[0]])
        return n_tr
    
    def get_itv(self, condition,  time_range, sign=False, std='trials', 
                            ch_name='O1', n_tr=None, k=1000, 
                            reduce_iters=True, resample_freq=100):
        '''Calculates the inter-trial variance of increasing number 
        of trials for a given (set of) electrode(s)

        Parameters
        ----------
        time_range : time region of interest to calculate intertrial_variability 
            from
        sign : [True/False] whether to use the signflip method to reduce 
            variability to noise
        std : ['trials'/'time'] calculate standard deviation across trials OR 
            first calculate ERP to calculate Standard Deviation over time
        time_resolved : [True/False] Whether to return the Standard Deviation 
            for each time point individually (True). If False, the Mean Standard 
            Deviation is calculated and returned per number of trials.
        ch_name : [str/None] If None, all channels will be selected. If ch_name 
            is a string of a channel name (e.g. 'Cz') only this channel will be 
            used to calculate the intertrial variability
        n_tr : Specify number of trials if required.
        k : Number of repetitions for each number of trials computed
        reduce_iters : Reduces Computation time by using a higher number of 
            iterations 'k' for low number of trials and fewer iterations for high 
            number of trials. The reduced iterations are calculated as an hyperbole with 
            damping factor 'damp' of 50: f(x) = (1/x) + damp
        condition : str ['SF'/'SR'/'LF'/'LR'/'GPL'/'GPS'] 
        '''
        retr = self.call_fun('get_itv', condition, time_range, sign=sign, std=std, 
                            ch_name=ch_name, n_tr=n_tr, k=k, 
                            reduce_iters=reduce_iters)


        return retr

    def get_sequential_variability(self, condition, time_range, ch_name='O1', winsize=10, time_resolved=False, 
                                variance_metric=np.std, normalize=None):
        ''' Variability of EEG Epochs over the course of one block.
        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        time_range: list, [lower time limit, upper time limit], e.g. [0.04, 0.2] in seconds.
        ch_name : str, name of channel
        winsize : int, size of sliding window to calculate standard deviation from
        plotme : bool, plot the result
        '''
        retr = self.call_fun('get_sequential_variability', condition, time_range, ch_name=ch_name, 
            winsize=winsize, plotme=False, time_resolved=time_resolved, variance_metric=variance_metric,
            normalize=normalize)
        return retr

    def get_chaos(self, condition, ch_name='O1', time_range=None):
        ''' Variability of EEG Epochs over the course of one block.
        Parameters:
        -----------
        condition : str, 'SF'/ 'SR'/ 'LF'/ 'LR', name of condition of interest
        ch_name : str, name of channel
        '''
        # raise NameError('Not tested yet!')

        retr = self.call_fun('get_chaos', condition, ch_name=ch_name, time_range=time_range)



        return retr

    @staticmethod
    def smallest_denominator_analysis(traces, label, groupcolor, title, fixed_number_of_trials=None):
        ''' Compare Intertrial Variability based on RMS at smallest common denominator
        Parameters:
        -----------
        title : str, title of the plot
        fixed_number_of_trials : [int, None], if it is not None Type but an integer it is 
            the number of trials that should be selected for the analysis
        
        Return:
        -------
        min_num_trials : int, smalles number of trials present
        sda_arr : -
        '''
             
        x = np.arange(4, 200, 2)
        if fixed_number_of_trials is None:
            min_tr = 9999  # initialize minimum common index of number of trials as being large
            for i, group in enumerate(traces):
                for sub in group:
                    min_tr = np.min([min_tr, sub.shape[0]])
        else:
            # If uneven : round down the number to be even
            fixed_number_of_trials -= np.mod(fixed_number_of_trials, 2)
            # Get index of requested number of trials
            min_tr = np.argwhere(x==fixed_number_of_trials)[0][0]
        
        title += f' {x[min_tr]} trials'  

        # Now that we have figured min_tr, get the values at that index for each person
        sda_arr = [np.zeros(shape=(len(traces[0]))), np.zeros(shape=(len(traces[1])))]
        for i, group in enumerate(traces):
            for j, sub in enumerate(group):
                sda_arr[i][j] = sub[min_tr-1] *1e6
        
        scatter_plot(sda_arr[0], sda_arr[1], title=title, stat='wilc', paired=False, legend=label)
        
        min_num_trials = x[min_tr]
        return min_num_trials, sda_arr
 

    def grand_mean(self, condition):
        ''' Calculate the grand mean of all data sets per group.
        Parameters:
        ----------
        condition : str, name of condition (e.g. <SF> for small frequent)

        Return:
        -------
        grand_means : list, list of evoked data structures containing the grand mean per group
        '''
        grand_means = []
        for key in self.all_epochs.keys():
            subs = [key for key in self.all_epochs[key].keys()]
            list_of_epochs = [self.all_epochs[key][sub][condition].average() for sub in subs]
            grand_means.append(mne.grand_average(list_of_epochs))
        return grand_means
    
    def plot_itv_summary(self, index, errormeasure='SEM', title=None, plot_hyp=False, 
                        sda=False, fixed_number_of_trials=None, auc=False):
        ''' This function plots the Standard Deviation of noise over different numbers of trials with individual and mean+sd/sem traces.
        Not that all values are multiplied by 1e6 to convert them from Volts to Microvolts.
        
        Parameters:
        -----------
        index : int, index in the attribute list to choose from. Use 
            meta.check_common_itv(attribute) to find desired index
        errormeasure : str, can be 'SEM' or empty, then standard deviation will 
            be calculated
        title : str, title for the plot
        plot_hyp : Plot hyperbole in top plot 
        sda : [True/False] Smallest common denominator analysis: Select the least 
            amount of trials that all participants have and do the variability 
            analysis on it.
        fixed_number_of_trials : int, an integer with a fixed number of trials 
            to choose for each participant. if None, the highest available
            number of trials will be picked.
        auc : [True/False] uses the bayesian optimiser to find svm parameters 
            resulting in highest aucs. aucs are of fit combinations. prints out 
            fit combos and their auc. plots the auc of the highest combination.
        
        Return:
        -------
        No return, only plots.
        '''
        # Printing Info or some title
        if title is None:
            key = [key for key in self.all_epochs.keys()][0]
            sub = [key for key in self.all_epochs[key].keys()][0]
            info = self.all_epochs[key][sub].itv[index]['info']
            if 'n_tr' in info:
                info.pop('n_tr')
            print('------------')
            print(f'\nITV Settings: \n')
            print_dict(info)
        else:
            print(f'\n{title}\n')

        groupcolor = ['black', 'red']
        label = [key for key in self.all_epochs.keys()]
        # Extract ITV Traces and Fit Parameters from Object
        traces = [[list() for i in range(len(self.all_epochs[label[0]]))], 
                        [list() for i in range(len(self.all_epochs[label[1]]))]]
        fits = [[list() for i in range(len(self.all_epochs[label[0]]))], 
                        [list() for i in range(len(self.all_epochs[label[1]]))]]

        for i, group in enumerate(label):
            for j, sub in enumerate(self.all_epochs[group].keys()):
                traces[i][j] = self.all_epochs[group][sub].itv[index]['itv']
                fits[i][j] = self.all_epochs[group][sub].itv[index]['fits']

        fig= plt.figure(num=title, figsize=(10, 7))
        gridspec.GridSpec(2, 3)
        # Large Plot on Top
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3, rowspan=1)
        ## Single Trials
        for i, group in enumerate(traces):
            for sub in group:
                trials = np.arange(4, len(sub[:])*2 +4, 2)
                plt.plot(trials, sub[:]*1e6, color=groupcolor[i], alpha=0.1, linewidth=1)
        # Mean +- SD/SEM
        cont_mean, cont_std = tolerant_mean([i[:] for i in traces[0]], method='lcd')
        asd_mean, asd_std = tolerant_mean([i[:] for i in traces[1]], method='lcd')
        cont_mean, cont_std, asd_mean, asd_std = cont_mean*1e6, cont_std*1e6, asd_mean*1e6, asd_std*1e6
        if errormeasure=='SEM':
            cont_std = np.array(cont_std)
            asd_std = np.array(asd_std)
            cont_std /= np.sqrt(len(traces[0]))
            asd_std /= np.sqrt(len(traces[1]))

        trials = np.arange(4, len(cont_mean)*2 +4, 2)
        ax1.plot(trials, cont_mean, color=groupcolor[0], label=label[0], linewidth=3)
        ax1.fill_between(trials, cont_mean-cont_std, cont_mean+cont_std, color=groupcolor[0], alpha=0.4)
        trials = np.arange(4, len(asd_mean)*2 +4, 2)
        ax1.plot(trials, asd_mean, color=groupcolor[1], label=label[1], linewidth=3)
        ax1.fill_between(trials, asd_mean-asd_std, asd_mean+asd_std, color=groupcolor[1], alpha=0.4)
        ax1.legend()
        ax1.set_title(title)
        ax1.set_xlabel('n trials', fontsize=16)
        ax1.set_ylabel('RMS [\u03BCV]', fontsize=16)
        ## Plot also wilcoxon p
        ### Minimum number of trials available
        mintr = np.min(np.concatenate([[len(i) for i in traces[0]], [len(i) for i in traces[1]]]))
        p = np.zeros((mintr,))
        for tr in range(mintr):
            # p[tr] = permutation_test([i[tr] for i in traces[0]], [i[tr] for i in traces[1]], n_perm=int(1e6))
            p[tr] = ranksums([i[tr] for i in traces[0]], [i[tr] for i in traces[1]])[1]
        ax11 = ax1.twinx()
        color = 'tab:blue'
        ax11.plot(trials[:mintr], p, color=color)
        # ax11.plot([np.min(trials), np.max(trials)], [0.05, 0.05], '--', color=color)
        ax11.set_ylabel('Wilcoxon p-value', color=color)  # we already handled the x-label with ax
        ax11.set_yscale("log")

        # Hyperbole inset
        if plot_hyp:
            ax2 = inset_axes(ax1, 
                            width="30%", # width = 30% of parent_bbox
                            height=0.8, # height : 1 inch
                            loc=7)    
            plot_hyperbole(ax2)
        # Boxplots
        ## Put fit parameters of correct electrode into 'fits separate'
        fits_separate = [ [[], []], [[], []], [[], []]]  # list of three entries: 0=A, 1=B and 2=C, each of which has two lists: 0=Controls and 1=ASDs
        fit_labels = ['A', 'B', 'C']
        bplots = [list() for i in range(3)]
        
        for fit in range(len(fits_separate)):
            print(f'\nfit {fit_labels[fit]}:\n')
            for group in range(2):
                if fit != 1:
                    fits_separate[fit][group] = [ i[fit]*1e6 for i in fits[group] ]
                else:
                    fits_separate[fit][group] = [ i[fit] for i in fits[group] ]

            plt.subplot2grid((2, 3), (1, fit), colspan=1, rowspan=1)
            omni = omni_stat(fits_separate[fit][0], fits_separate[fit][1], paired=False, n_perm=int(1e6), verbose=1, effort=0)


            bplots[fit] = plt.boxplot(fits_separate[fit], patch_artist=True, labels=label, bootstrap=1000, whis=[5, 95])
            plt.title(f'{fit_labels[fit]}')
            
        # Change global Font size
        plt.rcParams['font.size'] = 10
        # Change Boxplot Fill Colors
        for bplot in bplots:
            for patch, color in zip(bplot['boxes'], groupcolor):
                patch.set_facecolor(color)
        


        fig.tight_layout(pad=2.0)

        if sda:
            sda_title = 'Smallest common denominator analysis:'
            print(f'\n{sda_title}\n')
            min_tr, sda = self.smallest_denominator_analysis(traces, label, groupcolor, sda_title, fixed_number_of_trials=fixed_number_of_trials)
            if auc:
                print('\n')
                #create X and y arrays for classification with function from util
                Xsda, y = make_Xy(fits_separate,sda)
                #use bayesian optimiser function from until. best auc with corresponding c and gamma is saved to a dictionary
                bayes_dict = bayesian_opt(Xsda,y) 
                
                print(f'AUC for smallest common demoninator: {bayes_dict["target"]:.4}')
                #find parameters from dictionary for ROC plot
                C = bayes_dict['params']['C']
                gamma =  bayes_dict['params']['gamma']
                title = (f'Smallest common denominator (AUC: {bayes_dict["target"]:.4})')
                
                #plot the ROC
                plot_roc(Xsda, y, C, gamma, title)                
            
            print(f'\nNumber of trials: {min_tr}')
                
        if auc:
            #create X and y arrays for classification with function from util
            X, y = make_Xy(fits_separate)

            #dict to fill with column names and best auc and params for each combo
            dict_combos={}

            #loop through all potential rs for function below. r = length. [1] = columns
            for r in range(2,X.shape[1]+1): 
                #get all combos as list of tuples for index
                #combinations function requires list input and r
                perm = combinations(list(range(X.shape[1])),r) #output is tuple of various combinations

                #loop through all values in list
                for i in list(perm):
                    #make i list to allow for indexing
                    i = list(i) 
                    # indexing X (df.vals). all rows, specific columns
                    result = X[:,i] 
        
                    #create key for dictionary using fit_labels list
                    key = [fit_labels[index] for index in i]
                    #needs to be string in order to be dict key
                    key = str(key)
                    #clean up string for dict key
                    key = key.strip("[]")
                    key = key.replace("'","")
                    print(key)

                    #use bayesian optimizer from function in util. save best auc and its params for dict vals
                    value = bayesian_opt(result,y) 

                    #add to empty dictionary
                    dict_combos[key] = value 
             
            
            #print out parameters and aucs
            print(f'AUCs for fit method:')
            for key, val in dict_combos.items():
                print(f'parameter combination: {key} \nauc: {val["target"]:.4}\n')
            
            
            #plot the roc of the parameter combination that results in the highest auc
            #loops through dictionary getting key and val to access target and puts parameters and auc in list
            aucs = [[key,val['target']] for key,val in dict_combos.items()] 

            #find max auc from list above
            #key is an additional argument of max function, indexing to access second element of each nested loop
            maxauc = max(aucs, key=lambda x: x[1])

            #get C and gamma for highest auc from dictionary
            #maxauc[0] indexes the first element of the aucs list which is the key of dict aka parameter combinations
            C = dict_combos[maxauc[0]]['params']['C']
            gamma = dict_combos[maxauc[0]]['params']['gamma']
            title= (f'ROC Curve for Parameters: {maxauc[0]} (AUC: {dict_combos[maxauc[0]]["target"]: .4})')
            
            #get indeces for data array (X) from index of fit_labels
            indexlist=[]
            #loop through fit_labels 
            for fitindex, fit in enumerate(fit_labels):
                #find fit in list from dictionary key of max auc list
                if fit in maxauc[0]:
                    #append the index of that fit to list that is used as index for X
                    indexlist.append(fitindex)
            
            #plot roc using plot_roc function
            plot_roc(X[:,indexlist], y, C, gamma, title)
            
    def plot_scd_itv(self, index, title='', convertSI=True, colors=None, legend=None):
        ''' Plot the smallest common denominator itv.
        Parameters:
        -----------
        index : int, index in the attribute list to choose from. Use 
            meta.check_common_itv(attribute) to find desired index
        title : str, title for the plot
        Return:
        -------
        No return, only plots.
        '''
        
        key = 'data'
        if legend is None:
            legend = [key for key in self.all_epochs.keys()]
        # Retrieve Data

        list_of_attributes = self.retrieve_attr('scd_itv')

        extracted_data = [list(), list()]
        for group in range(len(list_of_attributes)):
            for sub in range(len(list_of_attributes[group])):
                if len(list_of_attributes[group][sub]) >= index:
                    extracted_data[group].append(list_of_attributes[group][sub][index][key])
                else:
                    msg = f'index {index} is not available ({len(list_of_attributes[group][sub])} indices in data)'
                    raise IndexError(msg)

        # Get the info dictionary of the very last subject (doesnt matter which since they should be same anyway)
        info = list_of_attributes[group][sub][index]['info']
        time_range = info['time_range']
        n_tr = list_of_attributes[group][sub][index]['trials']
        group = [key for key in self.all_epochs.keys()][0]
        sub = [key for key in self.all_epochs[group].keys()][0]
        times = self.all_epochs[group][sub].times
        extracted_data = [np.stack(extracted_data[0], axis=0), np.stack(extracted_data[1], axis=0)]
        if convertSI:
            extracted_data = [x * 1e6 for x in extracted_data]
            ylabel = 'Standard Deviation [\u03BCV]'
        else:
            ylabel = 'Standard Deviation [V]'
        time_resolved = len(extracted_data[0].shape) > 1
        if title == '':
            title = f'{n_tr} trials'

        if time_resolved:
            # time = np.arange(len(extracted_data[0][0]))
            time = times[nearest(times, time_range[0]):nearest(times, time_range[1])]
            plot_two_with_error(time, extracted_data[0], extracted_data[1], measure='SEM', \
                labels=legend, title=title, ylabel=ylabel, xlabel='Time [s]', \
                xticklabels=None, test='perm', colors=colors)
        else:
            scatter_plot(extracted_data[0], extracted_data[1], title=title, stat='wilc', paired=False, legend=legend, colors=colors)

    def plot_sequential_variability(self, index, plottype='all', title='', colors=None):
        ''' Plot sequential variability.
        Parameters:
        -----------
        index : int, index in the attribute list to choose from. Use 
            meta.check_common_itv(attribute) to find desired index
        plottype : str, can be 'single_subject', 'mean' or 'all' to plot both.
        title : str, title for the plot

        Return:
        -------
        No return, only plots.
        '''
        key = 'data'
        list_of_attributes = self.retrieve_attr('sequential_variability')
        legend = [key for key in self.all_epochs.keys()]
        # if colors is None:
        #     colors = ['darkblue', 'darkorange']
        
        extracted_data = [list(), list()]
        for group in range(len(list_of_attributes)):
            for sub in range(len(list_of_attributes[group])):
                assert len(list_of_attributes[group][sub]) >= index and list_of_attributes[group][sub] != [], 'Index out of range.'
                extracted_data[group].append(list_of_attributes[group][sub][index][key])
        else:
            info = list_of_attributes[group][sub][index]['info']
        print('\ninfo:\n')
        print_dict(info)        

        time_resolved = len(extracted_data[0][0].shape) > 1

        # Plot single subject ordered by group
        if plottype == 'all' or plottype == 'singlesubject':
            self.plot_subjects_columnwise(extracted_data, legend=legend, title='Sequential Variability ' + title)

        # Plot grand means per group
        if plottype == 'all' or plottype == 'mean':
            # if time_resolved == False
            if not time_resolved:
                lens = [[len(i) for i in extracted_data[0]], [len(i) for i in extracted_data[1]]]
                minlen = np.min([np.min(lens[0]), np.min(lens[1])])
                extracted_data_cut = [[i[0:minlen] for i in extracted_data[0]], [i[0:minlen] for i in extracted_data[1]]]

                means = [np.mean(extracted_data_cut[0], axis=0), np.mean(extracted_data_cut[1], axis=0)]
                sems = [sem(extracted_data_cut[0], axis=0), sem(extracted_data_cut[1], axis=0)]
                xlabel = 'Trial number (ordered)'
                ylabel = 'Standard Deviation [\u03BCV]'

                plot_two_with_error([], extracted_data_cut[0], extracted_data_cut[1], title=title, xlabel=xlabel, ylabel=ylabel, \
                    labels=legend, colors=colors)
            # if time_resolved == True
            else:
                lens = [[i.shape[0] for i in extracted_data[0]], [i.shape[0] for i in extracted_data[1]]]
                minlen = np.min([np.min(lens[0]), np.min(lens[1])])
                extracted_data_cut = [np.array([i[0:minlen, :] for i in extracted_data[0]]), np.array([i[0:minlen, :] for i in extracted_data[1]])]


                img_shape = extracted_data_cut[0].shape[1:]                
                controls_flat = extracted_data_cut[0].reshape(extracted_data_cut[0].shape[0], np.prod(extracted_data_cut[0].shape[1:]))
                asd_flat = extracted_data_cut[1].reshape(extracted_data_cut[1].shape[0], np.prod(extracted_data_cut[1].shape[1:]))
                
                means = [np.mean(extracted_data_cut[0], axis=0), np.mean(extracted_data_cut[1], axis=0)]
                
                sems_time = [ \
                    sem(np.mean(extracted_data_cut[0], axis=1), axis=0), \
                    sem(np.mean(extracted_data_cut[1], axis=1), axis=0)
                ]
                sems_trials = [ \
                    sem(np.mean(extracted_data_cut[0], axis=2), axis=0), \
                    sem(np.mean(extracted_data_cut[1], axis=2), axis=0)
                ]

     
                diff = means[0] - means[1]

                p_vals = np.ones(img_shape)
                p_flat = np.array( [ttest_ind(controls_flat[:, i], asd_flat[:, i])[1] for i in range(controls_flat.shape[1])])
                p = p_flat.reshape(img_shape)
                p_mask = np.zeros(img_shape)
                p_mask[p<0.025] = 1
                p_mask_smoothed = smooth_image(p_mask)
                diff_thresh = deepcopy(diff)
                diff_thresh[p_mask_smoothed<0.5] = 0


                time_range = info['time_range']
                n_tr = means[0].shape[0]
                extent = [time_range[0], time_range[1], n_tr+1, 1]
                time = np.linspace(time_range[0], time_range[1], num=img_shape[1])
                trials = np.arange(1, img_shape[0]+1)
                vmin, vmax = [np.min(means), np.max(means)]
                plt.figure()
                plt.suptitle(title)

                plt.subplot(241)
                plt.imshow(means[0], extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
                cb = plt.colorbar()
                cb.set_label('Standard Deviation [\u03BCV]')
                plt.xlabel('Time [s]')
                plt.ylabel('Trial No.')
                plt.title(legend[0])
                
                plt.subplot(242)
                plt.imshow(means[1], extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
                plt.colorbar()
                plt.title(legend[1])

                plt.subplot(243)
                plt.imshow(diff, extent=extent, aspect='auto')
                plt.colorbar()
                plt.title(f'{legend[0]} - {legend[1]}')

                plt.subplot(244)
                plt.imshow(diff_thresh, extent=extent, aspect='auto')
                plt.colorbar()
                plt.title(f'diff_thresh')

                colors = ['darkblue', 'darkorange']

                plt.subplot(245)
                avg_time_controls = np.mean(np.mean(extracted_data_cut[0], axis=0), axis=0)
                avg_time_asd = np.mean(np.mean(extracted_data_cut[1], axis=0), axis=0)

                plt.fill_between(time, avg_time_controls-sems_time[0], avg_time_controls+sems_time[0], color=colors[0], alpha=0.3)
                plt.fill_between(time, avg_time_asd-sems_time[1], avg_time_asd+sems_time[1], color=colors[1], alpha=0.3)
                
                plt.plot(time, avg_time_controls, color=colors[0], label='controls')

                plt.plot(time, avg_time_asd, color=colors[1], label='asd')
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel('Standard Deviation [\u03BCV]')
                plt.title('Standard Deviation over time')

                plt.subplot(247)
                avg_trials_controls = np.mean(np.mean(extracted_data_cut[0], axis=0), axis=1)
                avg_trials_asd = np.mean(np.mean(extracted_data_cut[1], axis=0), axis=1)


                plt.fill_between(trials, avg_trials_controls-sems_trials[0], avg_trials_controls+sems_trials[0], color=colors[0], alpha=0.3)
                plt.fill_between(trials, avg_trials_asd-sems_trials[1], avg_trials_asd+sems_trials[1], color=colors[1], alpha=0.3)

                plt.plot(trials, avg_trials_controls, color=colors[0], label='controls')
                plt.plot(trials, avg_trials_asd, color=colors[1], label='asd')
                plt.legend()
                plt.xlabel('Trial number (ordered)')
                plt.ylabel('Standard Deviation [\u03BCV]')
                plt.title('Standard Deviation over trials')

                # plt.tight_layout(pad=2)

    def plot_sequential_variability_fits(self, index, title=''):
        ''' Plot the slope of the sequential_variability traces.
        Parameters:
        -----------
        index : int, position in sequential_variability list
        title : str, title of the plot
        
        Return:
        -------
        No return, only plots.
        '''
        slopes = [list(), list()]
        all_seq_itv = self.retrieve_attr('sequential_variability')
        legend = [key for key in self.all_epochs.keys()]

        for f in range(2):
            for sub in range(len(all_seq_itv[f])):
                slopes[f].append(all_seq_itv[f][sub][index]['slope'])
        scatter_plot(slopes[0], slopes[1], title='slopes: '+ title, legend=legend)

        offsets = [list(), list()]
        
        for f in range(2):
            for sub in range(len(all_seq_itv[f])):
                offsets[f].append(all_seq_itv[f][sub][index]['offset'])
        scatter_plot(offsets[0], offsets[1], title='offsets: '+ title, legend=legend)

    @staticmethod
    def plot_subjects_columnwise(data, extent=None, legend=('A', 'B'), title='', xlabel='', ylabel=''):
        ''' Plots two columns with individual traces, e.g. ERPs, sorted 
        by group.
        Parameters:
        -----------
        extent : list, extent for matplotlib.pyplot.imshow() method
        legend : list, list containing the legend strings
        title : str, title of the plot
        xlabel : str, x-axis label
        ylabel : str, y-axis label
        '''
    

        n_subs = [len(data[0]), len(data[1])]

        plt.figure()
        plt.suptitle(f'{title}; Left: {legend[0]}, Right: {legend[1]}', fontsize=20)
        cnt = 1
        for sub in range(np.max(n_subs)):
            for f in range(len(data)):
                if sub < n_subs[f]:
                    plt.subplot(np.max(n_subs), 2, cnt)
                    if len(data[f][sub].shape) == 1:
                        plt.plot(data[f][sub])
                        plt.tick_params(
                            axis='both',          # changes apply to the x- and y-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
                    else:
                        plt.imshow(data[f][sub], extent=extent, aspect='auto')
                        plt.axis('off')
                cnt += 1
        else:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axis('on')

                
        plt.tight_layout(pad=2)
    
    def plot_chaos(self, index, title=''):
        ''' Plot the calculated chaos parameters per group.
        Parameters:
        -----------
        index : int, index in the attribute list to choose from. Use 
            meta.check_common_itv(attribute) to find desired index
        title : str, title of the plot
        '''
        supkey = 'chaos'
        legend = [key for key in self.all_epochs.keys()]
        list_of_attributes = self.retrieve_attr('chaos')

        meta_dict = dict()
        
        for group in range(len(list_of_attributes)):
            for sub in range(len(list_of_attributes[group])):
                chaostypes = list_of_attributes[group][sub][index]['data'].keys()
                for chaostype in chaostypes:
                    if not chaostype in meta_dict:
                        meta_dict[chaostype] = [list(), list()]
                    meta_dict[chaostype][group].append(list_of_attributes[group][sub][index]['data'][chaostype])
        else:   
            cond = list_of_attributes[group][sub][index]['info']['condition']
            ch_name = list_of_attributes[group][sub][index]['info']['ch_name']
        
        for param, val in meta_dict.items():
            finTitle = f'{cond}, {param}, ch:{ch_name}, {title}'
            if 'mse' in param:
                time = np.arange(len(val[0][0]))
                plot_two_with_error(time, val[0], val[1], labels=legend, title=finTitle, xlabel='scales', ylabel='sample entropy')
            else:
                scatter_plot(val[0], val[1], title=finTitle, legend=legend)

class ElementsNotEqualError(Exception):
    """Raised when the input value is too small"""
    def __init__(self, message="Not all elements are equal"):
        self.message = message
        super().__init__(self.message)

