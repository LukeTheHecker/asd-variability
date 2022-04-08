from scipy.stats import sem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne

def plot_topomap(epoch_list, error='sem', condition=None, channel_label_offsets=[0.15, 0.6],
    sem_alpha=0.5, window_margins=None, title='', legend=None, colors='default',
    plot_vert=False, linewidth=1, figsize=(9, 7), font_scale=0.8, font='georgia',
    convert_units=True, ylim_tol=0.2, plot_gap=0.04, time_range=None,combine_conds=False):
    '''
    Parameters:
    -----------
    epochs_list : list, a list of mne.Epochs objects
    error : str, defines the type of error shadings. Can be either
        'std', 'sem' or None.
    condition : str, defines the condition to be extracted from epochs
    channel_label_offsets : list, [x_offset, y_offset] defines the offsets of the channel 
        label from the leftmost position and the top of the axis
    sem_alpha  : float, the transparency value (alpha) for error shadings
    window_margins : list, [left_gap, right_gap, bottom_gap, top_gap] the 
        minimum allowed gap between the plots and the window margins
    title : str, title of the plot
    legend : list, legend names for the plot
    colors : str/list, can be either 'default' for the default color palette or 
        a custom list of matplotlib-readable color strings
    plot_vert : bool, plot vertical axes
    linewidth : float, line width of the ERPs
    figsize : tuple, size of the whole topoplot figure
    font_scale : float, determines the font size
    font : str, default is 'georgia'
    ylim_tol : float, the higher the wider the y-axis limits
    plot_gap : float, defines the minimum gap (i.e. spacing) between channel plots
    combine_conds : bool, if True then condition is a list of >1 event IDs that shall be combined. 
        If False, then multiple conditions are plotted separately
    '''

    # Handle input list
    if type(epoch_list[0]) != list:
        # single list of epochs, i.e. single group
        epoch_list = [epoch_list]

    # Extract channel positions
    epoch = epoch_list[0][0]
    montage = epoch.get_montage()
    digs = montage.dig[3:]
    pos = np.stack([dig['r'][:2] for dig in digs], axis=0)
    times = epoch.times

    if time_range is None:
        pnt_range = np.arange(len(times))
    else:
        pnt_range = [np.argmin(np.abs(times-time_range[0])), np.argmin(np.abs(times-time_range[1]))]
        pnt_range = np.arange(*pnt_range)
    


    # Calculate ERP and measurement errors
    ERP, SEM = group_data(epoch_list, condition, error,combine_conds=combine_conds)
 
    if convert_units:
        ERP *= 1e6
        SEM *= 1e6

    # Extract time range of interest
    ERP = ERP[..., pnt_range]
    SEM = SEM[..., pnt_range]
    times = times[..., pnt_range]

    # Handle Data Shape
    if len(ERP.shape) < 3:
        # Single Group needs empty first dim:
        ERP = np.expand_dims(ERP, axis=0)
    if SEM is not None:
        if len(SEM.shape) < 3:
            # Single Group needs empty first dim:
            SEM = np.expand_dims(SEM, axis=0)

    #  Plot properties

    ## Sizes and gaps

    ### Handle sizes and distances
    if window_margins is None:
        window_margins = [0.1, 0.15, 0.1, 0.2]

    ### Define sizes and distances
    left_gap, right_gap, bottom_gap, top_gap = window_margins
    pos_binned = tidy_channel_positions(pos)
    width, height = get_plot_size(pos_binned, gap=plot_gap)
    ylim = [np.min(ERP)-abs(ylim_tol*np.min(ERP)) , np.max(ERP)+abs(ylim_tol*np.max(ERP))]

    ## Colors
    if colors == 'default':
        colors = sns.color_palette()

    ## Legend
    if type(legend) == str:
        legend = [legend]
    elif type(legend) == list:
        if len(legend) != ERP.shape[0]:
            print(f'Legend has {len(legend)} entries, although ERP indicates {ERP.shape[0]} groups')
            legend = None

    # Create distance between plots and the frame (size=(1,1))
    pos_binned[:, 0] = norm_to_range(pos_binned[:, 0], 0+left_gap, 1-right_gap)
    pos_binned[:, 1] = norm_to_range(pos_binned[:, 1], 0+bottom_gap, 1-top_gap)

    fig = plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale, style='ticks', context='notebook', font=font)
    plt.suptitle(title)
    plt.axis('off')

    if legend is not None:
        patches = list()
        for leg, color in zip(legend, colors):
            patches.append( mpatches.Patch(color=color, label=leg) )
            
        plt.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.1, 0))
    
    for ch, pos_, ch_name in zip(np.flip(np.arange(pos_binned.shape[0])), reversed(pos_binned), reversed(epoch.ch_names)):
    # for ch, (pos_, ch_name) in enumerate(zip(pos_binned, epoch.ch_names)):
        ax = plt.axes(np.append(pos_, [width, height]))
        plt.axis('off')
        # Horizontal Line
        plt.plot([times.min(), times.max()], [0, 0], color='black')
        # Vertical Line
        if plot_vert:
            plt.plot([0, 0], ylim, color='black')
        plt.ylim(ylim)
        ax.text(times.min()+channel_label_offsets[0]*(times.max()-times.min()),
            ylim[1]*channel_label_offsets[1], 
            ch_name,
            horizontalalignment='center')
        # Loop through groups
        for group in range(ERP.shape[0]):
            if SEM is not None:
                ax.fill_between(times, ERP[group, ch]-SEM[group, ch], ERP[group, ch]+SEM[group, ch], alpha=sem_alpha, color=colors[group])
            ax.plot(times, ERP[group, ch], color=colors[group], linewidth=linewidth)
            

        # Plot axes info on botto left plot:
        if pos_[1] == np.min(pos_binned[:, 1]) and pos_[0] == np.min(pos_binned[np.where(pos_binned[:, 1]==pos_[1])[0], 0]):
            plt.axis('on')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Amplitude [\u03BCV]')
            plt.xlabel('Time [s]')
    return fig

def discretize(arr, n_bins):
    if type(arr) == list:
        arr = np.array(arr)
    arr_bins = np.zeros((len(arr)))
    bins = np.arange(n_bins)
    boarders = np.linspace(arr.min(), arr.max(), n_bins)
    for bin in range(n_bins-1):
        indices_in_current_bin = (arr >= boarders[bin]) & (arr <= boarders[bin+1])
        arr_bins[indices_in_current_bin] = boarders[bin]
    return arr_bins

def tidy_channel_positions(pos, n_bins_vert=7):
    ''' Tidy up the channel position with a row-structure
    Parameters:
    -----------
    pos : numpy.ndarray, n_channelsx2 list of x and y coordinates of 
        channel positions
    n_bins_vert : int, number of rows to arrange electrodes to.
    Return:
    -------
    pos_binned : numpy.ndarray, the tidied up channel positions
    '''
    bins_vert = np.linspace(pos[:, 1].min(), pos[:, 1].max(), num=n_bins_vert)
    bins_y = discretize(pos[:, 1], n_bins_vert)

    ## Row binning
    bins_x = np.zeros((len(bins_y))) 
    for vertpos in np.unique(bins_y):
        # Get indices of electrodes in current row
        row_indices = np.where(bins_y==vertpos)[0]
        # Know how to sort and unsort them
        sort_ascend = np.argsort(pos[row_indices, 0])
        unsort_ascend = np.argsort(sort_ascend)
        # Find x positions that are equally spaced:
        new_row_pos = np.linspace(pos[row_indices, 0].min(), pos[row_indices, 0].max(), num=len(pos[row_indices, 0]))
        # Put the ascending values in the previous order
        new_row_pos = new_row_pos[unsort_ascend]
        bins_x[row_indices] = new_row_pos

    pos_binned = np.stack([bins_x, bins_y,], axis=1)
    # Normalize
    pos_binned[:, 0] -= pos_binned[:, 0].min()
    pos_binned[:, 0] /= pos_binned[:, 0].max()
    pos_binned[:, 1] -= pos_binned[:, 1].min()
    pos_binned[:, 1] /= pos_binned[:, 1].max()
    
    return pos_binned

def norm_to_range(arr, lo, hi):
    return lo + ((arr - arr.min()) * (hi - lo) / (arr.max() - arr.min()))

def get_plot_size(pos, gap=0.05):
    '''Get maximum window size for plots
    Parameters:
    -----------
    pos : numpy.ndarray, n_channelsx2 list of x and y coordinates of 
        channel positions
    gap : float, defines the minimal gap between axes
    Return:
    -------
    width : float, maximal allowed width of axes
    height : float, maximal allowed height of axes
    '''

    # Width
    ## Find the closest two plots can get horizontally
    row_bins = np.unique(pos[:, 1])
    n_plots = []
    min_dist = 9999
    for row_bin in row_bins:
        # for col in 
        n_plots.append( len(np.where(pos[:, 1]==row_bin)[0]) )
        min_row_dists = np.min(np.diff(np.sort(pos[np.where(pos[:, 1]==row_bin)[0], 0])))
        min_dist = np.min([min_row_dists, min_dist])

    width = min_dist - gap
    height = np.min(np.diff(np.sort(row_bins))) - gap
    return width, height

def group_data(all_data,cond,error,combine_conds=False):
    '''Get mean and error of group
    Parameters:
    ----------------------
    all_data: list, nested list of epochs or evoked objects for every participant in group
    cond: str, condition ID
    error: str, type of error 'sem' or 'std'
    Output:
    -----------------
    mean: numpy.ndarray, ERP for each group and/or condition. group and/or condition x elecs x datapoints
    error: numpy.ndarray, same shape as mean array.'''
    
    if isinstance(all_data[0][0], mne.epochs.Epochs):
        # call on functions to get epochs and epoch averages
        group = get_epoch_for_cond(all_data, cond, combine_conds=combine_conds)
        group_avg = [evoked_list(group) for group in group]
    
    if isinstance(all_data[0][0], mne.evoked.EvokedArray):
        #get average data from evoked objects
        data_array = []
        for group in all_data:
            group_list = []
            for participant in group:
                participant_data=participant.data
                group_list.append(participant_data)
            data_array.append(group_list)
        group_avg = data_array
    
    #get group averages and errors from averaged participant data
    group_avg_error = [group_mean_error(group_avg, error) for group_avg in group_avg]
    
    #reshape so all group data/condition data is in single 3d array
    mean = np.stack([group[0] for group in group_avg_error])
    
    if error is not None:
        error = np.stack([group[1] for group in group_avg_error])

    return mean, error



def evoked_list(group_epoch_list):
    '''get average data from epoch list.
    Parameters:
    -----------------
    group_epoch_list: list, epoch objects to average over
    Output:
    ------------------
    group: list, ERP array for each participant in group/condition. elec x samples points
    '''
    
    group = []
    #loop through input to access averaged data for each participant
    for i in range(len(group_epoch_list)):
        # if type(group_epoch_list[i]) == mne.EpochsArray
        evoked = group_epoch_list[i].average()
        avg_data = evoked.data
        group.append(avg_data)

    return group

def group_mean_error(listofarrays,error_type):
    '''Get ERP and error for group.
    Parameters:
    ---------------------------
    listofarrays: list, evoked arrays for single group/condition
    error_type: str, 'sem' or 'std'
    Output:
    --------------
    group_mean: numpy.ndarray, group ERP values. elec x sample points
    group_error: numpy.ndarray, group sem/std. elec x sample points
    '''
    #get group mean and group error
    group_array = np.stack(listofarrays,axis=0)
    group_mean = np.mean(group_array,axis=0)

    if error_type is None:
        return group_mean, None

    if error_type == 'sem':
        group_error = sem(group_array,axis=0)
    elif error_type == 'std':
        group_error = np.std(group_array,axis=0)

    return group_mean, group_error

def get_epoch_for_cond(all_data_epoch_list,cond, combine_conds=False):
    '''Get Epoch data for specific condition from Epoch list.
    Parameters:
    -----------------------
    all_data_epoch_list: list, nested list of epoch objects.
    cond: str, if single condition (i.e. 'SR')
          list, if multiple conditions are desired (i.e. ['SR','LR'])
    Output:
    -----------------------
    group_list: list, epoch data of single condition for single group
    output_data: tuple, list of epoch data for single condition for multiple groups
    cond_lists: tuple, list of epoch data for each desired condition. 
                If more than 1 group: 
                all conditions of group 1 listed first, then all conditions of group 2, etc.
    '''
    #single condition
    if type(cond) == str or combine_conds:
        #single group
        if len(all_data_epoch_list) == 1:
            group_list = [[]]
        
            for group in all_data_epoch_list:
                for part in group:
                    group_list[0].append(part[cond])
                
            return group_list
        #more than one group
        elif len(all_data_epoch_list) > 1:
            group_dict = {f'group_{group+1}': [] for group in range(len(all_data_epoch_list))}
        
        
            for groupidx,group in enumerate(all_data_epoch_list):
                for part in group:
                    group_dict[f'group_{groupidx+1}'].append(part[cond])
                
            output_data = [group_dict[f'group_{group+1}'] for group in range(len(all_data_epoch_list))]
        
            return tuple(output_data)
        
    #more than one condition
    elif type(cond) == list:
        cond_lists = []
        for groupidx, group in enumerate(all_data_epoch_list):
            for condition in cond:
                single_cond = []
                cond_lists.append(single_cond)
                for part in group:
                    single_cond.append(part[condition])
                
                    
        return tuple(cond_lists)