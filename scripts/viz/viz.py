# from scripts.classes import 
from scripts.signal import group_correlation_permutation_test, bootstrap_correlation
from scripts.stat import permutation_test, omni_stat
from scripts.util import hyperbole_fit, print_omni_dict, scale_to_freq
# from scripts.viz import 

from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ranksums, wilcoxon, ttest_rel, ttest_ind

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from numpy import interp
from sklearn.metrics import auc 
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns; sns.set(font_scale=1.2,style='ticks',context='notebook',font= 'georgia')
import mne
import pandas as pd
from copy import deepcopy

def plot_hyperbole(ax):
    rc('font', family='sserif')
    x = np.arange(4, 100, 2)
    a, b, c = 30, 6.5, 0.4

    # fig = plt.figure(figsize=(4, 4))
    ax.plot(x, hyperbole_fit(x, a, b, c), color='black', linewidth=2)
    ax.set_title('Hyperbolic Function', fontsize=14)
    ax.text(30, 2.5, r'$f(x) = \left(\frac{A}{x + B}\right)+c$', fontsize=14)
    ax.set_yticks(np.arange(1, 4))
    ax.set_xticks([0, 50, 100])
    # return fig

def scatter_plot(x1, x2, title='', stat='wilc', paired=False, n_perm=int(1e6), 
        legend=('Group A', 'Group B'), plot_stat_summary=True, effort=1, 
        colors=None, ownFig=True):
    ''' This function takes two vectors and plots them as scatter 
    plot and adds statistics to title'''

    if len(x1) > 0:
        if not x1[0]:
            return
    elif not x1:
        return
    if colors is None:
        colors = ['#0066ff', '#ff9900']
    if type(x1) == list:
        x1 = np.asarray(x1)
    if type(x2) == list:
        x2 = np.asarray(x2)

    val = [-2, 2]
    pos_x1 = val[0] + (np.random.rand(x1.size) * 2 - 1) * 0.3
    pos_x2 = val[1] + (np.random.rand(x2.size) * 2 - 1) * 0.3
    if stat=='wilc':
        if paired:
            t, p = wilcoxon(x1, x2)
        else:
            t, p = ranksums(x1, x2)
    elif stat=='ttest':
        if paired:
            t, p = ttest_rel(x1, x2)
        else:
            t, p = ttest_ind(x1, x2)
    elif stat == 'perm':
        p = permutation_test(x1, x2, paired=paired, tails=2, plot_me=False, n_perm=n_perm)
        t = 0
    if ownFig:
        plt.figure()
    plt.plot(pos_x1, x1, 'o', color=colors[0], markeredgecolor='black', label=legend[0])
    bbox_props = dict(facecolor=colors[0], alpha=0.6)
    plt.boxplot(x1, positions=[-2], patch_artist=True, notch=False, bootstrap=1000, whis=[5, 95], widths=0.5, boxprops=bbox_props, capprops=dict(color='black'), medianprops=dict(color='black', linewidth=2), showfliers=False)


    plt.plot(pos_x2, x2, 'o', color=colors[1], markeredgecolor='black', label=legend[1])
    bbox_props = dict(facecolor=colors[1], alpha=0.6)
    plt.boxplot(x2, positions=[2], patch_artist=True, notch=False, bootstrap=1000, whis=[5, 95], widths=0.5, boxprops=bbox_props, capprops=dict(color='black'), medianprops=dict(color='black', linewidth=2), showfliers=False)

    first_patch = mpatches.Patch(color=colors[0], label=legend[0])
    second_patch = mpatches.Patch(color=colors[1], label=legend[1])
    plt.legend(handles=[first_patch, second_patch])


    plt.title(f'{title}')
    
    if plot_stat_summary:
        om = omni_stat(x1, x2, paired=False, tails=2, verbose=0, effort=effort)
        txt = print_omni_dict(om)

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        pos_x = -5.9
        pos_y = np.max(np.concatenate((x1, x2), axis=0))
        plt.text(pos_x, pos_y, txt, fontsize=8, verticalalignment='top', bbox=props)
        plt.xlim((-6, 6))
    else:
        plt.xlim((-5, 5))

    plt.show()

def plot_roc(data, label, C, gamma, title):
    '''Function that uses cross validation to get an ROC curve of each fold, resulting in an average ROC curve.
    Uses the standard scaler function to normalise training data and apply the values found in the training set
    on the test set.
    input: data array(X), label array(y), SVM parameters C and gamma, and a string for the title'''
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.125, random_state=70)
    clf = SVC(kernel = 'rbf', C=C, gamma=gamma) #rbf kernel because it requires no background knowledge
    X = data
    y = label
    scaler = StandardScaler()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(sss.split(X, y)):
        #normalising the training data and applying the training sets mean and std on the testing set
        X[train] = scaler.fit_transform(X[train])
        X[test] = scaler.transform(X[test])

        clf.fit(X[train], y[train])
        
        viz = plot_roc_curve(clf, X[test], y[test], 
                         name='Fold {}'.format(i+1),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orangered',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='mediumblue',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='cornflowerblue', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title= title)
    ax.set_aspect('equal', 'box')
    
    #new legend because of 100 folds make the legend above very big. and i dont know how to remove the legend in code above
    ax.get_legend().remove()
    
    custom_lines = [Line2D([0], [0], linestyle='--', color='orangered', lw=2),
                    Line2D([0], [0], color='mediumblue', lw=2),
                    Line2D([0], [0], color='cornflowerblue', lw=8,alpha=.2)]
    
    ax.legend(custom_lines,['Chance','Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),r'$\pm$ 1 std. dev.'],loc='center',bbox_to_anchor=(1.5, 0.45))
    #plt.show()

def plot_two_with_error(X, Y1, Y2, measure='SEM', labels=['Group A', 'Group B'], 
        title='', ylabel='', xlabel='', xticklabels=None, test='perm', colors=None):
    '''
    Plots two Grand mean traces with underlying error shading
    Parameters:
    -----------
    X : Time array
    Y1 : Group of observations 1
    Y2 : Group of observations 2
    measure : ['SEM', 'SD'] Choose standard error of the mean (SEM) or standard deviation (SD)
    test : ['perm', 'wilc', 'ttest', None]
    '''
    if type(Y1) == list:
        Y1 = np.squeeze(np.array(Y1))
        Y2 = np.squeeze(np.array(Y2))

    if colors is None:
        # colors = ['blue', 'orange']
        colors = ['#1f77b4', '#ff7f0e']
    
    
    m_y1 = np.mean(Y1, axis=0)
    sd_y1 = np.std(Y1, axis=0)

    m_y2 = np.mean(Y2, axis=0)
    sd_y2 = np.std(Y2, axis=0)

    if measure=='SEM':
        sd_y1 = np.array(sd_y1)
        sd_y1 /= np.sqrt(len(Y1))
        sd_y2 = np.array(sd_y2)
        sd_y2 /= np.sqrt(len(Y2))
    elif measure=='SD' or measure=='STD':
        print('')
    else:
        print(f'measure {measure} not available. Choose SEM or SD.')
        return

    if test is not None:
        ax1 = plt.subplot(211)
    if len(X) == 0:
        X = np.arange(len(m_y1))
    # Plot Y1
    plt.plot(X, m_y1, label=labels[0], color=colors[0])
    plt.fill_between(X, m_y1-sd_y1, m_y1+sd_y1, alpha=0.3, color=colors[0])
    # Plot Y2
    plt.plot(X, m_y2, label=labels[1], color=colors[1])
    plt.fill_between(X, m_y2-sd_y2, m_y2+sd_y2, alpha=0.3, color=colors[1])
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xticklabels != None:
        plt.xticks(X, xticklabels)
    plt.title(title)
    if test is not None:
        # P-values
        ax2 = plt.subplot(212)
        p_vals = np.zeros_like(m_y1)
        for i in range(len(p_vals)):
            if test == 'perm':
                p_vals[i] = permutation_test(Y1[:, i], Y2[:, i])
            if test == 'wilc':
                p_vals[i] = ranksums(Y1[:, i], Y2[:, i])[1]
            if test == 'ttest':
                p_vals[i] = ttest_ind(Y1[:, i], Y2[:, i])[1]
        plt.plot(X, p_vals, color='dimgrey')  # , aspect='auto')
        plt.plot(X, np.ones_like(X)*0.05, '--', color='orangered')
        plt.yscale('log')
        plt.title(f'{test} p-value')
        if xticklabels != None:
            plt.xticks(X, xticklabels)
    # Styling
    # if squareplot:
    #     square_axis(ax1)
    #     square_axis(ax2)
        
    plt.tight_layout(pad=1)

def square_axis(ax):
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

# def plot_two_with_error(X, Y1, Y2, measure='SEM', labels=['Group A', 'Group B'], 
#         title='', ylabel='', xlabel='', xticklabels=None, test='Perm.'):
#     '''
#     Plots two Grand mean traces with underlying error shading
#     Parameters:
#     -----------
#     X : Time array
#     Y1 : Group of observations 1
#     Y2 : Group of observations 2
#     measure : ['SEM', 'SD'] Choose standard error of the mean (SEM) or standard deviation (SD)
#     test : ['perm', 'wilc', 'ttest']
#     '''
#     if type(Y1) == list:
#         Y1 = np.squeeze(np.array(Y1))
#         Y2 = np.squeeze(np.array(Y2))

#     m_y1 = np.mean(Y1, axis=0)
#     sd_y1 = np.std(Y1, axis=0)

#     m_y2 = np.mean(Y2, axis=0)
#     sd_y2 = np.std(Y2, axis=0)
#     if measure=='SEM':
#         sd_y1 = np.array(sd_y1)
#         sd_y1 /= np.sqrt(len(Y1))
#         sd_y2 = np.array(sd_y2)
#         sd_y2 /= np.sqrt(len(Y2))
#     elif measure=='SD' or measure=='STD':
#         print('')
#     else:
#         print(f'measure {measure} not available. Choose SEM or SD.')
#         return
#     #plt.figure()
#     plt.figure(figsize=(6,11))
#     plt.subplot(211)
#     if len(X) == 0:
#         X = np.arange(len(m_y1))
#     # Plot Y1
#     plt.plot(X, m_y1, label=labels[0],color='darkblue')
#     plt.fill_between(X, m_y1-sd_y1, m_y1+sd_y1, alpha=0.3,color='darkblue')
#     # Plot Y2
#     plt.plot(X, m_y2, label=labels[1],color='orange')
#     plt.fill_between(X, m_y2-sd_y2, m_y2+sd_y2, alpha=0.3,color='orange')
#     #plt.legend()
#     plt.legend(loc='best',ncol=2)
#     plt.ylabel(ylabel)
#     plt.xlabel(xlabel)
    
    
#     if xticklabels != None:
#         plt.xticks(X, xticklabels)
#     plt.xticks(np.arange(0,22,2))
#     plt.title(title)
#     # P-values
#     plt.subplot(212)
#     p_vals = np.zeros_like(m_y1)
#     for i in range(len(p_vals)):
#         if test == 'Perm.':
#             p_vals[i] = permutation_test(Y1[:, i], Y2[:, i])
#         if test == 'wilc':
#             p_vals[i] = ranksums(Y1[:, i], Y2[:, i])[1]
#         if test == 'ttest':
#             p_vals[i] = ttest_ind(Y1[:, i], Y2[:, i])[1]
#     plt.plot(X, p_vals,color='dimgray')  # , aspect='auto')
#     plt.plot(X, np.ones_like(X)*0.05, '--',label='0.05',color='orangered')
#     plt.legend()
#     plt.yscale('log')
    
#     ####setting the ticks for the y axis p value plot
#     ax = plt.gca()
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
#     #finding lowest and highest values for ylabel
#     min_pval = np.min(p_vals)
#     max_pval = np.max(p_vals)
#     #creating ticks log. spaced
#     yticks = [i*(10**j) for j in range(-5,2) for i in range(1,11)]
#     #set minor ticks
#     ax.set_yticks(yticks,minor=True)
#     #set y ticklabels to be empty
#     ax.set_yticklabels(['' for x in yticks],minor=True)
#     #setting major ticks to be min and max of pvals
#     ax.set_yticks([min_pval,max_pval],minor=False)
#     #setting ylims and xlims for pval plot
#     if min_pval > 0.05:
#         plt.ylim([0.05-(0.05/4),max_pval+(max_pval/4)])
#     else:     
#         plt.ylim([min_pval-(min_pval/4),max_pval+(max_pval/4)])
#     plt.xlabel(xlabel)
#     plt.ylabel('P-value')

    
#     plt.title(f'{test} P-value')
#     if xticklabels != None:
#         plt.xticks(X, xticklabels)
#     plt.xticks(np.arange(0,22,2))
#     plt.tight_layout(pad=2)

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    
    Example
    -------
    from scripts.viz import circular_hist
    import numpy as np

    phase = np.pi / 4
    t = np.linspace(0, 10, num=200, endpoint=False)
    lag = np.pi/3
    noise = 0.8

    y1 = np.cos(2 * np.pi * t + phase) + np.random.randn(len(t)) * noise
    y2 = np.cos(2 * np.pi * t + phase+lag) + np.random.randn(len(t)) * noise
    
    phase = np.angle(sig.hilbert(y1)) - np.angle(sig.hilbert(y2))
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    # Visualise by area of bins
    circular_hist(ax[0], phase)
    """
    
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def plot_mse(mse, time, sr):

    scales = np.arange(mse.shape[0]) + 1
    freqs = scale_to_freq(scales, sr)
    df_mse = []
    df_time = []
    df_freqs = []
    df_scales = []
    for i, freq in enumerate(freqs):
        for j, t in enumerate(time):
            df_mse.append(mse[i, j])
            df_time.append(round(t, 2))
            df_freqs.append(round(freq, 2))
            df_scales.append(round(scales[i], 2))
    d = {'mse': df_mse, 'time': df_time, 'scales': df_scales, 'freqs': df_freqs}
    df = pd.DataFrame(data=d)
    pivot = df.pivot('freqs', 'time', 'mse')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(pivot, ax=ax)
    return fig

def plot_pca_results(pca, transformedData, labels):
    legend = ('Controls', 'ASD')
    nComponents = pca.n_components
    plt.figure(num=123)
    controlIndices, patIndices = (np.where(labels==0)[0], np.where(labels==1)[0])
    for n in range(nComponents):
        plotNumber = 101 + 10*nComponents + n
        print(plotNumber)
        plt.subplot(plotNumber)
        scatter_plot(transformedData[controlIndices, n], transformedData[patIndices, n], ownFig=False, legend=legend)
        text = f'Comp {n+1} ({100*pca.explained_variance_ratio_[n]:.1f}% var expl.)'
        plt.title(text)

    plt.figure()
    for n in range(nComponents):
        plt.plot(pca.components_[n], label=f'C{n+1}')
    plt.legend()


def subgroup_correlation_plot(var1, var2, corr, groupIndices, var1name, var2name):
    ratio = 1
    controlIndices, patIndices = groupIndices
    r, p = corr(var1, var2)
    p_intergroup = group_correlation_permutation_test(corr, var1, var2, (controlIndices, patIndices), plot=False)
    _, lower, upper = bootstrap_correlation(corr, var1, var2)
    summary = f'Corr({var1name}, {var2name}): {r:.2f}({lower:.2f}-{upper:.2f}) (p={p:.3f}). p_equal={p_intergroup:.3f}'
    r_controls, p_controls = corr(var1[controlIndices], var2[controlIndices])
    r_patients, p_patients = corr(var1[patIndices], var2[patIndices])
    print(f'Corr controls: {r_controls:.2f} (p={p_controls:.3f})\n')
    print(f'Corr patients: {r_patients:.2f} (p={p_patients:.3f})\n')
    

    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(var1[controlIndices], var2[controlIndices], color='darkorange', label='Controls')
    ax.text(np.mean(var1[controlIndices]), np.mean(var2[controlIndices]), f'r={r_controls:.2f}')

    ax.scatter(var1[patIndices], var2[patIndices], color='blue', label='ASD')
    ax.text(np.mean(var1[patIndices]), np.mean(var2[patIndices]), f'r={r_patients:.2f}')
    ax.set_xlabel(var1name)
    ax.set_ylabel(var2name)
    ax.set_title(summary)
    ax.legend()
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

def plot_itv(itv, times, ax, info, cond='', title=None, alpha=0.01, ch_picks=None):
    if ch_picks is None:
        ch_picks = info.ch_names
    
    ch_indices = [info.ch_names.index(ch) for ch in ch_picks]

    M = np.nanmean(itv[0][:, ch_indices, :], axis=(0,1))
    SEM = np.nanstd(np.nanmean(itv[0][:, ch_indices, :], axis=1), axis=0) / np.sqrt(itv[0].shape[0])
    ax.fill_between(times, M-SEM, M+SEM, alpha=0.5)
    ax.plot(times, M, label='Control')

    M = np.nanmean(itv[1][:, ch_indices, :], axis=(0,1))
    SEM = np.nanstd(np.nanmean(itv[1][:, ch_indices, :], axis=1), axis=0) / np.sqrt(itv[1].shape[0])
    ax.fill_between(times, M-SEM, M+SEM, alpha=0.5)
    ax.plot(times, M, label='ASD')

    significance_array = np.zeros(itv[0].shape[-1])
    
    for i in range(itv[0].shape[-1]):
        if np.any(np.isnan(itv[0][:, ch_indices, i])):
            print('nan!')
        t,p = ttest_ind(np.nanmean(itv[0][:, ch_indices, i], axis=-1), np.nanmean(itv[1][:, ch_indices, i], axis=-1))
        significance_array[i] = p<alpha
    
    ylim = plt.ylim()
    significance_array = significance_array.astype(np.float) * ylim[0]
    significance_array[significance_array==0] = np.nan
    ax.plot(times, significance_array, 'r-')
    
    if title is None:
        ax.set_title(cond)
    else:
        ax.set_title(title)

    # topoplot inlet
    topos = [np.mean(itv_group, axis=(0, 2)) for itv_group in itv]
    significance_mask = get_significance_topo_mask(itv, alpha=alpha)

    rect = [0.55,0.05,0.45,0.45]
    subax = ax.inset_axes(rect)
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
     linewidth=0, markersize=2.5)

    mne.viz.plot_topomap(topos[0]-topos[1], info, axes=subax, 
        mask=significance_mask, mask_params=mask_params)
    # subax.set_title('Control-ASD', fontsize=11)

def plot_etv(etv, ax, info, cond='', title=None, plot_inlet=False):
    times = np.arange(etv[0].shape[-1])
    M = np.mean(etv[0][:, :, :], axis=(0,1))
    SEM = np.std(np.mean(etv[0][:, :, :], axis=1), axis=0) / np.sqrt(etv[0].shape[0])
    ax.fill_between(times, M-SEM, M+SEM, alpha=0.5)
    ax.plot(times, M, label='Control')

    M = np.mean(etv[1][:, :, :], axis=(0,1))
    SEM = np.std(np.mean(etv[1][:, :, :], axis=1), axis=0) / np.sqrt(etv[1].shape[0])
    ax.fill_between(times, M-SEM, M+SEM, alpha=0.5)
    ax.plot(times, M, label='ASD')

    if title is None:
        ax.set_title(cond)
    else:
        ax.set_title(title)

    # topoplot inlet
    if plot_inlet:
        topos = [np.mean(etv_group, axis=(0, 2)) for etv_group in etv]
        significance_mask = get_significance_topo_mask(etv, alpha=0.01)

        rect = [0.55,0.05,0.45,0.45]
        subax = ax.inset_axes(rect)
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
        linewidth=0, markersize=2.5)

        mne.viz.plot_topomap(topos[0]-topos[1], info, axes=subax, 
            mask=significance_mask, mask_params=mask_params)
        # subax.set_title('Control-ASD', fontsize=11)

def get_significance_topo_mask(itv, alpha=0.05, dependence=False):
    group_dists = [np.mean(x, axis=-1) for x in itv]
    n_chan = itv[0].shape[1]
    if dependence:
        p_vals = np.array([ttest_rel(group_dists[0][:, i], group_dists[1][:, i])[1] for i in range(n_chan)])
    else:
        p_vals = np.array([ttest_ind(group_dists[0][:, i], group_dists[1][:, i])[1] for i in range(n_chan)])
    
    return p_vals<alpha

def image_plot(images, pvals, cmap='inferno', alpha=0.1):
    pvals_masked = deepcopy(pvals)

    pvals_masked = pvals_masked<alpha

    kwargs = dict(cmap=cmap, aspect='auto', vmin=0, vmax=0.5)

    plt.figure()
    plt.subplot(141)
    # sns.heatmap(np.mean(images[0], axis=0), cmap=cmap)
    plt.imshow(np.mean(images[0], axis=0), **kwargs)

    plt.subplot(142)
    # sns.heatmap(np.mean(images[1], axis=0), cmap=cmap)
    plt.imshow(np.mean(images[1], axis=0), **kwargs)

    plt.subplot(143)
    diff = np.mean(images[0], axis=0) - np.mean(images[1], axis=0)
    # sns.heatmap(diff, cmap=cmap)
    plt.imshow(diff, **kwargs)

    plt.subplot(144)
    diff_masked = diff
    diff_masked[~pvals_masked] = 0
    # sns.heatmap(diff_masked,cmap='RdBu')
    plt.imshow(diff_masked, **kwargs)
    plt.colorbar()
    plt.tight_layout(pad=2)

def sigline(ax, X, group_A, group_B, stat='rank', crits=(0.05, 0.01, 0.001), sigline_percentile=0.25, offset_percentile=0.05):
    colors = sns.color_palette("hls", 8)
    n_time = group_A.shape[1]
    p_vals = np.array([ranksums(group_A[:, i], group_B[:, i])[1] for i in range(n_time)])

    # Determine positions of the significance lines
    ylim = ax.get_ylim()
    lower_boundary = ((ylim[1]-ylim[0])*sigline_percentile) / (1-sigline_percentile)
    ax.set_ylim(bottom=ylim[0]-lower_boundary, top=ylim[1])
    ylim = ax.get_ylim()
    line_pos_end = ylim[0]+(ylim[1]-ylim[0])*offset_percentile
    line_pos_start = ylim[0]+(ylim[1]-ylim[0])*sigline_percentile
    line_pos = np.linspace(line_pos_start, line_pos_end, num=len(crits))

    for i, (crit, pos, color) in enumerate(zip(crits, line_pos, colors)):
        sig_array = (p_vals<crit).astype(int)*pos
        sig_array[sig_array==0] = np.nan
        ax.plot(X, sig_array, color=color)
