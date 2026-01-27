"""
Plot for minimum gap statistics: correlated disorder 
Distribution of minimum gaps (high correlations)
"""

# ------- #
# Modules #
# ------- #

import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import pk_functions as pk

# --------- #
# Functions #
# --------- #

# ------------------------ #
# Operation specifications #
# ------------------------ #

# Display options
flag_latexFonts = True # Enable latex fonts
flag_titleOff = True # Remove title 
flag_textbox = True # Display user-defined textbox
flag_variance = False # Convert references to the disorder ratio into the variance. Incompatible with flag_std
flag_std = True # Convert references to the disorder ratio into the standard deviation. Incompatible with flag_variance
flag_mingap_std = True # Convert references to the minimum gap variance into the standard deviation

# Display additional plotting elements
flag_plot_cleanGap = True # Plot vertical line representing clean minimum gap
flag_plot_xZeroLine = False # Plot a vertical line at x = 0. This is intended to be used when the horizontal axis is the minimum gap subtracted by its clean value. 

# Other
# -> The following flags are primarily used to load data without errors; they play no role in plotting
flag_Nfix = False # Fix the number of time steps
flag_timeoverride = False # The total times tau_init and tau_final are no longer relative to tau_rel
flag_SVD = False # Enable SVD method in calculation of unitary during the calculation
flag_SVD_END = True # Enable the SVD method at the end of the calculation

# Save figures
flag_saveFigs = False 

# ----------------------- #
# Protocol specifications #
# ----------------------- #

# Tuning function
# -> LIN: Linear
# -> SMOOTH: Smooth (sin^2)
# -> FQD: Fast-QUAD (adapted from Felix's/Bill C.'s paper)
# -> TRI: Tri-region. Custom tuning function which interpolates (cubic-ly) between regions
# -> TRIB: Tri-region, B-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a cubic function to interpolate between the regions
# -> TRIC: Tri-region, C-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a Gaussian kernel to smoothen tuning function
# -> TRID: Tri-region, D-variant. Custom tuning function. Uses a Gaussian kernel to smoothen tuning function
# -> TRIE: Tri-region, E-variant. Same as D-variant, but introduces a scale factor f = slope_mid*s_time_m. s_time_m is eliminated as a parameter in favour of this scale factor
str_tuning_choice = ["SMOOTH"]

# Disorder
# -> NONE: No disorder
# -> UCUN: Uncorrelated, uniformly distributed
# -> UCND: Uncorrelated, normally/Gaussian distributed
# -> GCUN: Gaussian correlated, uniformly distributed
# -> GCND: Gaussian correlated, normally/Gaussian distributed
# -> ECND: Exponentially correlated, normally/Gaussian distributed
# -> SCND: Simply correlated, normally/Gaussian distributed
str_disorder_choice = ["GCND"]

# Noise
# -> NONE: Default, no noise
# -> WHT: White noise
# -> WHTC: White noise, with cutoff frequencies
# -> WHTCB: White noise with cutoff frequencies, replace the disorder strength with the amplitude of the power spectral density as a parameter
# -> 1F1: 1/f noise, with cutoff frequencies
# -> 1F1B: 1/f noise, replace the disorder strength with the amplitude of the power spectral density as a parameter
# -> SIN: Sine wave with a specified amplitude, phase, and frequency, meant to represent a single frequency mode in a noise signal
# -> SINAV: (Phase-averaged) Sine wave with a specified amplitude, phase, and frequency, meant to represent a single frequency mode in a noise signal
str_noise_choice = ["NONE"]

# ------------------- #
# Protocol parameters #
# ------------------- #

# System
L = np.array([30])
R = np.array([30])
Delta = np.array([0.6])
w = np.array([3.0])
muLeft = np.array([2.8])
muRightStart = np.array([3.2])
muRightEnd = np.array([2.8])

# Piano key step
lp = R
n_steps = np.array([1]) # Number of piano key steps; note that lp/(n_steps) must be an integer

# Tuning functions
# --> For reference, the default slope for SMOOTH is pi/2 (~1.57) in units of tau
s_time_init = np.array([0.0])
s_time_mid = np.array([0.0])
slope_mid = np.array([0.0])
scale_factor = np.array([0.0])

# Disorder and noise: maximum number of realizations
nbr_realizations_min = 1500

# Disorder
disorder_ratio = np.array([0.075, 0.1875, 0.375]) # Disorder strength (default: in units of the chemical potential difference)
length_corr = np.array([15.0]) # (GCND, SCND) Correlation length, (ECND) decay length

# Noise
noise_ratio = np.array([0.0]) # Noise strength in units of the chemical potential difference
noise_psd_ampl = np.array([0.0]) # Amplitude of the power spectral density (note: A = 1e-7 corresponds roughly to a variance of 1e-5)
w_cutoff_low = np.array([0.0]) # Low frequency cutoff
w_cutoff_high = np.array([0.0]) # High frequency cutoff

# Noise: single mode
# -> noise_ratio will be used to determine the amplitude
freq_sin = np.array([0.0]) # Frequency
phase_sin = np.array([0.0]) # Phase

# -------------------------- #
# Directories and file names #
# -------------------------- #

# Directories
dirname = "data_mingap/managed" # Directory of raw minimum gap data
dirname_plots = "plots" # Directory to store plots

# ------------------ #
# Rescale parameters #
# ------------------ #

# Legend of useful keys and symbols
# length_corr, disorder_ratio
# l_{{\mathrm{{p}}}}, \Delta_{{\mathrm{{min}},0}}

# Useful rescale variables
delta_mu = 0.5*np.abs(muRightEnd - muRightStart)
tau_0 = pk.tau_LZ(w, muRightStart, lp, Delta)
Delta_m = np.pi*Delta/(lp)
hbar = (6.582e-16)*1e3 # hbar, in units of meV s
#tau_rescale = 1/hbar/1e9

# Rescale variables and symbls
# -> Note that for dict_rescale_values, each value must have the same size as nbr_datasets
nbr_datasets_rescale = 3
dict_rescale_values = {}
dict_rescale_symbols = {}
dict_rescale_values = {"disorder_ratio":np.full(nbr_datasets_rescale, w), "length_corr":np.full(nbr_datasets_rescale, lp)}
dict_rescale_symbols = {"disorder_ratio":"w", "length_corr":"R"}

# ---------------------------------- #
# Rescale and/or add shift to x-data #
# -----------------------------------#

# Legend of useful keys and symbols
# l_{{\mathrm{{p}}}}, \Delta_{{\mathrm{{m}},0}}

xaxis_shift_values = np.array([])
xaxis_shift_symbol = ""

xaxis_rescale_values = 1/np.full(nbr_datasets_rescale, w)
xaxis_rescale_symbol = r"/ w"


# --------------- #
# Plot parameters #
# --------------- #

# Histogram
nbr_bins = 50

# Figure sizes
figsize_length = 6
figsize_height = 6

# Legend
flag_legend = True
size_legend = 14.5
loc_legend = (0.54, 0.65)

# Font and label sizes
size_title = 10
size_axislabel_x = 20
size_axislabel_y = 20
size_ticklabel_x = 18
size_ticklabel_y = 18
size_tickmajor_x = 12
size_tickminor_x = 6
size_tickmajor_y = 12
size_tickminor_y = 6
size_text = 18

# Linewidth and markersizes
size_linewidth = 2.0
size_marker = 8.0

# Data transparency
alpha = 0.8

# Tick locations
#locate_majorticks = np.round(x_max/5/1)*0.5

# Axis scales
scale_x = "linear"
scale_y = "linear"

# Axis limits
flag_useAxisLims = True
x_min = -0.00
x_max = 0.05
y_min = 1
y_max = None

# Spine linewidth
spine_linewidth = 1.75

# Other
option_bboxinches = "tight" # Can use "tight" or 0

# ----------------------- #
# Plot textbox parameters #
# ----------------------- #

# Location
txt_x = 0.075
txt_y = 0.9

# String
#txt_str = r"$\xi/R = {:.2f}$".format(length_corr[0]/lp[0])
txt_str = r"$\xi/R = {:.1f}$".format(length_corr[0]/lp[0])

# Fontsizes
txt_fontsize = 18

# --------- #
# Variables #
# --------- #

# List of noise protocols where averaging is not present
list_noise_noAvg = ["NONE", "SIN"]

# ----------- #
# Latex fonts #
# ----------- #

if flag_latexFonts == True:
    #pl.rc('text', usetex=True)
    #pl.rcParams["backend"] = "ps"
    #pl.rcParams['text.usetex'] = True
    pl.rcParams.update({"text.usetex":True, "font.family":"sans-serif", "font.sans-serif":"Helvetica"})
else:
    pl.rcdefaults()
    
# ------------ #
# Dictionaries #
# ------------ #

# Dictionaries for all parameters
dict_specifications_all = {"tuning":str_tuning_choice, "disorder": str_disorder_choice, "noise":str_noise_choice}
dict_protocol_all = {"L":L, "R":R, "Delta":Delta, "w":w, "muLeft":muLeft, "muRightStart":muRightStart, "muRightEnd":muRightEnd, "lp":lp, "n_steps":n_steps, "tau":[100.0], "s_time_init":s_time_init, "s_time_mid":s_time_mid, "slope_mid":slope_mid, "scale_factor":scale_factor, "nbr_realizations_min":nbr_realizations_min}
dict_disorder_all = {"disorder_ratio":disorder_ratio, "length_corr":length_corr}
dict_noise_all = {"noise_ratio":noise_ratio, "noise_psd_ampl":noise_psd_ampl, "w_cutoff_low":w_cutoff_low, "w_cutoff_high":w_cutoff_high, "freq_sin":freq_sin, "phase_sin":phase_sin}

# Dictionary for varying parameters
dict_param_choice = {"TAU":"tau", "STIMES":"s_time_init",  "STIMEM":"s_time_mid",  "SLOPE":"slope_mid", "NR":"noise_ratio", "NA":"noise_psd_ampl", "WCL":"w_cutoff_low", "WCH":"w_cutoff_high", "W":"freq_sin", "PH":"phase_sin", "DR":"disorder_ratio", "LC":"length_corr"}

# Determine the number of data sets based on number of values requested for a given parameter
nbr_datasets = 1
keys_exception = ["nbr_realizations_min"]
keys_noadjust = []
for dict_all in [dict_specifications_all, dict_protocol_all, dict_disorder_all, dict_noise_all]:
    for key in dict_all.keys():
        if key in keys_exception:
            continue
        if len(dict_all[key]) > 1:
            nbr_datasets = len(dict_all[key])
            keys_noadjust.append(key)

# Adjust all values in dictionaries so that their lengths match that of nbr_dataset
for key in dict_specifications_all.keys():
    if key in keys_exception or key in keys_noadjust:
        continue
    dict_specifications_all[key] = [dict_specifications_all[key][0]]*nbr_datasets
    
for key in dict_protocol_all.keys():
    if key in keys_exception or key in keys_noadjust:
        continue
    dict_protocol_all[key] = np.full(nbr_datasets, dict_protocol_all[key][0])
    
for key in dict_disorder_all.keys():
    if key in keys_exception or key in keys_noadjust:
        continue
    dict_disorder_all[key] = np.full(nbr_datasets, dict_disorder_all[key][0])
    
for key in dict_noise_all.keys():
    if key in keys_exception or key in keys_noadjust:
        continue
    dict_noise_all[key] = np.full(nbr_datasets, dict_noise_all[key][0])
    
# -------------------------------- #
# Calculation of disorder variance #
# -------------------------------- #

# For each data set, convert the disorder strength into the disorder variance
disorder_variance = np.zeros(nbr_datasets)
for ind_data in range(0, nbr_datasets):
    disorder_type = dict_specifications_all["disorder"][ind_data]
    chempot_change = np.abs(dict_protocol_all["muRightStart"] - dict_protocol_all["muRightEnd"])[ind_data]
    if "ND" in disorder_type:
        disorder_variance[ind_data] = (dict_disorder_all["disorder_ratio"][ind_data]*chempot_change)**2
    if "UN" in disorder_type:
        disorder_variance[ind_data] = (2.0*dict_disorder_all["disorder_ratio"][ind_data]*chempot_change)**2/12.0

# Update appropriate dictionaries
if flag_variance == True:
    # Note that we need to be careful with the insertion order of the dictionary
    dict_disorder_all_temp = {}
    dict_disorder_all_temp["disorder_variance"] = disorder_variance
    dict_disorder_all_temp.update(dict_disorder_all)
    dict_disorder_all = dict_disorder_all_temp
    
if flag_std == True:
    # Note that we need to be careful with the insertion order of the dictionary
    dict_disorder_all_temp = {}
    dict_disorder_all_temp["disorder_std"] = np.sqrt(disorder_variance)
    dict_disorder_all_temp.update(dict_disorder_all)
    dict_disorder_all = dict_disorder_all_temp
    
# -------------------- #
# Extract data to plot #
# -------------------- #

# List of arrays for each data set
list_mingap_data_set = []

# Loop over data files to load
for ind_data in range(0, nbr_datasets):
    
    # Establish dictionaries
    dict_operations_main = {"cluster":False, "onesim":False, "average":False}
    dict_operations_other = {"savedata":False, "Nfix":flag_Nfix, "timeoverride":flag_timeoverride, "SVD":flag_SVD, "SVD_END":flag_SVD_END}
    dict_specifications = {key:dict_specifications_all[key][ind_data] for key in dict_specifications_all.keys()}
    dict_protocol = {key:dict_protocol_all[key][ind_data] for key in dict_protocol_all.keys() if key not in keys_exception}
    dict_protocol.update({"dt":0, "Nthresh":0, "Nfix":0, "nbr_realizations":0})
    dict_disorder = {key:dict_disorder_all[key][ind_data] for key in dict_disorder_all.keys()}
    dict_noise = {key:dict_noise_all[key][ind_data] for key in dict_noise_all.keys()}
    dict_noise.update({"tau_noise":0, "dt_noise":0})
    dict_other = {"NSamp":0, "str_desig":"", "str_desig2":""}
    
    # Establish file name and array of parameters
    filename, arr_params = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=False, flag_noTau=True)
    
    # Load file
    with np.load("{}/mingap{}.npz".format(dirname, filename)) as file:
        list_mingap_data_set.append(file['data'])

# --------------------------- #
# General plotting parameters #
# --------------------------- #

# Color palettes 
#color_black = "#4d4d4d"
color_black = "Black"
#list_colors = ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB"]
#list_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
#list_colors = ['#6699CC', '#004488', '#EECC66', '#994455', '#997700', '#EE99AA']
#list_colors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']
list_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

# Marker styles
#list_markers = ['o','v', '^', '<', '>', 's', 'p', 'P', '4', '8', '1', '2', '3', '*', 'h', 'H','+','x','X','D','d','|']
list_markers = ['o','v', "s", "D", "*", "h"]

# Generate title and legend text strings
str_title, list_legend = pk.str_plot(nbr_datasets, dict_specifications_all, dict_protocol_all, dict_disorder_all, dict_noise_all, False, dict_rescale_values=dict_rescale_values, dict_rescale_symbols=dict_rescale_symbols, flag_disorder_replace_var=flag_variance, flag_disorder_replace_std=flag_std)
str_title = str_title.replace(" $\\tau = 100.00$,", "")

# Labels
dict_xlabels = {"TAU":r"$\tau$", "STIMES":r"$\tau_{\mathrm{i}} / \tau$", "STIMEM":r"$\tau_{\mathrm{m}} / \tau$", "SLOPE":r"Slope $\alpha$", "WCL":r"$\omega_{{\mathrm{l}}}$", "WCH":r"$\omega_{{\mathrm{h}}}$", "NR":r"$r_{{\mathrm{n}}}$", "NA":r"$A_{{\mathrm{psd}}}$", "W":r"$\omega$", "PH":r"$\phi$", "DR":r"$r_{{\mathrm{{d}}}}$", "LC":r"$\xi$"}
dict_ylabels = {"mingap":r"$\Delta_{{\mathrm{{m}}}}$"}
#dict_ylabels = {"diabErr":r"$\mathcal{P}$", "diabErr_cc":r"$\mathcal{P}_{\mathrm{cc}}$", "diabErr_ct":r"$\mathcal{P}_{\mathrm{ct}}$"}

# ----- #
# Plots #
# ----- #

# ->-----------------------<- #
# Distribution of minimum gap #
# ->-----------------------<- #

# Set up figures and axes
fig = pl.figure(figsize=(figsize_length, figsize_height))
ax = fig.add_subplot()
    
# Loop over data sets
for ind_data in range(0, nbr_datasets):
    
    # x-data 
    x_data = 2.0*list_mingap_data_set[ind_data][:nbr_realizations_min]
    
    # Axis labels
    xlabel_plot = r"$\Delta_{\mathrm{m}}$"    
    ylabel_plot = "Probability density"
    
    # Shift and/or rescale x-data
    if xaxis_shift_values.shape[0] != 0:
        x_data = x_data + xaxis_shift_values[ind_data] 
        xlabel_plot = xlabel_plot + r"${}$".format(xaxis_shift_symbol)
        if xaxis_rescale_values.shape[0] != 0:
            xlabel_plot = "(" + xlabel_plot + ")"
    if xaxis_rescale_values.shape[0] != 0:
        x_data = x_data*xaxis_rescale_values[ind_data]  
        if xaxis_shift_values.shape[0] != 0:
            xlabel_plot = xlabel_plot + r"${}$".format(xaxis_rescale_symbol)
        else:
            #xlabel_plot = r"${}$".format(xaxis_rescale_symbol) + xlabel_plot
            xlabel_plot = xlabel_plot + r"${}$".format(xaxis_rescale_symbol)
            
    # Set up bins and plot histogram
    # -> Note that if density=true, a probability density is drawn, i.e. each bin's raw count will be divided by (total number of counts)*(width of the bin). This ensures that the area under the histogram is 1.
    # -> Note that the true minimum gap is actually twice the amount of the data. See notes.
    # -> histtypes can be "bar", "barstacked", "step", "stepfilled"
    
    #counts, bins, patches = ax.hist(list_mingap_data_set[ind_data], nbr_bins, density=True, label=", ".join(list_legend[ind_data]))
    ax.hist(x_data, nbr_bins, density=True, cumulative=False, histtype="step", color=list_colors[ind_data], linewidth=size_linewidth, label=", ".join(list_legend[ind_data]), alpha=alpha)
    
    """
    # Using the mean and std of the minimum gap, plot the pdf for a normal distribution
    sigma = muD_max[0]
    mean = np.average(list_mingaps[0])
    sigma = np.sqrt(np.average(list_mingaps[0]**2) - mean**2)
    pdf = 1/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*(bins - mean)**2/sigma**2)
    ax.plot(bins, pdf, "-", color="Red", linewidth=size_linewidth, label="Normal distribution")
    """
    
# Plot clean minimum gap has vertical line
# -> Note that the true minimum gap is actually twice the amount of the data. See notes.
if flag_plot_cleanGap == True:
    x_cleanGap = 0.5*(Delta*np.pi/lp)[0]/w
    ax.axvline(x=2.0*x_cleanGap, linestyle="--", markersize=size_marker, linewidth=size_linewidth, color=color_black, label=r"$\Delta_{{R}}$", alpha=1.0, zorder=100)
    
# Plot vertical line at x = 0.
if flag_plot_xZeroLine == True:
    ax.axvline(x=0.0, linestyle="--", markersize=size_marker, linewidth=size_linewidth, color=color_black, alpha=1.0, zorder=100)
    
# Plot user-defined textbox
if flag_textbox == True:
    ax.text(txt_x, txt_y, txt_str, transform=ax.transAxes, fontsize=txt_fontsize)

# Titles and labels
if flag_titleOff == False:
    ax.set_title("Kitaev chain - Piano key" + "\n" + str_title, fontsize=size_title, wrap=True, pad=15)
    
# Axis labels
#ylabel_plot = r"Count/$(N_{\mathrm{min}} w_{\mathrm{bin}})$"
ax.set_xlabel(xlabel_plot, fontsize=size_axislabel_x)
ax.set_ylabel(ylabel_plot, fontsize=size_axislabel_y, labelpad=8)
#ax.set_ylabel(diabErr_column + r"/$A_{psd}$", fontsize=size_axislabel_y, labelpad=8)

# Axis scales
ax.set_xscale(scale_x)
ax.set_yscale(scale_y)

# Axis ranges
if flag_useAxisLims == True:
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])    
#ax.set_xlim([0.001, 0.14])
#ax.set_xlim([0.001, 0.12])
#ax.set_ylim([y_min, y_max])

# Ticks and labels
# -> Locators for major ticks
#ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
#ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(locate_majorticks))
#ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=500))
#ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(locs = np.logspace(-9, 0, 10)))

# -> Locators for major and minor ticks
if scale_x == "linear":
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    
if scale_x == "log":
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
    
if scale_y == "linear":
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

if scale_y == "log":
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=500))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs=(.2, .3, .4, .5, .6, .7, .8, .9)))
    # Remove tick labels on y-axis which are odd powers
    fig.canvas.draw()
    yticks_major = ax.yaxis.get_major_ticks()
    for tick in yticks_major:
        if np.log10(tick.get_loc())%2 != 0.0:
            tick.label1.set_visible(False)

# Tick sizes
ax.xaxis.set_tick_params(which="major", direction="in", length=size_tickmajor_x, width=1.0, labelsize=size_ticklabel_x)
ax.xaxis.set_tick_params(which="minor", direction="in", length=size_tickminor_x)
ax.yaxis.set_tick_params(which="major", direction="in", length=size_tickmajor_y, width=1.0, labelsize=size_ticklabel_y)
ax.yaxis.set_tick_params(which="minor", direction="in", length=size_tickminor_y)

# -> Enable scientific notation on y-axis
#ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
#ax.xaxis.offsetText.set_fontsize(size_ticklabel_x)
#ax.get_xaxis().get_offset_text().set_position((1.0,0))

# Spines
pl.setp(ax.spines.values(), linewidth=spine_linewidth)

# Legend
if flag_legend == True:
    
    # Handles and labels
    leg_handles, leg_labels = pl.gca().get_legend_handles_labels()
    
    # Reorganize handles/labels so that clean results come first
    if flag_plot_cleanGap == True:
        leg_order = np.concatenate(([-1], np.arange(0, len(leg_handles)-1)))
        leg_handles = [leg_handles[i] for i in leg_order]
        leg_labels = [leg_labels[i] for i in leg_order]
    
    #ax.legend(loc="lower right", fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=True)
    ax.legend(leg_handles, leg_labels, loc=loc_legend, fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=False).set_zorder(201)
    
# Other options
#pl.tight_layout()
pl.subplots_adjust(left=0.25, bottom=0.25)

# Save figure
if flag_saveFigs == True:
    pl.savefig("{}/mingap_uncorr_distr_c.pdf".format(dirname_plots), format='pdf',bbox_inches=option_bboxinches)

pl.show()

        
        



