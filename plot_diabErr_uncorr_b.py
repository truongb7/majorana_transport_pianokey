"""
Plot for diabatic error: uncorrelated disorder 
Average/typical diabatic error vs. disorder strength
"""

# ------- #
# Modules #
# ------- #

import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
import pandas as pd
import os as os
import pk_functions as pk
from matplotlib.legend_handler import HandlerTuple

# ------------------------ #
# Operation specifications #
# ------------------------ #

# Data type to load and plot
# -> diabErr, diabErr_cc, diabErr_ct
dataType = "diabErr"

# Display options
flag_latexFonts = True # Enable latex fonts
flag_titleOff = True # Remove title 
flag_titleNoNmin = False # Remove minimum number of realizations in title
flag_variance = False # Convert references to the disorder ratio into the variance. Incompatible with flag_std
flag_std = True # Convert references to the disorder ratio into the standard deviation. Incompatible with flag_variance
flag_equalization = True # Enables ability to equally space data points according to a reference array (x_data_ref)
flag_hideLogErr = False # Hide geometric error
flag_sciNotation = False # Use standard scientific notation for labels
flag_leglabels_manual = False # Manually input legend labels

# Loading options
flag_clean = False # Clean data
flag_tls = False # Two-level system data
flag_fileManual = False # Load main data file manually
flag_dataManual = False # Load x/y data from a specific .npz file

# Display analytical expressions
flag_expr_diabVtau = False
flag_expr_diabVx_minGapAct = False

# Display additional plotting elements
flag_horizontal = False # Plot horizontal data
flag_relfreqs = False # Plot relevant frequencies corresponding to relevant energies

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

# Primary parameter to plot diabatic error against
# -> TAU: Total time  
# -> STIMES: Tuning function TRI, duration of initial/end regions
# -> STIMEM: Tuning function TRI, duration of middle region
# -> SLOPE: Tuning function TRI, slope of middle region
# -> DR: Disorder ratio
# -> LC: Correlation length
# -> NR: Noise ratio
# -> NA: Noise PSD amplitude
# -> WCL: Low frequency cutoff 
# -> WCH: High frequency cutoff
# -> W: Single mode frequency
# -> PH: Single mode phase
vary_param_choice = "DR"

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
str_disorder_choice = ["UCND"]

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

# Time
tau = np.array([1500.0]) # Protocol time

# Tuning functions
# --> For reference, the default slope for SMOOTH is pi/2 (~1.57) in units of tau
s_time_init = np.array([0.0])
s_time_mid = np.array([0.0])
slope_mid = np.array([0.0])
scale_factor = np.array([0.0])

# Disorder and noise: minimum number of realizations
nbr_realizations_min = 500

# Disorder
disorder_ratio = np.array([0.1875]) # Disorder strength (default: in units of the chemical potential difference)
length_corr = np.array([0.0]) # (GCND, SCND) Correlation length, (ECND) decay length

# Noise
noise_ratio = np.array([0.0]) # Noise strength in units of the chemical potential difference
noise_psd_ampl = np.array([0.0]) # Amplitude of the power spectral density (note: A = 1e-7 corresponds
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
dirname = "data_main" # Directory of main data file
dirname_plots = "plots" # Directory to store plots
dirname_mingap_managed = "data_mingap/managed" # Minimum gap data (all)

# Clean data general specifications
tuning_compare = "SMOOTH"
disorder_compare = "NONE"
noise_compare = "NONE"
dataType_compare = "diabErr"

# Manual loading
filename_manual = "TRI_NONE"

# Directories for two-level system data
dirname_tls = "data_main_tl"
filename_manual_tls = "CTL_SMOOTH_1F1B"

# Directory for active data-manual mode
dirname_dataManual = "figures"
filename_dataManual = "sinav_errVf_compare"

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
mingap_clean = 2.0*np.array([0.031644, 0.023750, 0.018984, 0.015804])
#tau_rescale = 1/hbar/1e9

# Rescale variables and symbols
# -> Note that for dict_rescale_values, each value must have the same size as nbr_datasets
nbr_datasets_rescale = 3
dict_rescale_values = {}
dict_rescale_symbols = {}

dict_rescale_values = {"tau":np.full(nbr_datasets_rescale, tau_0), "disorder_ratio":np.full(nbr_datasets_rescale, w), "length_corr":np.full(nbr_datasets_rescale, lp), "w_cutoff_high":np.full(nbr_datasets_rescale, mingap_clean[0]), "freq_sin":np.full(nbr_datasets_rescale, mingap_clean[0])}
dict_rescale_symbols = {"tau":r"\tau_{{\mathrm{{LZ}}}}", "disorder_ratio":"w", "length_corr":"R", "w_cutoff_high":"\Delta_{{\mathrm{{m}},0}}", "freq_sin":"\Delta_{{\mathrm{{m}},0}}"}

# ------------ #
# Data display #
# ------------ #

# Data equalization
x_data_ref = np.linspace(0.0, 0.08, 21)

# --------------- #
# Plot parameters #
# --------------- #

# Figure sizes
figsize_length = 7.5
figsize_height = 4

# Legend
flag_legend = False
size_legend = 14
loc_legend = "lower left"
size_leg_handle = 4.5
leg_labels_manual = ['Clean',
 '$A_{\\mathrm{w}} = 1.0 \\times 10^{-4}$',
 '$A_{\\mathrm{w}} = 1.0 \\times 10^{-2}$',
 '$A_{\\mathrm{w}} = 1.0$']

# Font and label sizes
size_title = 10
size_axislabel_x = 22
size_axislabel_y = 22
size_ticklabel_x = 20
size_ticklabel_y = 20
size_tickmajor_x = 12
size_tickminor_x = 6
size_tickmajor_y = 12
size_tickminor_y = 6
size_text = 20

# Linewidths and markersizes
size_linewidth = 2.0
size_marker = 9.0

# Data transparency
alpha = 0.8

# Tick locations
#locate_majorticks = np.round(x_max/5/1)*0.5

# Axis scales
scale_x = "linear"
scale_y = "log"

# Axis limits
flag_useAxisLims = True

x_min = -0.0025
x_max = 0.082
y_min = 1e-7
y_max = 1.0

# Spine linewidth
spine_linewidth = 1.75

# Other
option_bboxinches = "tight" # Can use "tight" or 0

# For results vs. frequency, textboxes indicating relevant energies/frequencies (clean)
spec_gap_min = 2*0.03170931176293834
spec_gap_max = 2*0.0949881384651461
spec_bandwidth_min = 2*2.898051356270677
spec_bandwidth_max = 2*3.0931262147722607
tau_gap_min = 2*np.pi/spec_gap_min
tau_gap_max = 2*np.pi/spec_gap_max
str_textbox = r"$\omega_{{\mathrm{{g,min}}}} = {:.3f}$".format(spec_gap_min) + "\n" + r"$\omega_{{\mathrm{{g,max}}}} = {:.3f}$".format(spec_gap_max) + "\n" + r"$\omega_{{\mathrm{{b,max}}}} = {:.3f}$".format(spec_bandwidth_max) + "\n"

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
dict_protocol_all = {"L":L, "R":R, "Delta":Delta, "w":w, "muLeft":muLeft, "muRightStart":muRightStart, "muRightEnd":muRightEnd, "lp":lp, "n_steps":n_steps, "tau":tau, "s_time_init":s_time_init, "s_time_mid":s_time_mid, "slope_mid":slope_mid, "scale_factor":scale_factor, "nbr_realizations_min":nbr_realizations_min}
dict_disorder_all = {"disorder_ratio":disorder_ratio, "length_corr":length_corr}
dict_noise_all = {"noise_ratio":noise_ratio, "noise_psd_ampl":noise_psd_ampl, "w_cutoff_low":w_cutoff_low, "w_cutoff_high":w_cutoff_high, "freq_sin":freq_sin, "phase_sin":phase_sin}

# Dictionary for parameters which are varied 
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

# ------------------------- #
# Main data file management #
# ------------------------- #

list_str_main_core = []
# Loop through all data sets and construct the core strings of the main data files in the format TUNING_DISORDER_NOISE
for ind_data in range(0, nbr_datasets):
    str_main_core = dict_specifications_all["tuning"][ind_data] + "_" + dict_specifications_all["disorder"][ind_data] + "_" + dict_specifications_all["noise"][ind_data]
    if str_main_core in list_str_main_core:
        continue
    else:
        list_str_main_core.append(str_main_core)
        
# Loop through all files in data_main, identify all those that correspond to list_strmain_core, and load the most updated ones
dict_data_filenames = {}
if flag_fileManual == False:
    list_mainfiles = os.listdir(dirname)
    for str_main_core in list_str_main_core:
        
        # Identify files that correspond to str_main_core
        list_correspondfiles = []
        for file in list_mainfiles:
            #if "CTL_" + str_main_core + "_" not in file:
            if "CTL_" + str_main_core + "." not in file:
                continue
            else:
                list_correspondfiles.append(file)
                
        latestfile = list_correspondfiles[0]
        
        """
        # Of the files identified, determine the most updated one
        valdate_largest = 0
        latestfile = ""
        for correspondfile in list_correspondfiles:
            valdate = int(correspondfile[-17:-4].replace("_",""))
            if valdate > valdate_largest:
                valdate_largest = valdate
                latestfile = correspondfile
        """
        
        # Save to dictionary
        dict_data_filenames[str_main_core] = latestfile
else:
    #dict_data_filenames[list_str_main_core[0]] = list_str_main_core[0] + ".csv"
    dict_data_filenames[list_str_main_core[0]] = filename_manual + ".csv"
    
# -------------- #
# Load main data #
# -------------- #

# Loop through dict_data_filename and load data files and save to dictionary
dict_data_files = {}
for key in dict_data_filenames.keys():
    dict_data_files[key] = pd.read_csv("{}/{}".format(dirname, dict_data_filenames[key]))

# Load clean data
if flag_clean == True:
    df_data_main_clean = pd.read_csv("{}/{}.csv".format(dirname, "CTL_" + tuning_compare + "_" + disorder_compare + "_" + noise_compare))

# -------------------- #
# Extract data to plot #
# -------------------- #

# List of data frames for each data set
list_df_data_set = []

# For clean data, run the following loop an extra time and append the data to the end of list_df_data_set
if flag_clean == True:
    count_clean = 1
else:
    count_clean = 0
    
# Loop over number of data sets
for ind_data in range(0, nbr_datasets + count_clean):
    
    if ind_data < nbr_datasets:
        ind_data_act = ind_data
        df_data_main = dict_data_files[dict_specifications_all["tuning"][ind_data_act] + "_" + dict_specifications_all["disorder"][ind_data_act] + "_" + dict_specifications_all["noise"][ind_data_act]]
        df_main = df_data_main
    else:
        ind_data_act = 0
        df_main = df_data_main_clean
    
    # Establish dictionaries
    dict_operations_main = {"cluster":False, "onesim":False, "average":False}
    dict_operations_other = {"savedata":False, "Nfix":flag_Nfix, "timeoverride":flag_timeoverride, "SVD":flag_SVD, "SVD_END":flag_SVD_END}
    
    dict_specifications = {key:dict_specifications_all[key][ind_data_act] for key in dict_specifications_all.keys()}
    dict_protocol = {key:dict_protocol_all[key][ind_data_act] for key in dict_protocol_all.keys() if key not in keys_exception}
    dict_protocol.update({"dt":0, "Nthresh":0, "Nfix":0, "nbr_realizations":0})
    dict_disorder = {key:dict_disorder_all[key][ind_data_act] for key in dict_disorder_all.keys()}
    dict_noise = {key:dict_noise_all[key][ind_data_act] for key in dict_noise_all.keys()}
    dict_noise.update({"tau_noise":0, "dt_noise":0})
    dict_other = {"NSamp":0, "str_desig":"", "str_desig2":""}
    
    # For clean data, replace str_disorder_choice and str_noise_choice in the dictionaries
    if ind_data == nbr_datasets:
        dict_specifications["tuning"] = tuning_compare
        dict_specifications["disorder"] = disorder_compare
        dict_specifications["noise"] = noise_compare
        
    # Establish parameter columns and set index of df_main to parameter columns
    cols_nodata = pk.create_csvcolumns_nodata(dict_specifications)
    cols_nodata.remove(dict_param_choice[vary_param_choice])
    df_main.set_index(cols_nodata, inplace=True)
    
    # Establish the multi-index for parameters with varied parameter accounted for
    multiIndex_pars = pk.create_csvmultiIndex(dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise)
    multiIndex_pars = multiIndex_pars.droplevel([dict_param_choice[vary_param_choice]])
    
    # Find all data corresponding to multi-index with varied parameter accounted for; reset index
    df_data_set = df_main.loc[multiIndex_pars]
    df_data_set.reset_index(inplace=True)
    df_main.reset_index(inplace=True)
    
    # In the case of disorder or noise, keep only the rows of data where the number of realizations >= nbr_realizations_min
    if dict_specifications["noise"] not in list_noise_noAvg or dict_specifications["disorder"] not in ["NONE"]:
        # Identify columns which have "_count" in their name
        #list_str_count = [str_count for str_count in list(df_data_set.columns) if "_count" in str_count]
        # Consider data only corresponding to dataType
        list_str_count = [dataType + '_count']
        for str_count in list_str_count:  
            df_data_set = df_data_set.drop(df_data_set[df_data_set[str_count] < nbr_realizations_min].index)
            
    # Sort the data according to the parameter choice
    df_data_set = df_data_set.sort_values(by=[dict_param_choice[vary_param_choice]])
    
    # Append to list
    list_df_data_set.append(df_data_set)
    
# Load two-level system data
if flag_tls == True:
    list_df_data_set_tls = pk.load_tls(dict_specifications_all, dict_protocol_all, dict_disorder_all, dict_noise_all, vary_param_choice, dirname_tls, filename_manual_tls, dataType, flag_filemanual=flag_fileManual)
    
# --------------------------- #
# General plotting parameters #
# --------------------------- #

# Color palettes 
color_black = "#4d4d4d"
#color_black = "Black"
#list_colors = ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB"]
#list_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
#list_colors = ['#6699CC', '#004488', '#EECC66', '#994455', '#997700', '#EE99AA']
#list_colors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']
list_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

# Marker styles
#list_markers = ['o','v', '^', '<', '>', 's', 'p', 'P', '4', '8', '1', '2', '3', '*', 'h', 'H','+','x','X','D','d','|']
list_markers = ['o','v', "s", "D", "P", "h"]

# Generate title and legend text strings
str_title, list_legend = pk.str_plot(nbr_datasets, dict_specifications_all, dict_protocol_all, dict_disorder_all, dict_noise_all, vary_param_choice, dict_rescale_values=dict_rescale_values, dict_rescale_symbols=dict_rescale_symbols, flag_disorder_replace_var=flag_variance, flag_disorder_replace_std=flag_std, flag_noNmin=flag_titleNoNmin, flag_sciNotation=flag_sciNotation)

# Labels
#dict_xlabels = {"TAU":r"$\tau$", "STIMES":r"$\tau_{\mathrm{i}} / \tau$", "STIMEM":r"$\tau_{\mathrm{m}} / \tau$", "SLOPE":r"Slope $\alpha$", "WCL":r"$\omega_{{\mathrm{l}}}$", "WCH":r"$\omega_{{\mathrm{h}}}$", "NR":r"$r_{{\mathrm{n}}}$", "NA":r"$A_{{\mathrm{psd}}}$", "W":r"$\omega$", "PH":r"$\phi$", "DR":r"$r_{{\mathrm{{d}}}}$", "LC":r"$\xi$"}
dict_ylabels = {"diabErr":r"$\mathcal{P}$", "diabErr_cc":r"$\mathcal{P}_{\mathrm{cc}}$", "diabErr_ct":r"$\mathcal{P}_{\mathrm{ct}}$"}
#dict_xlabels = {"TAU":r"$\tau$", "STIMES":r"$\tau_{\mathrm{i}} / \tau$", "STIMEM":r"$\tau_{\mathrm{m}} / \tau$", "SLOPE":r"Slope $\alpha$", "WCL":r"$\omega_{{\mathrm{l}}}$", "WCH":r"$\omega_{{\mathrm{h}}}$", "NR":r"$r_{{\mathrm{n}}}$", "NA":r"$A$", "W":r"$\omega$", "PH":r"$\phi$", "DR":r"$r_{{\mathrm{{d}}}}$", "LC":r"$\xi$"}
dict_xlabels = {"TAU":r"$\tau$", "STIMES":r"$\tau_{\mathrm{i}} / \tau$", "STIMEM":r"$\tau_{\mathrm{m}} / \tau$", "SLOPE":r"Slope $\alpha$", "WCL":r"$\omega_{{\mathrm{l}}}$", "WCH":r"$\omega_{{\mathrm{h}}}$", "NR":r"$r_{{\mathrm{n}}}$", "NA":r"$A_{{\mathrm{{w}}}}$", "W":r"$\omega$", "PH":r"$\phi$", "DR":r"$r_{{\mathrm{{d}}}}$", "LC":r"$\xi$"}
dict_ylabels = {"diabErr":r"$\mathcal{P}$", "diabErr_cc":r"$\mathcal{P}_{\mathrm{cc}}$", "diabErr_ct":r"$\mathcal{P}_{\mathrm{ct}}$"}
vary_column = dict_param_choice[vary_param_choice]
xlabel_plot = dict_xlabels[vary_param_choice]

# Consider case where vary_param_choice = disorder strength ("DR"): if desired, plot against the variance instead
if vary_param_choice == "DR" and flag_variance == True:
    xlabel_plot = r"$\sigma^2$"
if vary_param_choice == "DR" and flag_std == True:
    xlabel_plot = r"$\sigma$" 
    
# If x-axis is rescaled, change symbol accordingly
if dict_param_choice[vary_param_choice] in dict_rescale_values.keys():
    xlabel_plot = xlabel_plot + "/" + "$" + dict_rescale_symbols[dict_param_choice[vary_param_choice]] + "$"

# ----- #
# Plots #
# ----- #

# ->-----------------------<- #
# Diabatic error vs. variable #
# ->-----------------------<- #

# Set up figures and axes
fig = pl.figure(figsize=(figsize_length, figsize_height))
ax = fig.add_subplot()

# Arrays which will store plots
list_pl_avg = []
list_pl_log = []
    
# Loop over data sets
for ind_data in range(0, nbr_datasets):
    
    # Dictionaries specific to data set
    dict_specifications = {key:dict_specifications_all[key][ind_data] for key in dict_specifications_all.keys()}
    dict_protocol = {key:dict_protocol_all[key][ind_data] for key in dict_protocol_all.keys() if key not in keys_exception}
    dict_protocol.update({"dt":0, "Nthresh":0, "Nfix":0, "nbr_realizations":0})
    dict_disorder = {key:dict_disorder_all[key][ind_data] for key in dict_disorder_all.keys()}
    dict_noise = {key:dict_noise_all[key][ind_data] for key in dict_noise_all.keys()}
    dict_noise.update({"tau_noise":0, "dt_noise":0})
    dict_other = {"NSamp":0, "str_desig":"", "str_desig2":""}
    
    # x-data
    x_data = list_df_data_set[ind_data][vary_column]
    
    # y-data
    # -> Append appropriate text to "dataType" for averaged data
    if dict_specifications_all["noise"][ind_data] not in list_noise_noAvg or dict_specifications_all["disorder"][ind_data] not in ["NONE"]:
        y_data = list_df_data_set[ind_data][dataType + "_avg"]
        y_data_log = list_df_data_set[ind_data][dataType + "_logAvg"]
    else:
        y_data = list_df_data_set[ind_data][dataType]
        
    # If plotting against disorder strength and variance/standard deviation is requested, adjust x-data accordingly
    if vary_param_choice == "DR" and True in [flag_variance, flag_std]:
        disorder_type = dict_specifications_all["disorder"][ind_data]
        chempot_change = np.abs(dict_specifications_all["muRightStart"] - dict_specifications_all["muRightEnd"])[ind_data]
        if "ND" in disorder_type:
            if flag_variance == True:
                x_data = (list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)**2
            if flag_std == True: 
                x_data = (list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)
        if "UN" in disorder_type:
            if flag_variance == True:
                x_data = (2*list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)**2/12.0
            if flag_std == True:
                x_data = (2*list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)/np.sqrt(12.0)
                
    # Rescale x-data
    if dict_param_choice[vary_param_choice] in dict_rescale_values.keys():
        x_data = x_data/dict_rescale_values[dict_param_choice[vary_param_choice]][ind_data]
    
    # Data equalization
    x_data_og = np.copy(x_data)
    if flag_equalization == True:
        x_data, y_data = pk.data_equalizer(np.array(x_data_og), np.array(y_data), x_data_ref)
        y_data_log = pk.data_equalizer(np.array(x_data_og), np.array(y_data_log), x_data_ref)[1]
        
    # Plot labels
    if nbr_datasets == 1:
        #label_plot = ", ".join(list_legend[ind_data])
        label_plot = r"$1/f$"
        if flag_expr_diabVx_minGapAct == True:
            label_plot = "Numerical"
    else:
        label_plot = list_legend[ind_data][0]
        #label_plot = r"$1/f$ noise"
        
    # Off-set the marker index if clean data is plotted
    if flag_clean == True:
        index_marker = ind_data + 1 
    else:
        index_marker = ind_data
    
    # Plot average
    #pl_avg, = ax.plot(x_data, y_data, "-" + list_markers[index_marker], markersize=size_marker, linewidth=size_linewidth, color='C{}'.format(ind_data), label=", ".join(list_legend[ind_data]))
    pl_avg, = ax.plot(x_data, y_data, "-" + list_markers[index_marker], markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data], label=label_plot, zorder=99-ind_data, alpha=alpha)
    #ax.plot(x_data, y_data/noise_psd_ampl[ind_data], "-" + list_markers[index_marker], markersize=size_marker, linewidth=size_linewidth, color='C{}'.format(ind_data), label=", ".join(list_legend[ind_data]))
    list_pl_avg.append(pl_avg)
    
    # Plot geometric average
    if flag_hideLogErr == False:
        #pl_log, = ax.plot(x_data, np.exp(y_data_log), "--" + list_markers[index_marker], markersize=size_marker, linewidth=size_linewidth, color='C{}'.format(ind_data), mfc="white", label="")
        pl_log, = ax.plot(x_data, np.exp(y_data_log), "--" + list_markers[index_marker], markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data], mfc="white", label="", zorder=ind_data, alpha=1.0)
        list_pl_log.append(pl_log)
        
    # Plot analytical expression for diabatic error vs. tau
    if flag_expr_diabVtau == True:
        
        # Variables
        delta_mu_tp = 0.5*np.abs(dict_protocol["muRightStart"] - dict_protocol["muRightEnd"])
        Delta_tp = dict_protocol["Delta"]
        lp_tp = dict_protocol["lp"]
        Delta_LZ_tp = np.pi*Delta_tp/lp_tp/2.0
        
        x_data_expr = list_df_data_set[ind_data][vary_column].to_numpy()
        y_data_expr = pk.exprAnaly_diabErr_smooth(delta_mu_tp, Delta_LZ_tp, x_data_expr)
        
        ax.plot(x_data_expr, y_data_expr, "--", markersize=size_marker, linewidth=size_linewidth, color="Black", label="", alpha=1.0)
        
    # Plot semi-analytical expression for diabatic error using actual values of minimum gap
    # -> Note that this function is only appropriate when nbr_datasets = 1 and flag_clean = False
    if flag_expr_diabVx_minGapAct == True: 
        
        # x-data
        x_data_minGapAct = list_df_data_set[ind_data][vary_column].to_numpy()
        
        # y-data
        y_data_minGapAct_avg, y_data_minGapAct_log = pk.expr_diabErr_minGapAct(dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, vary_param_choice, dict_param_choice, nbr_realizations_min, x_data_minGapAct, dirname_mingap_managed, mode="static")
        
        # Plot
        pl_minGapAct_avg, = ax.plot(x_data_og, y_data_minGapAct_avg, "-", markersize=size_marker, linewidth=size_linewidth, color=color_black, label="Semi-analytical", alpha=alpha, zorder=100)
        pl_minGapAct_log, = ax.plot(x_data_og, np.exp(y_data_minGapAct_log), "--", markersize=size_marker, linewidth=size_linewidth, color=color_black, label="", alpha=alpha, zorder=100)
        list_pl_avg.append(pl_minGapAct_avg)
        list_pl_log.append(pl_minGapAct_log)

# Plot clean data 
if flag_clean == True:
    
    # x- and y-data
    x_data_clean = list_df_data_set[-1][vary_column]
    y_data_clean = list_df_data_set[-1][dataType_compare]
    
    # Rescale x-data
    # -> Rescaling is based on the first element of corresponding value in dict_rescale_values
    if dict_param_choice[vary_param_choice] in dict_rescale_values.keys():
        x_data_clean = x_data_clean/dict_rescale_values[dict_param_choice[vary_param_choice]][0]
        
    # Data equalization
    if flag_equalization == True:
        x_data_clean, y_data_clean = pk.data_equalizer(np.array(x_data_clean), np.array(y_data_clean), x_data_ref)
    
    # Plot
    #label_clean = tuning_compare + ", " + disorder_compare + ", " + noise_compare
    label_clean = "Clean"
    ax.plot(x_data_clean, y_data_clean, "-" + list_markers[0], markersize=size_marker, linewidth=size_linewidth, color=color_black, label=label_clean, zorder=100, alpha=alpha)
        
# Plot a line    
#ax.plot(x_data, powerlaw(x_data, 50.0, 1.0), "--", linewidth=2.5, color="Black", alpha=0.90, label=r"$\sim A_{\mathrm{psd}}$")
    
# For vs. frequency results, plot relevant frequencies
if flag_relfreqs == True: 
    
    # Plot frequencies corresponding to min/max bulk gap and min/max bandwidth
    """
    ax.axvline(x=spec_gap_min, linewidth=size_linewidth, linestyle="--", color=color_black)
    ax.axvline(x=spec_gap_max, linewidth=size_linewidth, linestyle="--", color=color_black)
    #ax.axvline(x=spec_bandwidth_max, linewidth=1.25*size_linewidth, linestyle="--", color="Black")
    """
    ax.axvline(x=1.0, linewidth=0.75*size_linewidth, linestyle="--", color="Black", zorder=100)
    ax.axvline(x=spec_gap_max/mingap_clean[0], linewidth=0.75*size_linewidth, linestyle="--", color="Black", zorder=100)
    #ax.axvline(x=spec_gap_max, linewidth=size_linewidth, linestyle="--", color=color_black)
    #ax.axvline(x=spec_bandwidth_max, linewidth=1.25*size_linewidth, linestyle="--", color="Black")
    
    #ax.axvline(x=tau_gap_min/tau[0], linestyle="--", color="Black")
    #ax.axvline(x=tau_gap_max/tau[0], linestyle="--", color="Black")
    
    ax.axvline(x=w[0]/mingap_clean[0], linewidth=0.75*size_linewidth, linestyle="--", color="Black", zorder=100)

    # Text boxes
    #ax.text(0.05, 0.95, str_textbox, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    
# Plot data which does not depend on vary_param_choice, i.e. a horizontal line
# -> User specified
if flag_horizontal == True:
    y_data_horizontal = 3.76E-07
    ax.axhline(y=y_data_horizontal, linestyle="--", markersize=size_marker, linewidth=size_linewidth, color="Black", label="SMOOTH, NONE, NONE", alpha=1.0)
    
# Load x/y data manually and plot
if flag_dataManual == True:
    with np.load("{}/{}.npz".format(dirname_dataManual, filename_dataManual)) as dataManual:
        x_data_manual = dataManual["x"]
        y_data_manual = dataManual["y"]
    label_compare = "Single mode"
    pl_compare, = ax.plot(x_data_manual, y_data_manual, "-", markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data+1], label=label_compare, zorder=99-ind_data-1, alpha=alpha)
    list_pl_avg.append(pl_avg)

# Title
if flag_titleOff == False:
    ax.set_title("Kitaev chain - Piano key" + "\n" + str_title, fontsize=size_title, wrap=True, pad=15)
    
# Axis labels
ax.set_xlabel(xlabel_plot, fontsize=size_axislabel_x)
#ax.set_ylabel(ylabel_plot, fontsize=size_axislabel_y, labelpad=8)
ax.set_ylabel("Diabatic error", fontsize=size_axislabel_y, labelpad=8)
#ax.set_ylabel(diabErr_column + r"/$A_{psd}$", fontsize=size_axislabel_y, labelpad=8)

# Axis scales
ax.set_xscale(scale_x)
ax.set_yscale(scale_y)

# Axis ranges
if flag_useAxisLims == True:
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

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
    #ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=500))
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs=(.2, .3, .4, .5, .6, .7, .8, .9)))
    # Remove tick labels on y-axis which are odd powers
    fig.canvas.draw()
    xticks_major = ax.xaxis.get_major_ticks()
    
    for tick in xticks_major:
        if np.log10(tick.get_loc())%2 != 0.0:
            tick.label1.set_visible(False)
    
    
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

# Spines
pl.setp(ax.spines.values(), linewidth=spine_linewidth)

# Legend
if flag_legend == True:
     
    # Handles and labels
    leg_handles, leg_labels = pl.gca().get_legend_handles_labels()
    
    # Reorganize handles/labels so that clean results come first
    if flag_clean == True:
        leg_order = np.concatenate(([-1], np.arange(0, len(leg_handles)-1)))
        leg_handles = [leg_handles[i] for i in leg_order]
        leg_labels = [leg_labels[i] for i in leg_order]
        
    # Place average and geometric average labels on same line
    if flag_hideLogErr == False:
        leg_handles_act = []
        if flag_clean == True:
            leg_handles_act.append(leg_handles[0])
            for i in range(0, len(leg_handles)-1):
                leg_handles_act.append((leg_handles[i+1], list_pl_log[i]))
        else:
            for i in range(0, len(leg_handles)):
                leg_handles_act.append((leg_handles[i], list_pl_log[i]))
    else:
        leg_handles_act = leg_handles
    
    # Additional legend items
    if flag_expr_diabVx_minGapAct == True:
        ax.plot([],[], "--", markersize=size_marker, linewidth=0.85*size_linewidth, color="Black", alpha=alpha, label="Semi-analytical")
        
    if flag_leglabels_manual == True:
        leg_labels = leg_labels_manual
        
    # Activate legend
    ax.legend(leg_handles_act, leg_labels, loc=loc_legend, fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=size_leg_handle).set_zorder(201)
    
    #ax.legend(loc="lower right", fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=True)
    #ax.legend(leg_handles, leg_labels, loc=loc_legend, fontsize=size_legend, framealpha=0.65, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=True)
    
# Other options
#pl.tight_layout()
pl.subplots_adjust(left=0.15, bottom=0.175)

# Save figure
if flag_saveFigs == True:
    pl.savefig("{}/diabErr_uncorr_b.pdf".format(dirname_plots), format='pdf',bbox_inches=option_bboxinches)

pl.show()
