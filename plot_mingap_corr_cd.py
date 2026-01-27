"""
Plot for minimum gap statistics: correlated disorder 
Minimum gap average and standard deviation vs. correlation length
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

# ------------------------ #
# Operation specifications #
# ------------------------ #

# Data type to load and plot
# -> mingap
dataType = "mingap"

# Display options
flag_latexFonts = True # Enable latex fonts
flag_titleOff = True # Remove title 
flag_variance = False # Convert references to the disorder ratio into the variance. Incompatible with flag_std
flag_std = True # Convert references to the disorder ratio into the standard deviation. Incompatible with flag_variance
flag_mingap_std = True # Convert references to the minimum gap variance into the standard deviation
flag_equalization = False # Enables ability to equally space data points according to a reference array (x_data_ref)

# Loading options
flag_fileManual = False # Load main data file manually

# Display additional plotting elements
flag_horizontal = False # Plot horizontal data
flag_errorbars = False # Plot error bars representing the standard deviation

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

# Primary parameter to plot minimum gap against
# -> DR: Disorder ratio
# -> LC: Correlation length
# -> NR: Noise ratio
# -> NA: Noise PSD amplitude
# -> WCL: Low frequency cutoff 
# -> WCH: High frequency cutoff
# -> W: Single mode frequency
# -> PH: Single mode phase
vary_param_choice = "LC"

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
L = np.array([30, 30, 30])
R = np.array([30, 40, 50])
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
disorder_ratio = np.array([0.1875]) # Disorder strength (default: in units of the chemical potential difference)
length_corr = np.array([6.0]) # (GCND, SCND) Correlation length, (ECND) decay length

# Noise
noise_ratio = np.array([0.0]) # Noise strength in units of the chemical potential difference
noise_psd_ampl = np.array([0.0]) # Amplitude of the power spectral density (note: A = 1e-7 corresponds roughly to a variance of 1e-5)
w_cutoff_low = np.array([1e-8]) # Low frequency cutoff
w_cutoff_high = np.array([0.01]) # High frequency cutoff

# Noise: single mode
# -> noise_ratio will be used to determine the amplitude
freq_sin = np.array([0.03]) # Frequency
phase_sin = np.array([0.0]) # Phase

# Momenta specification for analytical expressions
k1 = np.pi + np.pi/lp # Momentum of single-particle excited state corresponding to energy epsilon_{1}
k_quant = np.pi/lp # Momentum quantization

# -------------------------- #
# Directories and file names #
# -------------------------- #

# Directories
dirname = "data_mingap/averaged" # Directory of averaged minimum gap data
dirname_plots = "plots" # Directory to store plots

"""
# Clean data general specifications
tuning_compare = "SMOOTH"
disorder_compare = "NONE"
noise_compare = "NONE"
dataType_compare = "mingap"
"""

# Manual loading
filename_manual = "TRI_NONE"

# ------------------ #
# Rescale parameters #
# ------------------ #

# Legend of useful keys and symbols
# length_corr, disorder_ratio
# l_{{\mathrm{{p}}}}, \Delta_{{\mathrm{{m}},0}}

# Useful rescale variables
delta_mu = 0.5*np.abs(muRightEnd - muRightStart)
tau_0 = pk.tau_LZ(w, muRightStart, lp, Delta)
Delta_m = np.pi*Delta/(lp)
hbar = (6.582e-16)*1e3 # hbar, in units of meV s
#tau_rescale = 1/hbar/1e9

# Rescale variables and symbols
# -> Note that for dict_rescale_values, each value must have the same size as nbr_datasets
nbr_datasets_rescale = 3
dict_rescale_values = {}
dict_rescale_symbols = {}

dict_rescale_values = {"tau":np.full(nbr_datasets_rescale, tau_0), "disorder_ratio":np.full(nbr_datasets_rescale, w), "length_corr":np.full(nbr_datasets_rescale, lp)}
dict_rescale_symbols = {"tau":r"\tau_{{0}}", "disorder_ratio":"w", "length_corr":"R"}

# ---------------------------------- #
# Rescale and/or add shift to y-data #
# -----------------------------------#

# Legend of useful keys and symbols
# l_{{\mathrm{{p}}}}, \Delta_{{\mathrm{{m}},0}}

# -> Average
yaxis_shift_values_avg = -2.0*np.array([0.031644, 0.023750, 0.018984, 0.015804])
#yaxis_shift_symbol_avg = r"-\Delta_{{\mathrm{{m}}},0}"
yaxis_shift_symbol_avg = r"-\Delta_{R}"
yaxis_rescale_values_avg = 1/np.full(nbr_datasets_rescale, w)
yaxis_rescale_symbol_avg = r"/ w"

# -> Standard deviation
yaxis_shift_values_std = np.array([])
yaxis_shift_symbol_std = ""
yaxis_rescale_values_std = np.sqrt(lp)/np.full(nbr_datasets_rescale, w)
yaxis_rescale_symbol_std = r"(\sqrt{R} / w)"


# ------------ #
# Data display #
# ------------ #

# Data equalization
x_data_ref = np.linspace(0.0, 1.0, 26)

# --------------- #
# Plot parameters #
# --------------- #

# Figure sizes
figsize_length = 8.5
figsize_height = 4.5

# Legend
flag_legend = True
size_legend = 18
loc_legend_avg = (0.65,0.04)
loc_legend_var = (0.15,0.04)

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
# -> Average
x_min_avg = -0.05
x_max_avg = 1.05
y_min_avg = -0.00425
y_max_avg = 0.0005
# -> Standard deviation
x_min_std = -0.05
x_max_std = 1.05
y_min_std = 0.015
y_max_std = 0.05

# Spine linewidth
spine_linewidth = 1.75

# Other
option_bboxinches = "tight" # Can use "tight" or 0

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
            if str_main_core + "." not in file:
                continue
            else:
                list_correspondfiles.append(file)
                
        latestfile = list_correspondfiles[0]
        
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

# -------------------- #
# Extract data to plot #
# -------------------- #

# List of data frames for each data set
list_df_data_set = []
    
# Loop over number of data sets
for ind_data in range(0, nbr_datasets):
    
    # Extract data frame
    ind_data_act = ind_data
    df_data_main = dict_data_files[dict_specifications_all["tuning"][ind_data_act] + "_" + dict_specifications_all["disorder"][ind_data_act] + "_" + dict_specifications_all["noise"][ind_data_act]]
    df_main = df_data_main
    
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
        
    # Establish parameter columns and set index of df_main to parameter columns
    cols_nodata = pk.create_csvcolumns_nodata(dict_specifications)
    cols_nodata.remove("tau")
    #cols_nodata.append("{}_count".format(dataType))
    cols_nodata.remove(dict_param_choice[vary_param_choice])
    df_main.set_index(cols_nodata, inplace=True)
    
    # Establish the multi-index for parameters with varied parameter accounted for
    multiIndex_pars = pk.create_csvmultiIndex(dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise)
    multiIndex_pars = multiIndex_pars.droplevel([dict_param_choice["TAU"]])
    multiIndex_pars = multiIndex_pars.droplevel([dict_param_choice[vary_param_choice]])
    
    # Find all data corresponding to multi-index with varied parameter accounted for; reset index
    df_data_set = df_main.loc[multiIndex_pars]
    df_data_set.reset_index(inplace=True)
    df_main.reset_index(inplace=True)
    
    # In the case of disorder or noise, keep only the rows of data where the number of realizations = nbr_realizations_min
    if dict_specifications["noise"] not in list_noise_noAvg or dict_specifications["disorder"] not in ["NONE"]:
        # Identify columns which have "_count" in their name
        #list_str_count = [str_count for str_count in list(df_data_set.columns) if "_count" in str_count]
        # Consider data only corresponding to dataType
        list_str_count = [dataType + '_count']
        for str_count in list_str_count:  
            df_data_set = df_data_set.drop(df_data_set[df_data_set[str_count] != nbr_realizations_min].index)
            
    # Sort the data according to the parameter choice
    df_data_set = df_data_set.sort_values(by=[dict_param_choice[vary_param_choice]])
    
    # Append to list
    list_df_data_set.append(df_data_set)
    
# --------------------------- #
# General plotting parameters #
# --------------------------- #

# Color palettes 
color_black = "#4d4d4d"
#list_colors = ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988", "#BBBBBB"]
#list_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
#list_colors = ['#6699CC', '#004488', '#EECC66', '#994455', '#997700', '#EE99AA']
#list_colors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499']
list_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

# Marker styles
#list_markers = ['o','v', '^', '<', '>', 's', 'p', 'P', '4', '8', '1', '2', '3', '*', 'h', 'H','+','x','X','D','d','|']
list_markers = ['o','v', "s", "D", "*", "h"]

# Generate title and legend text strings
str_title, list_legend = pk.str_plot(nbr_datasets, dict_specifications_all, dict_protocol_all, dict_disorder_all, dict_noise_all, vary_param_choice,  dict_rescale_values=dict_rescale_values, dict_rescale_symbols=dict_rescale_symbols, flag_disorder_replace_var=flag_variance, flag_disorder_replace_std=flag_std)
str_title = str_title.replace(" $\\tau = 100.00$,", "")

# Labels
dict_xlabels = {"TAU":r"$\tau$", "STIMES":r"$\tau_{\mathrm{i}} / \tau$", "STIMEM":r"$\tau_{\mathrm{m}} / \tau$", "SLOPE":r"Slope $\alpha$", "WCL":r"$\omega_{{\mathrm{l}}}$", "WCH":r"$\omega_{{\mathrm{h}}}$", "NR":r"$r_{{\mathrm{n}}}$", "NA":r"$A_{{\mathrm{psd}}}$", "W":r"$\omega$", "PH":r"$\phi$", "DR":r"$r_{{\mathrm{{d}}}}$", "LC":r"$\xi$"}
dict_ylabels = {"mingap":r"$\Delta_{{\mathrm{{m}}}}$"}
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

# ->--------------------<- #
# Minimum gap vs. variable #
# ->--------------------<- #

# Set up figures and axes
fig = pl.figure(figsize=(figsize_length, figsize_height))
ax = fig.add_subplot()
  
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
    
    # Temporary parameters
    L_temp = dict_protocol_all["L"][ind_data]
    R_temp = dict_protocol_all["R"][ind_data]
    lp_temp = dict_protocol_all["lp"][ind_data]
    muLeft_temp = dict_protocol_all["muLeft"][ind_data]
    Delta_temp = dict_protocol_all["Delta"][ind_data]
    w_temp = dict_protocol_all["w"][ind_data]
    disorder_type = dict_specifications_all["disorder"][ind_data]
    sigma_temp = np.sqrt(disorder_variance[ind_data])
    xi_temp = dict_disorder_all["length_corr"][ind_data]
    
    # Set the varying parameter
    if vary_param_choice == "DR":
        chempot_change_temp = np.abs(dict_protocol_all["muRightStart"] - dict_protocol_all["muRightEnd"])[ind_data]
        if "ND" in disorder_type:
            disorder_variance[ind_data] = (dict_disorder_all["disorder_ratio"][ind_data]*chempot_change)**2
            sigma_temp = np.sqrt((list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)**2)
        if "UN" in disorder_type:
            disorder_variance[ind_data] = (2.0*dict_disorder_all["disorder_ratio"][ind_data]*chempot_change)**2/12.0
            sigma_temp = np.sqrt((2*list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)**2/12.0)
    if vary_param_choice == "LC":
        xi_temp = list_df_data_set[ind_data][vary_column].to_numpy()
    
    # x-data
    x_data = list_df_data_set[ind_data][vary_column]
    
    # y-data
    # -> Note that the true minimum gap is actually twice the amount of the data. See notes. 
    # -> Append appropriate text to "dataType" for averaged data
    if dict_specifications_all["noise"][ind_data] not in list_noise_noAvg or dict_specifications_all["disorder"][ind_data] not in ["NONE"]:
        y_data = 2.0*list_df_data_set[ind_data][dataType + "_avg"]
        y_data_sqr = 4.0*list_df_data_set[ind_data][dataType + "_avg" + "_sqr"]
        ylabel_plot = r"$\langle${}$\rangle$".format(dict_ylabels[dataType])
    else:
        y_data = 2.0*list_df_data_set[ind_data][dataType]
        y_data_sqr = 4.0*list_df_data_set[ind_data][dataType + "_sqr"]
        ylabel_plot = dict_ylabels[dataType]

    """
    # For the purposes of finding missing data
    ind_zeros = np.where(y_data.to_numpy() == 0.0)[0]
    for ind in ind_zeros:
        print("(",x_data.to_numpy()[ind],",",dict_disorder_all["length_corr"][ind_data],")")
    """
    
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
    if flag_equalization == True:
        x_data, y_data = pk.data_equalizer(np.array(x_data), np.array(y_data), x_data_ref)
        
    # Shift and/or rescale y-data
    if yaxis_shift_values_avg.shape[0] != 0:
        y_data = y_data + yaxis_shift_values_avg[ind_data] 
        ylabel_plot = ylabel_plot + r"${}$".format(yaxis_shift_symbol_avg)
        if yaxis_rescale_values_avg.shape[0] != 0:
            ylabel_plot = "(" + ylabel_plot + ")"
    if yaxis_rescale_values_avg.shape[0] != 0:
        y_data = y_data*yaxis_rescale_values_avg[ind_data]  
        if yaxis_shift_values_avg.shape[0] != 0:
            ylabel_plot = ylabel_plot + r"${}$".format(yaxis_rescale_symbol_avg)
        else:
            ylabel_plot = r"${}$".format(yaxis_rescale_symbol_avg) + ylabel_plot
        
    # Plot labels
    if nbr_datasets == 1:
        label_plot = ", ".join(list_legend[ind_data])
    else:
        label_plot = list_legend[ind_data][0]
        
    # Plot
    #ax.plot(x_data, y_data, "-" + list_markers[ind_data], markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data], label=", ".join(list_legend[ind_data]))
    ax.plot(x_data, y_data, "-" + list_markers[ind_data], markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data], label=label_plot, alpha=alpha)
    #ax.plot(x_data, y_data/noise_psd_ampl[ind_data], "-" + list_markers[ind_data], markersize=size_marker, linewidth=size_linewidth, color='C{}'.format(ind_data), label=", ".join(list_legend[ind_data]))
   
# Plot a horizontal line through y = =
ax.axhline(y=0, linestyle="--", markersize=size_marker, linewidth=size_linewidth, color="Black")
    
# Plot data which does not depend on vary_param_choice, i.e. a horizontal line
# -> User specified
if flag_horizontal == True:
    y_data_horizontal = 3.76E-07
    ax.axhline(y=y_data_horizontal, linestyle="--", markersize=size_marker, linewidth=size_linewidth, color="Black", label="SMOOTH, NONE, NONE", alpha=1.0)

# Titles and labels
if flag_titleOff == False:
    ax.set_title("Kitaev chain - Piano key" + "\n" + str_title, fontsize=size_title, wrap=True, pad=15)
    
# Axis labels
ax.set_xlabel(xlabel_plot, fontsize=size_axislabel_x)
ax.set_ylabel(ylabel_plot, fontsize=size_axislabel_y, labelpad=8)
#ax.set_ylabel(diabErr_column + r"/$A_{psd}$", fontsize=size_axislabel_y, labelpad=8)
#ax.set_ylabel(r"$\langle \Delta_{\mathrm{m}} \rangle / \Delta_{\mathrm{m},0} - 1$")

# Axis scales
ax.set_xscale(scale_x)
ax.set_yscale(scale_y)

# Axis ranges
if flag_useAxisLims == True:
    ax.set_xlim([x_min_avg, x_max_avg])
    ax.set_ylim([y_min_avg, y_max_avg])

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
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
ax.yaxis.offsetText.set_fontsize(size_ticklabel_y)

# Spines
pl.setp(ax.spines.values(), linewidth=spine_linewidth)

# Legend
if flag_legend == True:
    #ax.legend(loc="lower right", fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=True)
    ax.legend(loc=loc_legend_avg, fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=False).set_zorder(201)
    
# Other options
#pl.tight_layout()
pl.subplots_adjust(left=0.25, bottom=0.25)

# Save figure
if flag_saveFigs == True:
    pl.savefig("{}/mingap_corr_xi_avg.pdf".format(dirname_plots), format='pdf',bbox_inches=option_bboxinches)

pl.show()

# ->-----------------------------------------------------<- #
# Variance / standard deviation of minimum gap vs. variable #
# ->-----------------------------------------------------<- #

fig_var = pl.figure(figsize=(figsize_length, figsize_height))
ax_var = fig_var.add_subplot()
    
# Loop over the number of data sets and plot data within each
for ind_data in range(0, nbr_datasets):
    
    # Dictionaries specific to data set
    dict_specifications = {key:dict_specifications_all[key][ind_data] for key in dict_specifications_all.keys()}
    dict_protocol = {key:dict_protocol_all[key][ind_data] for key in dict_protocol_all.keys() if key not in keys_exception}
    dict_protocol.update({"dt":0, "Nthresh":0, "Nfix":0, "nbr_realizations":0})
    dict_disorder = {key:dict_disorder_all[key][ind_data] for key in dict_disorder_all.keys()}
    dict_noise = {key:dict_noise_all[key][ind_data] for key in dict_noise_all.keys()}
    dict_noise.update({"tau_noise":0, "dt_noise":0})
    dict_other = {"NSamp":0, "str_desig":"", "str_desig2":""}
    
    # Temporary parameters
    L_temp = dict_protocol_all["L"][ind_data]
    R_temp = dict_protocol_all["R"][ind_data]
    lp_temp = dict_protocol_all["lp"][ind_data]
    muLeft_temp = dict_protocol_all["muLeft"][ind_data]
    Delta_temp = dict_protocol_all["Delta"][ind_data]
    w_temp = dict_protocol_all["w"][ind_data]
    disorder_type = dict_specifications_all["disorder"][ind_data]
    sigma_temp = np.sqrt(disorder_variance[ind_data])
    xi_temp = dict_disorder_all["length_corr"][ind_data]
    
    # Set the varying parameter
    if vary_param_choice == "DR":
        chempot_change_temp = np.abs(dict_protocol_all["muRightStart"] - dict_protocol_all["muRightEnd"])[ind_data]
        if "ND" in disorder_type:
            disorder_variance[ind_data] = (dict_disorder_all["disorder_ratio"][ind_data]*chempot_change)**2
            sigma_temp = np.sqrt((list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)**2)
        if "UN" in disorder_type:
            disorder_variance[ind_data] = (2.0*dict_disorder_all["disorder_ratio"][ind_data]*chempot_change)**2/12.0
            sigma_temp = np.sqrt((2*list_df_data_set[ind_data][vary_column].to_numpy()*chempot_change)**2/12.0)
    if vary_param_choice == "LC":
        xi_temp = list_df_data_set[ind_data][vary_column].to_numpy()
    
    # x-data
    x_data = list_df_data_set[ind_data][vary_column]
    
    # y-data
    # -> Note that the true minimum gap is actually twice the amount of the data. See notes. 
    # -> Append appropriate text to "dataType" for averaged data
    if dict_specifications_all["noise"][ind_data] not in list_noise_noAvg or dict_specifications_all["disorder"][ind_data] not in ["NONE"]:
        y_data_avg = 2.0*list_df_data_set[ind_data][dataType + "_avg"]
        y_data_sqr = 4.0*list_df_data_set[ind_data][dataType + "_avg" + "_sqr"]
        y_data = y_data_sqr - y_data_avg**2
        ylabel_plot = r"$\langle${}$^2$$\rangle - \langle${}$\rangle^2$".format(dict_ylabels[dataType], dict_ylabels[dataType])
        if flag_mingap_std == True:
            y_data = np.sqrt(np.abs(y_data))
            ylabel_plot = ylabel_plot.replace("$","")
            ylabel_plot = "\sqrt{{{}}}".format(ylabel_plot)
            ylabel_plot = r"${}$".format(ylabel_plot)
    else:
        y_data_avg = 2.0*list_df_data_set[ind_data][dataType]
        y_data_sqr = 4.0*list_df_data_set[ind_data][dataType + "_sqr"]
        y_data = y_data_sqr - y_data_avg**2
        ylabel_plot = dict_ylabels[dataType]

    """
    # For the purposes of finding missing data
    ind_zeros = np.where(y_data.to_numpy() == 0.0)[0]
    for ind in ind_zeros:
        print("(",x_data.to_numpy()[ind],",",dict_disorder_all["length_corr"][ind_data],")")
    """
    
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
    if flag_equalization == True:
        x_data, y_data = pk.data_equalizer(np.array(x_data), np.array(y_data), x_data_ref)
        
    # Shift and/or rescale y-data
    if yaxis_shift_values_std.shape[0] != 0:
        y_data = y_data + yaxis_shift_values_std[ind_data] 
        ylabel_plot = ylabel_plot + r"${}$".format(yaxis_shift_symbol_std)
        if yaxis_rescale_values_std.shape[0] != 0:
            ylabel_plot = "(" + ylabel_plot + ")"
    if yaxis_rescale_values_std.shape[0] != 0:
        y_data = y_data*yaxis_rescale_values_std[ind_data]  
        if yaxis_shift_values_std.shape[0] != 0:
            ylabel_plot = ylabel_plot + r"${}$".format(yaxis_rescale_symbol_std)
        else:
            ylabel_plot = r"${}$".format(yaxis_rescale_symbol_std) + ylabel_plot
        
    # Plot labels
    if nbr_datasets == 1:
        label_plot = ", ".join(list_legend[ind_data])
    else:
        label_plot = list_legend[ind_data][0]
      
    # Plot
    #ax_var.plot(x_data, y_data, "-" + list_markers[ind_data], markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data], label=", ".join(list_legend[ind_data]))
    ax_var.plot(x_data, y_data, "-" + list_markers[ind_data], markersize=size_marker, linewidth=size_linewidth, color=list_colors[ind_data], label=label_plot, alpha=alpha)
    #ax_var.plot(x_data, y_data/noise_psd_ampl[ind_data], "-" + list_markers[ind_data], markersize=size_marker, linewidth=size_linewidth, color='C{}'.format(ind_data), label=", ".join(list_legend[ind_data]))
    
# Plot data which does not depend on vary_param_choice, i.e. a horizontal line
# -> User specified
if flag_horizontal == True:
    y_data_horizontal = 3.76E-07
    ax_var.axhline(y=y_data_horizontal, linestyle="--", markersize=size_marker, linewidth=size_linewidth, color="Black", label="SMOOTH, NONE, NONE", alpha=1.0)

# Titles and labels
if flag_titleOff == False:
    ax_var.set_title("Kitaev chain - Piano key" + "\n" + str_title, fontsize=size_title, wrap=True, pad=15)

# Axis labels
ax_var.set_xlabel(xlabel_plot, fontsize=size_axislabel_x)
ax_var.set_ylabel(ylabel_plot, fontsize=size_axislabel_y, labelpad=8)
ylabel_manual = '$\\sqrt{R} \\sqrt{\\langle\\Delta_{{\\mathrm{{m}}}}^2\\rangle - \\langle\\Delta_{{\\mathrm{{m}}}}\\rangle^2} / w$'
#ylabel_manual = '$\\sqrt{R/\\xi} \\sqrt{\\langle\\Delta_{{\\mathrm{{m}}}}^2\\rangle - \\langle\\Delta_{{\\mathrm{{m}}}}\\rangle^2} / w$'
ax_var.set_ylabel(ylabel_manual, fontsize=size_axislabel_y, labelpad=8)


# Axis scales
ax_var.set_xscale(scale_x)
ax_var.set_yscale(scale_y)

# Axis ranges
if flag_useAxisLims == True:
    ax_var.set_xlim([x_min_std, x_max_std])
    ax_var.set_ylim([y_min_std, y_max_std])

# Ticks and labels
# -> Locators for major ticks
#ax_var.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
#ax_var.xaxis.set_major_locator(mpl.ticker.MultipleLocator(locate_majorticks))
#ax_var.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=500))
#ax_var.yaxis.set_major_locator(mpl.ticker.FixedLocator(locs = np.logspace(-9, 0, 10)))

# -> Locators for major and minor ticks
if scale_x == "linear":
    ax_var.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    
if scale_x == "log":
    ax_var.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=999))
    
if scale_y == "linear":
    ax_var.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

if scale_y == "log":
    ax_var.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=500))
    ax_var.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs=(.2, .3, .4, .5, .6, .7, .8, .9)))
    # Remove tick labels on y-axis which are odd powers
    fig_var.canvas.draw()
    yticks_major = ax_var.yaxis.get_major_ticks()
    for tick in yticks_major:
        if np.log10(tick.get_loc())%2 != 0.0:
            tick.label1.set_visible(False)

# Tick sizes
ax_var.xaxis.set_tick_params(which="major", direction="in", length=size_tickmajor_x, width=1.0, labelsize=size_ticklabel_x)
ax_var.xaxis.set_tick_params(which="minor", direction="in", length=size_tickminor_x)
ax_var.yaxis.set_tick_params(which="major", direction="in", length=size_tickmajor_y, width=1.0, labelsize=size_ticklabel_y)
ax_var.yaxis.set_tick_params(which="minor", direction="in", length=size_tickminor_y)

# Spines
pl.setp(ax_var.spines.values(), linewidth=spine_linewidth)

# Legend
if flag_legend == True:
    #ax_var.legend(loc="lower right", fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=True)
    ax_var.legend(loc=loc_legend_var, fontsize=size_legend, framealpha=0.75, edgecolor="Black", fancybox=False, markerscale=1.0, borderaxespad=1.0, ncol=1, frameon=False).set_zorder(201)
    
# Other options
#pl.tight_layout()
pl.subplots_adjust(left=0.15, bottom=0.15)

# Save figure
if flag_saveFigs == True:
    pl.savefig("{}/mingap_corr_xi_std.pdf".format(dirname_plots), format='pdf',bbox_inches=option_bboxinches)

pl.show()
    