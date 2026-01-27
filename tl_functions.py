"""
A suite of functions related to two-level systems used to simply model the transport simulation dynamics
"""
# ------- #
# Modules #
# ------- #

import argparse as ap
#import matplotlib.pyplot as pl
#import mpmath as mp
import numpy as np
import os as os
import pandas as pd
#import scipy as sc
import scipy.fft as sc_fft
import scipy.linalg as sc_lin
#import scipy.special as sc_spec
#import scipy.signal as sc_sig

# -------- #
# Overhead #
# -------- #

# Argument parser, allows for optional command line arguments of parameters
# Returns all parameters updated after command line arguments have been provided
def argument_parser(dict_operations, dict_operations_fullsim, dict_operations_onesim, dict_specifications, dict_protocol, dict_directories, dict_other):
    
    # Command line arguments and parameters
    parser = ap.ArgumentParser(description="Specifications and parameters for TLS protocols")
    # -> Operations specifications
    parser.add_argument('--op_clus', type=str, dest='op_cluster')
    parser.add_argument('--op_osim', type=str, dest='op_onesim')
    parser.add_argument('--op_avg', type=str, dest='op_average')
    parser.add_argument('--op_save', type=str, dest='op_savedata')
    parser.add_argument('--op_nfix', type=str, dest='op_Nfix')
    parser.add_argument('--op_tord', type=str, dest='op_timeoverride')
    # -> Operation specifications, fullsim
    parser.add_argument('--op_f_diab', type=str, dest='op_f_diabErr')
    parser.add_argument('--op_f_diab_cc', type=str, dest='op_f_diabErr_cc')
    parser.add_argument('--op_f_diab_ct', type=str, dest='op_f_diabErr_ct')
    # -> Operation specifications, onesim
    parser.add_argument('--op_s_diab', type=str, dest='op_s_diabErr')
    parser.add_argument('--op_s_diab_cc', type=str, dest='op_s_diabErr_cc')
    parser.add_argument('--op_s_diab_ct', type=str, dest='op_s_diabErr_ct')
    parser.add_argument('--op_s_exen', type=str, dest='op_s_exptEnergy')
    parser.add_argument('--op_s_mate', type=str, dest='op_s_matElement')
    # -> Protocol specifications
    parser.add_argument('--pr_tune', type=str, dest='pr_tuning')
    parser.add_argument('--pr_nois', type=str, dest='pr_noise')
    # -> Protocol parameters
    # ---> System
    parser.add_argument('--pa_dmu', type=float, dest='pa_delta_mu')
    parser.add_argument('--pa_dsc', type=float, dest='pa_Delta_SC')
    parser.add_argument('--pa_lp', type=int, dest='pa_lp')
    # ---> Time
    parser.add_argument('--pa_tau', type=float, dest='pa_tau')
    parser.add_argument('--pa_dt', type=float, dest='pa_dt')
    parser.add_argument('--pa_nthr', type=int, dest='pa_Nthresh')
    parser.add_argument('--pa_nfix', type=int, dest='pa_Nfix')
    # ---> Tuning function
    parser.add_argument('--pa_sti', type=float, dest='pa_s_time_init')
    parser.add_argument('--pa_stm', type=float, dest='pa_s_time_mid')
    parser.add_argument('--pa_spm', type=float, dest='pa_slope_mid')
    parser.add_argument('--pa_scl', type=float, dest='pa_scale_factor')
    # ---> Noise, universal
    parser.add_argument('--pa_nois_r', type=float, dest='pa_noise_ratio')
    parser.add_argument('--pa_nois_R', type=int, dest='pa_nbr_realizations')
    # ---> Noise, 1/f
    parser.add_argument('--pa_nois_A', type=float, dest='pa_noise_psd_ampl')
    parser.add_argument('--pa_nois_tau', type=float, dest='pa_tau_noise')
    parser.add_argument('--pa_nois_dt', type=float, dest='pa_dt_noise')
    parser.add_argument('--pa_nois_wcl', type=float, dest='pa_w_cutoff_low')
    parser.add_argument('--pa_nois_wch', type=float, dest='pa_w_cutoff_high')
    # ---> Noise, single mode
    parser.add_argument('--pa_sin_freq', type=float, dest='pa_freq_sin')
    parser.add_argument('--pa_sin_phase', type=float, dest='pa_phase_sin')
    # ---> Directories
    parser.add_argument('--dr_main', type=str, dest='dr_data')
    parser.add_argument('--dr_avg', type=str, dest='dr_data_average')
    parser.add_argument('--dr_one', type=str, dest='dr_data_onesim')
    parser.add_argument('--dr_oavg', type=str, dest='dr_data_onesim_average')
    # -> Other parameters
    parser.add_argument('--po_nsamp', type=int, dest='po_NSamp')
    parser.add_argument('--po_desig', type=str, dest='po_str_desig')
    parser.add_argument('--po_desig2', type=str, dest='po_str_desig2')
    
    # Collect all arguments and parameters into a "input" dictionary
    dict_args_cmdline = vars(parser.parse_args())
    
    # Adjust the values of dict_args_cmdline so that they are boolean
    for key_bool in dict_args_cmdline.keys():
        if "op_" in key_bool and dict_args_cmdline[key_bool] != None:
            if dict_args_cmdline[key_bool] == 'true':
                dict_args_cmdline[key_bool] = True
            else:
                dict_args_cmdline[key_bool] = False
                
    # Adjust the designation strings in dict_args_cmdline if NONE is given
    for key_strdesig in ["po_str_desig", "po_str_desig2"]:
        if dict_args_cmdline[key_strdesig] == "NONE":
            dict_args_cmdline[key_strdesig] = ""
            
    # Place the dictionaries into a list
    list_dicts = [dict_operations, dict_operations_fullsim, dict_operations_onesim, dict_specifications, dict_protocol, dict_directories, dict_other]
    
    # Loop through each dictionary. Replace the values in the original dictionary with the command line input if provided
    # -> List of dictionary prefixes; must match order of list_dicts
    list_dict_prefix = ["op_", "op_f_", "op_s_", "pr_", "pa_", "dr_", "po_"]
    
    for ind_dict in range(0, len(list_dicts)):
        dict_pars = list_dicts[ind_dict]
        dict_prefix = list_dict_prefix[ind_dict]
        for key in dict_pars.keys():
            if dict_args_cmdline[dict_prefix + key] != None: 
                arg_cmd = dict_args_cmdline[dict_prefix + key]
                try:
                    if arg_cmd != dict_pars[key]:
                        list_dicts[ind_dict][key] = arg_cmd
                except:
                    pass
                
    # Return updated dictionaries
    return list_dicts[0], list_dicts[1], list_dicts[2], list_dicts[3], list_dicts[4], list_dicts[5], list_dicts[6]

# Parameter string creator, for data file names
# -> Input: dictionaries of parameters and specifications
# -> Output: parameter string
def str_params(dict_operations, dict_specifications, dict_protocol, dict_other, flag_NSamp=False):
    
    # Unpack parameters from dictionaries  
    # -> Operations
    flag_cluster = dict_operations["cluster"]
    flag_timeoverride = dict_operations["timeoverride"]
    # -> Specifications
    str_tuning_choice = dict_specifications["tuning"]
    str_noise_choice = dict_specifications["noise"]
    # -> Other
    NSamp = dict_other["NSamp"]
    str_desig = dict_other["str_desig"]
    
    # Dictionary of text strings corresponding to numerical parameter (order is important here!)
    dict_textstr = {"tau":"tI", "delta_mu":"dM", "Delta_SC":"D", "lp":"lp", "s_time_init":"sTI", "s_time_mid":"sTM", "slope_mid":"sM", "scale_factor":"sF", "noise_ratio":"mN", "noise_psd_ampl":"aN", "w_cutoff_low":"wCL", "w_cutoff_high":"wCH", "freq_sin":"w", "phase_sin":"pH"}
    
    # Build text strings corresponding to string/boolean designations
    # -> Assign appropriate string depending on tuning/time evolution type
    if str_tuning_choice == "LIN":
        str_evol = "_LIN"
    elif str_tuning_choice == "SMOOTH":
        str_evol = "_SMT"
    elif str_tuning_choice == "FQD":
        str_evol = "_FQD"
    elif str_tuning_choice == "SHARP":
        str_evol = "_SHP"
    elif str_tuning_choice == "TRI":
        str_evol = "_TRI"
    elif str_tuning_choice == "TRIB":
        str_evol = "_TRIB"
    elif str_tuning_choice == "TRIC":
        str_evol = "_TRIC"
    elif str_tuning_choice == "TRID":
        str_evol = "_TRID"
    elif str_tuning_choice == "TRID2":
        str_evol = "_TRID2"
    elif str_tuning_choice == "TRIE":
        str_evol = "_TRIE"
    elif str_tuning_choice == "SNP":
        str_evol = "_SNP"
    elif str_tuning_choice == "SNPB":
        str_evol = "_SNPB"
    else:
        str_evol = "_SMT"
        
    # -> Assign appropriate string for NSamp if desired:
    if flag_NSamp == True:
        str_NSamp = "_Ns{}".format(NSamp)
    else:
        str_NSamp = ""
        
    # -> Assign appropriate string depending on noise type
    if str_noise_choice == "1F1":
        str_noise = "_1F1"
    elif str_noise_choice == "1F1A":
        str_noise = "_1F1A"
    elif str_noise_choice == "WHT":
        str_noise = "_WHT"
    elif str_noise_choice == "WHTA":
        str_noise = "_WHTA"
    elif str_noise_choice == "WHTC":
        str_noise = "_WHTC"
    elif str_noise_choice == "WHTCB":
        str_noise = "_WHTCB"
    elif str_noise_choice == "SIN":
        str_noise = "_SIN"
    elif str_noise_choice == "SINAV":
        str_noise = "_SINAV"
    else:
        str_noise = "" 
        
    # -> If cluster mode is on, clear str_evol and str_noise
    if flag_cluster == True:
        str_evol = ""
        str_noise = ""
        
    # -> Assign appropriate string designator if time override is active
    if flag_timeoverride == True:
        str_timeoverride = "_T"
    else:
        str_timeoverride = ""
        
    # Build text strings corresponding to numerical protocol parameters
    str_protocol = ""
    # Omit certain keys from dict_textstr depending on the string/boolean designations
    list_omit_textstr_noise = []
    if str_noise_choice == "NONE":
        list_omit_textstr_noise = ["noise_ratio", "noise_psd_ampl", "w_cutoff_low", "w_cutoff_high", "freq_sin", "phase_sin"]
    if str_noise_choice == "WHT" or str_noise_choice == "WHTA":
        list_omit_textstr_noise = ["noise_psd_ampl", "w_cutoff_low", "w_cutoff_high", "freq_sin", "phase_sin"]
    if str_noise_choice == "1F1" or str_noise_choice == "1F1A" or str_noise_choice == "WHTC":
        list_omit_textstr_noise = ["noise_psd_ampl", "freq_sin", "phase_sin"]
    if str_noise_choice in ["1F1B", "WHTCB"]:
        list_omit_textstr_noise = ["noise_ratio", "freq_sin", "phase_sin"]
    if str_noise_choice == "SIN":
        list_omit_textstr_noise = ["noise_psd_ampl", "w_cutoff_low", "w_cutoff_high"]
    if str_noise_choice == "SINAV":
        list_omit_textstr_noise = ["noise_psd_ampl", "w_cutoff_low", "w_cutoff_high", "phase_sin"]
    # -> For the tuning functions
    list_omit_textstr_tuning = []
    if str_tuning_choice in ["LIN", "SMOOTH", "FQD", "SHARP"]:
        list_omit_textstr_tuning = ["s_time_init", "s_time_mid", "slope_mid", "scale_factor"]
    if str_tuning_choice in ["TRIB", "TRIC", "SNP", "SNPB"]:
        list_omit_textstr_tuning = ["slope_mid", "scale_factor"]
    if str_tuning_choice in ["TRIE"]:
        list_omit_textstr_tuning = ["slope_mid"]
    # Combine the exemption lists
    list_omit_textstr = list_omit_textstr_tuning + list_omit_textstr_noise
        
    # Build text strings
    for key in list(dict_textstr.keys()):
        if key in list_omit_textstr:
            continue
        value_par = dict_protocol[key]
        str_value_par = str(value_par).replace('.', '')
        str_par = "_" + dict_textstr[key] + str_value_par
        str_protocol = str_protocol + str_par
        
    # Complete text string of parameters and designations
    str_params = str_desig + str_evol + str_timeoverride + str_noise + str_NSamp + str_protocol

    return str_params

# ------ #
# Common #
# ------ #

# Average of averages
# -> Note that nbr_a and nbr_b must be integers; avg_a and avg_b are allowed to be arrays
def avg_averages(nbr_a, nbr_b, avg_a, avg_b):
    #return (nbr_a*avg_a + nbr_b*avg_b)/(nbr_a + nbr_b)
    return np.divide(nbr_a*avg_a + nbr_b*avg_b, nbr_a + nbr_b)

# --------------- #
# Data management #
# --------------- #

# Adds columns to a data frame, ignores if a column already exists
def df_column_adder(df, cols_add):
    cols_df = list(df.columns)
    cols_new = [col for col in cols_add if col not in cols_df]
    """
    for col in cols_new:
        df = pd.concat([df, pd.DataFrame(columns=[col])])
    """
    for col in cols_new:
        df[col] = np.nan
    return df

# Establish non-data columns of full simulation .csv files
def create_csvcolumns_nodata(dict_specifications, flag_desig=False):
    
    # Establish base-line, non-data columns
    list_str_cols = ["timeoverride", "delta_mu", "Delta_SC", "lp", "tau"]
    
    # -> Add columns depending on tuning function choice
    str_tuning_choice = dict_specifications["tuning"]
    if str_tuning_choice in ["TRI", "TRID", "TRID2"]:
        list_str_cols.extend(["s_time_init", "s_time_mid", "slope_mid"])
    if str_tuning_choice in ["TRIB", "TRIC", "SNP", "SNPB"]:
        list_str_cols.extend(["s_time_init", "s_time_mid"])
    if str_tuning_choice in ["TRIE"]:
        list_str_cols.extend(["s_time_init", "s_time_mid", "scale_factor"])
    # -> Add columns depending on noise choice
    str_noise_choice = dict_specifications["noise"]
    if str_noise_choice in ["WHT", "WHTA"]:
        list_str_cols.append("noise_ratio")
    if str_noise_choice in ["1F1", "1F1A", "WHTC"]:
        list_str_cols.append("noise_ratio")
        list_str_cols.append("w_cutoff_low")
        list_str_cols.append("w_cutoff_high")
    if str_noise_choice in ["1F1B", "WHTCB"]:
        list_str_cols.append("noise_psd_ampl")
        list_str_cols.append("w_cutoff_low")
        list_str_cols.append("w_cutoff_high")
    if str_noise_choice == "SIN":
        list_str_cols.append("noise_ratio")
        list_str_cols.append("freq_sin")
        list_str_cols.append("phase_sin")
    if str_noise_choice == "SINAV":
        list_str_cols.append("noise_ratio")
        list_str_cols.append("freq_sin")
        
    # -> Add columns depending on str_desig
    if flag_desig == True:
        list_str_cols = ["str_desig"] + list_str_cols
        
    return list_str_cols 

# Establish non-data columns of single simulation .csv files
def create_csvcolumns_nodata_onesim(flag_desig=False):
    
    # Establish base-line, non-data columns
    list_str_cols = ["time"]
        
    # -> Add columns depending on str_desig
    if flag_desig == True:
        list_str_cols = ["str_desig"] + list_str_cols
        
    return list_str_cols 

# Establish data columns of full simulation .csv files
def create_csvcolumns_data(dict_operations_fullsim, flag_avg=False):
    
    # Establish base-line, data columns
    list_str_cols = []

    # -> Add columns depending on data choice
    if dict_operations_fullsim["diabErr"] == True:
        list_str_cols.append("diabErr")
    if dict_operations_fullsim["diabErr_cc"] == True:
        list_str_cols.append("diabErr_cc")
    if dict_operations_fullsim["diabErr_ct"] == True:
        list_str_cols.append("diabErr_ct")
        
    # For averages, make columns that hold averages and log averages (also of the square)
    # Include also counters (number of realizations) for each data quantity
    if flag_avg == True:
        list_str_cols_final = []
        for str_cols_data in list_str_cols:
            for str_avgs in ["_avg", "_avg_sqr", "_logAvg", "_logAvg_sqr"]:
                list_str_cols_final.append(str_cols_data + str_avgs)  
            list_str_cols_final.append(str_cols_data + "_count")       
    else:
        list_str_cols_final = list_str_cols
        
    return list_str_cols_final

# Establish data columns of single simulation .csv files
def create_csvcolumns_data_onesim(dict_operations_onesim, flag_avg=False):
    
    # Establish base-line, data columns
    list_str_cols = []

    # -> Add columns depending on data choice
    if dict_operations_onesim["diabErr"] == True:
        list_str_cols.append("diabErr")
    if dict_operations_onesim["diabErr_cc"] == True:
        list_str_cols.append("diabErr_cc")
    if dict_operations_onesim["diabErr_ct"] == True:
        list_str_cols.append("diabErr_ct")
    if dict_operations_onesim["exptEnergy"] == True:
        list_str_cols.append("exptEnergy_0")
        list_str_cols.append("exptEnergy_1")
    if dict_operations_onesim["matElement"] == True:
        list_str_cols.append("matElement")
        
    # For averages, make columns that hold averages and log averages (also of the square)
    # Include also counters (number of realizations) for each data quantity
    if flag_avg == True:
        list_str_cols_final = []
        for str_cols_data in list_str_cols:
            for str_avgs in ["_avg", "_avg_sqr", "_logAvg", "_logAvg_sqr"]:
                list_str_cols_final.append(str_cols_data + str_avgs)  
            list_str_cols_final.append(str_cols_data + "_count")       
    else:
        list_str_cols_final = list_str_cols
        # Add a noise column
        list_str_cols_final.append("noise")
        
    return list_str_cols_final

# Creates a .csv file where data may be stored for full simulations
def create_csvfile(dict_specifications, dict_operations_fullsim, dirname, filename, flag_avg=False, flag_desig=False):

    # Establish non-data columns in .csv
    list_str_cols = create_csvcolumns_nodata(dict_specifications, flag_desig=flag_desig)
        
    # Establish data columns in .csv
    list_str_cols_data = create_csvcolumns_data(dict_operations_fullsim, flag_avg=flag_avg)

    # Check that folders exist; if a folder does not exist, create it
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    # Check that .csv files exists in folder; if a file does not exist, create it
    if os.path.exists("{}/{}.csv".format(dirname, filename)) == False:
        df_empty = pd.DataFrame(columns=list_str_cols + list_str_cols_data)
        # Save to file
        df_empty.to_csv("{}/{}.csv".format(dirname, filename), index=False)
        
# Creates a .csv file where data may be stored for single simulations
def create_csvfile_onesim(dict_operations_onesim, dirname, filename, flag_avg=False, flag_desig=False):

    # Establish non-data columns in .csv
    list_str_cols = create_csvcolumns_nodata_onesim(flag_desig=flag_desig)

    # Establish data columns in .csv
    list_str_cols_data = create_csvcolumns_data_onesim(dict_operations_onesim, flag_avg=flag_avg)

    # Check that folders exist; if a folder does not exist, create it
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    # Check that .csv files exists in folder; if a file does not exist, create it
    if os.path.exists("{}/{}.csv".format(dirname, filename)) == False:
        df_empty = pd.DataFrame(columns=list_str_cols + list_str_cols_data)
        # Save to file
        df_empty.to_csv("{}/{}.csv".format(dirname, filename), index=False)
        
# Create a multi-index that labels data in a .csv for full simulations
def create_csvmultiIndex(dict_operations, dict_specifications, dict_protocol, flag_desig=False):
    
    # Establish non-data columns that are present in .csv which we use as the names of the multi-index
    list_str_cols = create_csvcolumns_nodata(dict_specifications, flag_desig=flag_desig)
        
    # Create the multi-index as a list of lists
    index = []
    index.append([dict_operations["timeoverride"]])
    for name in list_str_cols:
        if name not in ["timeoverride"]:
            index.append([dict_protocol[name]])
    multiIndex = pd.MultiIndex.from_arrays(index, names=tuple(list_str_cols))
    
    return multiIndex
        
# Generates a dictionary containing the column name (of data frame):value
def dict_paramset(dict_operations, dict_specifications, dict_protocol, dict_other, flag_desig=False):
    
    # Establish variables to save to .csv
    dict_fileparams = {"timeoverride":[dict_operations["timeoverride"]], "delta_mu":[dict_protocol["delta_mu"]], "Delta_SC":[dict_protocol["Delta_SC"]], "lp":[dict_protocol["lp"]], "tau":[dict_protocol["tau"]]}
    str_tuning_choice = dict_specifications["tuning"]
    str_noise_choice = dict_specifications["noise"]

    # -> Tuning function parameters
    if str_tuning_choice in ["TRI", "TRID", "TRID2"]:
        for key in ["s_time_init", "s_time_mid", "slope_mid"]:
            dict_fileparams[key] = [dict_protocol[key]]
    if str_tuning_choice in ["TRIB", "TRIC", "SNP", "SNPB"]:
        for key in ["s_time_init", "s_time_mid"]:
            dict_fileparams[key] = [dict_protocol[key]]
    if str_tuning_choice in ["TRIE"]:
        for key in ["s_time_init", "s_time_mid", "scale_factor"]:
            dict_fileparams[key] = [dict_protocol[key]]
        
    # -> Noise parameters 
    if str_noise_choice in ["WHT", "WHTA"]:
        dict_fileparams["noise_ratio"] = [dict_protocol["noise_ratio"]]
    if str_noise_choice in ["1F1", "1F1A", "WHTC"]:
        dict_fileparams["noise_ratio"] = [dict_protocol["noise_ratio"]]
        dict_fileparams["w_cutoff_low"] = [dict_protocol["w_cutoff_low"]]
        dict_fileparams["w_cutoff_high"] = [dict_protocol["w_cutoff_high"]]
    if str_noise_choice in ["1F1B", "WHTCB"]:
        dict_fileparams["noise_psd_ampl"] = [dict_protocol["noise_psd_ampl"]]
        dict_fileparams["w_cutoff_low"] = [dict_protocol["w_cutoff_low"]]
        dict_fileparams["w_cutoff_high"] = [dict_protocol["w_cutoff_high"]]
    if str_noise_choice == "SIN":
        dict_fileparams["noise_ratio"] = [dict_protocol["noise_ratio"]]
        dict_fileparams["freq_sin"] = [dict_protocol["freq_sin"]]
        dict_fileparams["phase_sin"] = [dict_protocol["phase_sin"]]
    if str_noise_choice == "SINAV":
        dict_fileparams["noise_ratio"] = [dict_protocol["noise_ratio"]]
        dict_fileparams["freq_sin"] = [dict_protocol["freq_sin"]]
        
    if flag_desig == True:
        dict_fileparams["str_desig"] = [dict_other["str_desig"]]
        
    return dict_fileparams

# For a given data frame, find the row index that corresponds to the data given
# -> Inputs: dataFrame, dictionary with column names:[parameter], and the column names (as a list) to be omitted (default is [diabErr])
# -> Outputs: the row index of the set of parameters if it exists, otherwise an empty list
def find_rowInd_df(dataFrame, dict_params, key_omit=["diabErr"]):
    
    # Make a deep copy of dataFrame
    dataFrame_cp = dataFrame.copy()
    
    # Search for rows containing requested parameters. Do this iteratively, one parameter at a time
    for key in dict_params:
        if key in key_omit:
            continue
        else:
            dataFrame_cp = dataFrame_cp[dataFrame_cp[key] == dict_params[key][0]]
            
    # Extract the index of the row
    ind_currentdata = (dataFrame_cp.index).tolist()
    return ind_currentdata

# ------ #
# System #
# ------ #

# Hamiltonian of a two level system
# e1_ and e2_ are the diagonal terms
# Delta is the off-diagonal term
def ham_twolevel(e1_, e2_, Delta_):
    return np.array([[e1_, Delta_],[np.conj(Delta_), e2_]])

# Tuning function for sharp protocol (SHP)
def tuning_sharp(mu_i, mu_f, w, Delta, arr_t):
    # Variables
    ti = arr_t[0]
    tf = arr_t[-1]
    mu_i_b = mu_i - w
    mu_f_b = mu_f - w
    sqrroot_i = np.sqrt(mu_i_b**2 + Delta**2) - Delta
    sqrroot_f = np.sqrt(mu_f_b**2 + Delta**2) - Delta
    # Define t0 (where the chemical potential hits zero) and C (controls slope of energy)
    t0 = (ti*sqrroot_f + tf*sqrroot_i)/(sqrroot_i + sqrroot_f)
    C = (sqrroot_i + sqrroot_f)/(tf-ti)
    # Separate time array using t0
    arr_t_before = arr_t[arr_t < t0] 
    arr_t_after = arr_t[arr_t >= t0] 
    # Evaluate chemical potential for each time
    arr_mu_t_b_before = np.sqrt((C*np.abs(arr_t_before - t0) + Delta)**2 - Delta**2)
    arr_mu_t_b_after = -np.sqrt((C*np.abs(arr_t_after - t0) + Delta)**2 - Delta**2)
    arr_mu_b = np.concatenate([arr_mu_t_b_before, arr_mu_t_b_after])
    return arr_mu_b + w

# Cubic polynomial
def cubic(arr_s_c, coeffCubic):
    return coeffCubic[0]*arr_s_c**3 + coeffCubic[1]*arr_s_c**2 + coeffCubic[2]*arr_s_c + coeffCubic[3]

# Custom tuning function, separated into three regions (first and last are symmetric)
# Uses a cubic function to interpolate between the regions
# -> s = t/tau
# -> s1: end point of first region (1 - s1 is the start point for last region)
# -> s2_st: startpoint of middle region (1 - st_st is the end point)
# -> alpha: the slope of the line in the middle region
def tuning_custom_triRegion(arr_s, s1, s2_st, alpha):
    
    # Variables
    s2 = 0.5 - s2_st 
    
    # Make three main regions of tuning function
    f1 = np.zeros(arr_s[arr_s <= s1].shape[0])
    f2 = alpha*(arr_s[(arr_s > 0.5 - s2) & (arr_s < 0.5 + s2)]) + 0.5*(1 - alpha)
    f3 = np.ones(arr_s[arr_s >= 1.0 - s1].shape[0])
    
    # Values of f2 at bounds
    f2_low = alpha*(0.5 - s2) + 0.5*(1 - alpha)
    #f2_high = alpha*(0.5 + s2) + 0.5*(1 - alpha)
    
    # Perform cubic interpolation, manually solving a system of equations
    matCubic = np.array([[s1**3, s1**2, s1, 1.0], [3*s1**2, 2*s1, 1.0, 0.0], [s2_st**3, s2_st**2, s2_st, 1.0], [3*s2_st**2, 2*s2_st, 1.0, 0.0]])
    matVec = np.array([0.0, 0.0, f2_low, alpha])
    
    coeffCubic = sc_lin.solve(matCubic, matVec)
    arr_s_c = arr_s[(arr_s > s1) & (arr_s <= 0.5 - s2)]
    fconnect = cubic(arr_s_c, coeffCubic)
    arr_s_c_2 = arr_s[(arr_s >= 0.5 + s2) & (arr_s < 1.0 - s1)]
    fconnect2 = -cubic(-arr_s_c_2 + 1, coeffCubic) + 1

    """
    # Perform cubic interpolation, using scipy interpolate
    """
    
    # Connect all regions and connectors together into single array and return
    return np.concatenate((f1, fconnect, f2, fconnect2, f3))

# Custom tuning function, separated into three regions (first and last are symmetric)
# Variant B: Tuning function is determined directly from features of the exact spectrum
# Uses a cubic function to interpolate between the regions
# -> s = t/tau
# -> s1: end point of first region (1 - s1 is the start point for last region)
# -> s2_st: startpoint of middle region (1 - st_st is the end point)
# -> alpha: the slope of the line in the middle region
def tuning_custom_triRegion_B(arr_s, s1, s2_st, delta_mu, Delta):
    
    # Variables
    s2 = 0.5 - s2_st 
    
    # Make three main regions of tuning function
    f1 = np.zeros(arr_s[arr_s <= s1].shape[0])
    f2 = np.full(arr_s[(arr_s > 0.5 - s2) & (arr_s < 0.5 + s2)].shape[0], 0.5)
    f3 = np.ones(arr_s[arr_s >= 1.0 - s1].shape[0])
    
    # Values of the energy epsilon at key points
    epsilon_ends = 0.5*np.sqrt(delta_mu**2 + Delta**2)
    epsilon_mid = 0.5*Delta
    
    # Perform cubic interpolation for epsilon, manually solving a system of equations
    matCubic = np.array([[s1**3, s1**2, s1, 1.0], [3*s1**2, 2*s1, 1.0, 0.0], [s2_st**3, s2_st**2, s2_st, 1.0], [3*s2_st**2, 2*s2_st, 1.0, 0.0]])
    matVec = np.array([epsilon_ends, 0.0, epsilon_mid, 0.0])
    coeffCubic = sc_lin.solve(matCubic, matVec)
    
    arr_s_c = arr_s[(arr_s > s1) & (arr_s <= 0.5 - s2)]
    
    if s2_st != 0.5:
        arr_s_c_2 = arr_s[(arr_s >= 0.5 + s2) & (arr_s < 1.0 - s1)]
    else:        
        arr_s_c_2 = arr_s[(arr_s > 0.5 + s2) & (arr_s < 1.0 - s1)]
        
    epsilon_connect = cubic(arr_s_c, coeffCubic)
    epsilon_connect2 = cubic(-arr_s_c_2 + 1, coeffCubic)
    
    """
    if s2_st != 0.5:
        epsilon_connect2 = np.flip(epsilon_connect)
    else:
        epsilon_connect2 = np.flip(epsilon_connect[1:])
    """

    # Find the tuning function for the connecting regions
    fconnect = 0.5*(1 - (1/delta_mu)*np.sqrt(np.abs(4*epsilon_connect**2 - Delta**2)))
    fconnect2 = 0.5*(1 + (1/delta_mu)*np.sqrt(np.abs(4*epsilon_connect2**2 - Delta**2)))
    
    # Connect all regions and connectors together into single array and return
    return np.concatenate((f1, fconnect, f2, fconnect2, f3))

# Custom tuning function, separated into three regions (first and last are symmetric)
# Variant C: Tuning function is determined directly from features of the exact spectrum
# Lines are used to initially create the desired spectrum. This is then smoothened via convolution with a Gaussian kernel
# -> arr_s = t/tau
# -> s_i: end point of first region (1 - s1 is the start point for last region)
# -> s_m: size of the middle region 
def tuning_custom_triRegion_C(arr_s, s_i, s_m, delta_mu, Delta):
    
    # Variables
    epsilon_i = 0.5*np.sqrt(delta_mu**2 + Delta**2)
    epsilon_m = 0.5*Delta
    tau = 1.0 # Set tau = 1 be default. This is a workaround in consideration of the fact that the tuning function should vary with t/tau
    tau_i = s_i
    tau_m = s_m
    arr_time = arr_s
    N = arr_time.shape[0]
    
    # Variables for transition region curves
    # -> Recall that the curves are lines taking the form At + B
    A1 = (epsilon_i - epsilon_m)/(tau_i - tau/2 + tau_m/2)
    A2 = (epsilon_m - epsilon_i)/(tau_i - tau/2 + tau_m/2)
    B1 = (epsilon_m*tau_i - epsilon_i*(tau/2 - tau_m/2))/(tau_i - tau/2 + tau_m/2)
    B2 = (epsilon_i*(tau/2 + tau_m/2) - epsilon_m*(tau - tau_i))/(tau_i - tau/2 + tau_m/2)
    
    # Time arrays for each region
    arr_time_init = arr_time[arr_time<=tau_i]
    arr_time_connectA = arr_time[(arr_time>tau_i)&(arr_time<=tau/2 - tau_m/2)]
    arr_time_mid = arr_time[(arr_time>tau/2 - tau_m/2)&(arr_time<=tau/2 + tau_m/2)]
    arr_time_connectB = arr_time[(arr_time>tau/2 + tau_m/2)&(arr_time<=tau - tau_i)]
    arr_time_end = arr_time[(arr_time>tau-tau_i)&(arr_time<=tau)]

    # Spectrum in each region
    spectrum_init = np.full(arr_time_init.shape[0], epsilon_i)
    spectrum_connectA = A1*arr_time_connectA + B1
    spectrum_mid = np.full(arr_time_mid.shape[0], epsilon_m)
    spectrum_connectB = A2*arr_time_connectB + B2
    spectrum_end = np.full(arr_time_end.shape[0], epsilon_i)
    
    # Combine above into a single array
    arr_spectrum = np.concatenate((spectrum_init, spectrum_connectA, spectrum_mid, spectrum_connectB, spectrum_end))
    
    # Pad the edges of arr_spectrum to avoid boundary effects of convolution
    # -> In the case where tau_i is zero, extend the transition region lines 
    pad_amt = int(N/2)
    if tau_i > 0.0:
        arr_spectrum_pad = np.pad(arr_spectrum, pad_amt, mode="edge")
    else:
        spacing_time = arr_time[1] - arr_time[0]
        tau_pad_left = arr_time[0] - spacing_time*pad_amt
        tau_pad_right = arr_time[-1] + spacing_time*pad_amt
        arr_time_pad_left = np.arange(tau_pad_left, arr_time[0]-spacing_time*0.5, spacing_time)
        arr_time_pad_right = np.arange(arr_time[-1] + spacing_time, tau_pad_right + spacing_time*0.5, spacing_time)
        arr_spectrum_pad_left = A1*arr_time_pad_left + B1
        arr_spectrum_pad_right = A2*arr_time_pad_right + B2
        arr_spectrum_pad = np.concatenate((arr_spectrum_pad_left, arr_spectrum, arr_spectrum_pad_right))
        
    # Generate a Gaussian kernel for smoothing
    sigma = 0.01*tau # By default, this scales with tau
    spacing_time = arr_time[1] - arr_time[0]
    tau_k = tau/2.0 # By default, this scales with tau
    arr_time_kernel = np.arange(-tau_k, tau_k+spacing_time, spacing_time) # It is important that the spacing in this array matches the spacing of the total time array
    y_gaussian = 1/np.sqrt(2*np.pi)/sigma*np.exp(-(arr_time_kernel)**2/2.0/sigma**2)
    y_gaussian = y_gaussian/np.sum(y_gaussian)

    # Smooth the spectrum function by convoluting with the Gaussian kernel
    y_convolve = np.convolve(arr_spectrum_pad, y_gaussian, mode="full")
    y_convolve_inter = y_convolve[int((arr_time_kernel.shape[0] - 1)/2):arr_spectrum_pad.shape[0]+int((arr_time_kernel.shape[0] - 1)/2)] # Extract the desired convoluted results from array, given by the index range here (do not change this)
    y_convolve_act = y_convolve_inter[pad_amt:arr_spectrum_pad.shape[0]-pad_amt] # Remove results which stem from padding
    
    # Actual values of delta_mu and Delta
    # -> Note that we need to update these parameters so that they are consistent with the smoothed spectrum
    epsilon_i_upd = y_convolve_act[0]
    epsilon_m_upd = np.min(y_convolve_act)
    #epsilon_m_upd = y_convolve_act[N//2]
    Delta_upd = 2*epsilon_m_upd
    delta_mu_upd = np.sqrt(4*epsilon_i_upd**2 - Delta_upd**2)
    
    # Calculuate the tuning function
    f_lowerhalf = 0.5*(1 - (1/delta_mu_upd)*np.sqrt(np.abs(4*y_convolve_act[:N//2]**2 - Delta_upd**2)))
    f_upperhalf = 0.5*(1 + (1/delta_mu_upd)*np.sqrt(np.abs(4*y_convolve_act[N//2:]**2 - Delta_upd**2)))
    f_tuning = np.concatenate((f_lowerhalf, f_upperhalf))
    
    #return f_tuning, y_convolve_act, arr_spectrum 
    return f_tuning

# Custom tuning function, separated into three regions (first and last are symmetric)
# Variant D: Lines are used to initially create the desired tuning function. This is then smoothened via convolution with a Gaussian kernel
# -> arr_s = t/tau
# -> s_i: end point of first region (1 - s1 is the start point for last region)
# -> s_m: size of the middle region 
# -> alpha: slope of middle region
def tuning_custom_triRegion_D(arr_s, s_i, s_m, alpha, delta_mu, Delta):
    
    # Variables
    f_init = 0.0
    f_end = 1.0
    slope = alpha
    tau = 1.0 # Set tau = 1 be default. This is a workaround in consideration of the fact that the tuning function should vary with t/tau
    tau_i = s_i
    tau_m = s_m
    arr_time = arr_s
    N = arr_time.shape[0]
    
    # Important points
    g2_P1 = slope*(-tau_m/2) + 0.5
    g2_P2 = slope*(tau_m/2) + 0.5
    
    # Variables for transition region curves
    # -> Recall that the curves are lines taking the form At + B
    A1 = -(g2_P1)/(tau_i - tau/2 + tau_m/2)
    A2 = (g2_P2-1)/(tau_i - tau/2 + tau_m/2)
    B1 = (g2_P1*tau_i)/(tau_i - tau/2 + tau_m/2)
    B2 = (-g2_P2*(tau-tau_i) + (tau/2 + tau_m/2))/(tau_i - tau/2 + tau_m/2)
    
    # Time arrays for each region
    arr_time_init = arr_time[arr_time<=tau_i]
    arr_time_connectA = arr_time[(arr_time>tau_i)&(arr_time<=tau/2 - tau_m/2)]
    arr_time_mid = arr_time[(arr_time>tau/2 - tau_m/2)&(arr_time<=tau/2 + tau_m/2)]
    arr_time_connectB = arr_time[(arr_time>tau/2 + tau_m/2)&(arr_time<=tau - tau_i)]
    arr_time_end = arr_time[(arr_time>tau-tau_i)&(arr_time<=tau)]

    # Tuning function in each region
    tuning_init = np.full(arr_time_init.shape[0], f_init)
    tuning_connectA = A1*arr_time_connectA + B1
    tuning_mid = slope*(arr_time_mid - tau/2) + 0.5
    tuning_connectB = A2*arr_time_connectB + B2
    tuning_end = np.full(arr_time_end.shape[0], f_end)
    
    # Combine above into a single array
    arr_tuning = np.concatenate((tuning_init, tuning_connectA, tuning_mid, tuning_connectB, tuning_end))
    
    # Pad the edges of arr_tuning to avoid boundary effects of convolution
    # -> In the case where tau_i is zero, extend the transition region lines 
    pad_amt = int(N/2)
    if tau_i > 0.0:
        arr_tuning_pad = np.pad(arr_tuning, pad_amt, mode="edge")
    else:
        spacing_time = arr_time[1] - arr_time[0]
        tau_pad_left = arr_time[0] - spacing_time*pad_amt
        tau_pad_right = arr_time[-1] + spacing_time*pad_amt
        arr_time_pad_left = np.arange(tau_pad_left, arr_time[0]-spacing_time*0.5, spacing_time)
        arr_time_pad_right = np.arange(arr_time[-1] + spacing_time, tau_pad_right + spacing_time*0.5, spacing_time)
        arr_tuning_pad_left = A1*arr_time_pad_left + B1
        arr_tuning_pad_right = A2*arr_time_pad_right + B2
        arr_tuning_pad = np.concatenate((arr_tuning_pad_left, arr_tuning, arr_tuning_pad_right))
        
    # Generate a Gaussian kernel for smoothing
    sigma = 0.01*tau # By default, this scales with tau
    spacing_time = arr_time[1] - arr_time[0]
    tau_k = tau/2.0 # By default, this scales with tau
    arr_time_kernel = np.arange(-tau_k, tau_k+spacing_time, spacing_time) # It is important that the spacing in this array matches the spacing of the total time array
    y_gaussian = 1/np.sqrt(2*np.pi)/sigma*np.exp(-(arr_time_kernel)**2/2.0/sigma**2)
    y_gaussian = y_gaussian/np.sum(y_gaussian)

    # Smooth the tuning function by convoluting with the Gaussian kernel
    y_convolve = np.convolve(arr_tuning_pad, y_gaussian, mode="full")
    y_convolve_inter = y_convolve[int((arr_time_kernel.shape[0] - 1)/2):arr_tuning_pad.shape[0]+int((arr_time_kernel.shape[0] - 1)/2)] # Extract the desired convoluted results from array, given by the index range here (do not change this)
    y_convolve_act = y_convolve_inter[pad_amt:arr_tuning_pad.shape[0]-pad_amt] # Remove results which stem from padding
    
    #return arr_tuning, y_convolve_act
    return y_convolve_act

# General function smoother using convolution with a Gaussian kernel
# -> arr_s: Array of dimensionless time 
# -> func: Function to smooth over
def smoother_gaussian(arr_s, func):
    
    # Variables
    tau = 1 # Set tau = 1 be default. This is a workaround in consideration of the fact that the tuning function should vary with t/tau
    arr_time = arr_s
    N = arr_time.shape[0]
    
    # Pad the edges of arr_tuning to avoid boundary effects of convolution
    # -> In the case where tau_i is zero, extend the transition region lines 
    pad_amt = int(N/2)
    func_pad = np.pad(func, pad_amt, mode="edge")

    # Generate a Gaussian kernel for smoothing
    sigma = 0.01*tau # By default, this scales with tau
    spacing_time = arr_time[1] - arr_time[0]
    tau_k = tau/2.0 # By default, this scales with tau
    arr_time_kernel = np.arange(-tau_k, tau_k+spacing_time, spacing_time) # It is important that the spacing in this array matches the spacing of the total time array
    y_gaussian = 1/np.sqrt(2*np.pi)/sigma*np.exp(-(arr_time_kernel)**2/2.0/sigma**2)
    y_gaussian = y_gaussian/np.sum(y_gaussian)

    # Smooth the tuning function by convoluting with the Gaussian kernel
    y_convolve = np.convolve(func_pad, y_gaussian, mode="full")
    y_convolve_inter = y_convolve[int((arr_time_kernel.shape[0] - 1)/2):func_pad.shape[0]+int((arr_time_kernel.shape[0] - 1)/2)] # Extract the desired convoluted results from array, given by the index range here (do not change this)
    y_convolve_act = y_convolve_inter[pad_amt:func_pad.shape[0]-pad_amt] # Remove results which stem from padding
    
    #return arr_tuning, y_convolve_act
    return y_convolve_act

# Custom tuning function, separated into three regions (first and last are symmetric) with transition regions given by sin^2 functions
# -> arr_s = t/tau
# -> s_i: end point of first region (1 - s1 is the start point for last region)
# -> s_m: size of the middle region, which is flat
def tuning_custom_SNP(arr_s, s_i, s_m, delta_mu, Delta):
    
    # Variables
    f_init = 0.0
    f_end = 1.0
    tau = 1.0 # Set tau = 1 be default. This is a workaround in consideration of the fact that the tuning function should vary with t/tau
    tau_i = s_i
    tau_m = s_m
    arr_time = arr_s
    #N = arr_time.shape[0]
    
    # Variables for transition region curves
    A1 = 0.5
    A2 = 0.5
    B1 = np.pi/2.0/(0.5 - 0.5*tau_m - s_i)
    B2 = np.pi/2.0/(0.5 - 0.5*tau_m - s_i)
    C1 = -s_i
    C2 = -0.5 - 0.5*tau_m
    D1 = 0.0
    D2 = 0.5
    
    # Time arrays for each region
    arr_time_init = arr_time[arr_time<=tau_i]
    arr_time_connectA = arr_time[(arr_time>tau_i)&(arr_time<=tau/2 - tau_m/2)]
    arr_time_mid = arr_time[(arr_time>tau/2 - tau_m/2)&(arr_time<=tau/2 + tau_m/2)]
    arr_time_connectB = arr_time[(arr_time>tau/2 + tau_m/2)&(arr_time<=tau - tau_i)]
    arr_time_end = arr_time[(arr_time>tau-tau_i)&(arr_time<=tau)]

    # Tuning function in each region
    tuning_init = np.full(arr_time_init.shape[0], f_init)
    tuning_connectA = A1*np.sin(B1*(arr_time_connectA + C1))**2 + D1
    tuning_mid = np.full(arr_time_mid.shape[0], 0.5)
    tuning_connectB = A2*np.sin(B2*(arr_time_connectB + C2))**2 + D2
    tuning_end = np.full(arr_time_end.shape[0], f_end)
    
    # Combine above into a single array
    arr_tuning = np.concatenate((tuning_init, tuning_connectA, tuning_mid, tuning_connectB, tuning_end))
    
    return arr_tuning
            
# ----- #
# Noise #
# ----- #

# Power spectral density for white noise with cutoff frequencies
def psd_whitenoise(ampl, w_cutoff_low, w_cutoff_high, arr_w):
    
    # Power spectral density 
    psd = np.full(arr_w.shape[0], ampl)
    
    # Identify indices where the psd is zero according to the cutoff frequencies
    ind_zero = [ind for ind in range(0, arr_w.shape[0]) if np.abs(arr_w[ind]) >= w_cutoff_high or np.abs(arr_w[ind]) <= w_cutoff_low]
    psd[ind_zero] = 0.0
    
    return psd

# Power spectral density for 1/f noise with cutoff frequencies
def psd_1f1noise(ampl, w_cutoff_low, w_cutoff_high, arr_w):
    
    # Power spectral density 
    psd = np.full(arr_w.shape[0], ampl)
    # -> Identify the zero frequency index
    ind_w_zero = np.argwhere(arr_w == 0.0)
    # -> Replace the zero frequency element of psd with zero
    psd[ind_w_zero] = 0.0
    # -> Introduce a copy of arr_w, which we modify
    arr_w_cp = np.copy(arr_w)
    arr_w_cp[ind_w_zero] = 1.0
    # -> Create the psd
    psd = np.divide(psd, np.abs(arr_w_cp))
    
    # Identify indices where the psd is zero according to the cutoff frequencies
    ind_zero = [ind for ind in range(0, arr_w.shape[0]) if np.abs(arr_w[ind]) >= w_cutoff_high or np.abs(arr_w[ind]) <= w_cutoff_low]
    psd[ind_zero] = 0.0
    
    return psd

# Generate white noise with cutoff frequencies
def gen_whitenoise(ampl, w_cutoff_low, w_cutoff_high, dt_noise, N_noise):
    
    # Sample frequencies
    # -> Note: Since N_noise is even, arr_w will take positive values from 0 to N_noise/2-1 and negative values thereafter
    # -> Note: The above is the natural order for ffts and iffts so best to keep this and adapt around it
    arr_w = 2*np.pi*sc_fft.fftfreq(N_noise, d=dt_noise)
    
    # Power spectral density
    psd = psd_whitenoise(ampl, w_cutoff_low, w_cutoff_high, arr_w)
    
    # Generate "base" white noise: zero mean, unity variance, no cutoffs, picked from a Gaussian distribution
    arr_whitenoise_base = np.random.normal(loc=0.0, scale=1.0, size=N_noise)
    
    # Fourier transform the base white noise
    arr_whitenoise_base_fft = sc_fft.fft(arr_whitenoise_base) 
    
    # Use the PSD and the FT of the base white noise to generate the desired white noise with cutoffs    
    arr_noise_fft = np.sqrt(psd/dt_noise)*arr_whitenoise_base_fft
    arr_noise = sc_fft.ifft(arr_noise_fft)
    
    # Return the real part of arr_noise, which is guaranteed to be real
    return np.real(arr_noise)

# Generate white noise with cutoff frequencies
def gen_1f1noise(ampl, w_cutoff_low, w_cutoff_high, dt_noise, N_noise):
    
    # Sample frequencies
    # -> Note: Since N_noise is even, arr_w will take positive values from 0 to N_noise/2-1 and negative values thereafter
    # -> Note: The above is the natural order for ffts and iffts so best to keep this and adapt around it
    arr_w = 2*np.pi*sc_fft.fftfreq(N_noise, d=dt_noise)
    
    # Power spectral density
    psd = psd_1f1noise(ampl, w_cutoff_low, w_cutoff_high, arr_w)
    
    # Generate "base" white noise: zero mean, unity variance, no cutoffs, picked from a Gaussian distribution
    arr_whitenoise_base = np.random.normal(loc=0.0, scale=1.0, size=N_noise)
    
    # Fourier transform the base white noise
    arr_whitenoise_base_fft = sc_fft.fft(arr_whitenoise_base) 
    
    # Use the PSD and the FT of the base white noise to generate the desired white noise with cutoffs    
    arr_noise_fft = np.sqrt(psd/dt_noise)*arr_whitenoise_base_fft
    arr_noise = sc_fft.ifft(arr_noise_fft)
    
    # Return the real part of arr_noise, which is guaranteed to be real
    return np.real(arr_noise)

def varToAmp_whitenoise(var, w_cutoff_low, w_cutoff_high, arr_w, tau_noise):
    # PSD with unit amplitude
    psd = psd_whitenoise(1.0, w_cutoff_low, w_cutoff_high, arr_w)
    return var*tau_noise/np.sum(psd)

def varToAmp_1f1noise(var, w_cutoff_low, w_cutoff_high, arr_w, tau_noise):
    # PSD with unit amplitude
    psd = psd_1f1noise(1.0, w_cutoff_low, w_cutoff_high, arr_w)
    return var*tau_noise/np.sum(psd)

def ampToVar_whitenoise(ampl, w_cutoff_low, w_cutoff_high, arr_w, tau_noise):
    # PSD
    psd = psd_whitenoise(ampl, w_cutoff_low, w_cutoff_high, arr_w)
    return np.sum(psd)/tau_noise

def ampToVar_1f1noise(ampl, w_cutoff_low, w_cutoff_high, arr_w, tau_noise):
    # PSD
    psd = psd_1f1noise(ampl, w_cutoff_low, w_cutoff_high, arr_w)
    return np.sum(psd)/tau_noise

def muNDmax_to_ratio(muND_max, muRightStart, muRightEnd):
    return muND_max/np.abs(muRightStart - muRightEnd)

###############################################################################

# Power spectral density for 1/f (2pi/w) noise with cutoff frequencies
def psd_1f(A, w, w_low, w_high):
    # Extract frequencies only between the cutoffs
    w_rel_negative = w[(w>=-w_high)&(w<=-w_low)]
    w_rel_positive = w[(w>=w_low)&(w<=w_high)]
    # Evaluate the psd for both relevant frequencies
    psd_negative = A*2*np.pi/np.abs(w_rel_negative)
    psd_positive = A*2*np.pi/w_rel_positive
    # Pad the psd with zero at the ends an in the middle so that its length matches that of w
    psd = np.concatenate((np.zeros(w[w<-w_high].shape[0]), psd_negative, np.zeros(w[(w>-w_low)&(w<w_low)].shape[0]), psd_positive, np.zeros(w[w>w_high].shape[0])))
    return psd

# Power spectral density for white noise with cutoff frequencies
def psd_white(A, w, w_low, w_high):
    # Extract frequencies only between the cutoffs
    w_rel_negative = w[(w>=-w_high)&(w<=-w_low)]
    w_rel_positive = w[(w>=w_low)&(w<=w_high)]
    # Evaluate the psd for both relevant frequencies
    psd_negative = np.full(w_rel_negative.shape[0], A)
    psd_positive = np.full(w_rel_positive.shape[0], A)
    # Pad the psd with zero at the ends an in the middle so that its length matches that of w
    psd = np.concatenate((np.zeros(w[w<-w_high].shape[0]), psd_negative, np.zeros(w[(w>-w_low)&(w<w_low)].shape[0]), psd_positive, np.zeros(w[w>w_high].shape[0])))
    return psd

# Generate 1/f (2pi/w) noise  with cutoff frequencies
def noise_1f(var_ns, arr_t_ns, w_cutlow, w_cuthigh):
    
    # Parameters
    N_noise = arr_t_ns.shape[0]
    dt = arr_t_ns[1] - arr_t_ns[0]
    arr_w_fft = 2*np.pi*sc_fft.fftshift(sc_fft.fftfreq(N_noise, dt)) # Sample frequencies w
    
    # Determine A based on the variance and frequency cutoffs (see for ref. equations in notes)
    # -> Integral variant
    #A = var_ns/np.log(w_cuthigh/w_cutlow)/2.0
    
    # Determine A based on the variance and frequency cutoffs (see for ref. equations in notes)
    # -> Sum variant
    var_unity = (1.0/N_noise/dt)*np.sum(psd_1f(1.0, arr_w_fft, w_cutlow, w_cuthigh))
    A = var_ns/var_unity 
    
    # Generate white noise with zero mean and unity variance from a Gaussian distribution
    arr_whitenoise = np.random.normal(loc=0.0, scale=1.0, size=N_noise)
    
    # Fourier transform the white noise vector
    arr_whitenoise_fft = sc_fft.fftshift(sc_fft.fft(arr_whitenoise)) # FT 
    
    # Generate the desired noise in frequency space using the PSD; inverse FT to get things in time
    arr_noise_fft = np.sqrt(psd_1f(A, arr_w_fft, w_cutlow, w_cuthigh)/dt)*arr_whitenoise_fft
    arr_noise = sc_fft.ifft(arr_noise_fft)
    
    return np.real(arr_noise)

# Generate white noise with cutoff frequencies
def noise_white(var_ns, arr_t_ns, w_cutlow, w_cuthigh):
    
    # Parameters
    N_noise = arr_t_ns.shape[0]
    dt = arr_t_ns[1] - arr_t_ns[0]
    arr_w_fft = 2*np.pi*sc_fft.fftshift(sc_fft.fftfreq(N_noise, dt)) # Sample frequencies w
    
    # Determine A based on the variance and frequency cutoffs (see for ref. equations in notes)
    # -> Integral variant
    #A = var_ns/np.log(w_cuthigh/w_cutlow)/2.0
    
    # Determine A based only on the desired noise variance and dt
    A = var_ns*dt
    
    # Generate white noise with zero mean and unity variance from a Gaussian distribution
    arr_whitenoise = np.random.normal(loc=0.0, scale=1.0, size=N_noise)
    
    # Fourier transform the white noise vector
    arr_whitenoise_fft = sc_fft.fftshift(sc_fft.fft(arr_whitenoise)) # FT 
    
    # Generate the desired noise in frequency space using the PSD; inverse FT to get things in time
    arr_noise_fft = np.sqrt(psd_white(A, arr_w_fft, w_cutlow, w_cuthigh)/dt)*arr_whitenoise_fft
    arr_noise = sc_fft.ifft(arr_noise_fft)
    
    return np.real(arr_noise)

# 1/f: Calculate the variance from the amplitude
def psd_1f_VarToAmp(var_ns, w_cutlow, w_cuthigh, tau_noise, dt):
    # Calculate N_noise
    N_noise = int(tau_noise/dt)
    # Adjust tau_noise so that tau_noise = N_noise*dt
    tau_noise = N_noise*dt
    # Create frequencies in w-space
    arr_w_fft = 2*np.pi*sc_fft.fftshift(sc_fft.fftfreq(N_noise, dt))
    # Calculate the value of the amplitude from the variance and the following sum
    var_unity = (1.0/N_noise/dt)*np.sum(psd_1f(1.0, arr_w_fft, w_cutlow, w_cuthigh))
    A = var_ns/var_unity 
    return A

# 1/f: Calculate the amplitude from the variance
def psd_1f_AmpToVar(A_ns, w_cutlow, w_cuthigh, tau_noise, dt):
    # Calculate N_noise
    N_noise = int(tau_noise/dt)
    # Adjust tau_noise so that tau_noise = N_noise*dt
    tau_noise = N_noise*dt
    # Create frequencies in w-space
    arr_w_fft = 2*np.pi*sc_fft.fftshift(sc_fft.fftfreq(N_noise, dt))
    # Calculate the value of the variance from the amplitude and the following sum
    var_unity = (1.0/N_noise/dt)*np.sum(psd_1f(1.0, arr_w_fft, w_cutlow, w_cuthigh))
    var_ns = A_ns*var_unity
    return var_ns

################################################################################

# -------- #
# Plotting #
# -------- #

# Generate title and legend text strings for plots
def str_plot(nbr_datasets, dict_specifications, dict_protocol, vary_param_choice, flag_noNmin=False):
    
    # Save minimum number of realizations
    nbr_realizations_min = dict_protocol["nbr_realizations_min"]
    
    # Create a dictionaries with entries parameter:label
    dict_all = dict_specifications
    dict_all.update(dict_protocol)
    del dict_all["nbr_realizations_min"]
    #dict_all["s_time_init"] = dict_all["s_time_init"]*dict_all["tau"]
    #dict_all["s_time_mid"] = dict_all["s_time_mid"]*dict_all["tau"]
    list_keys = list(dict_all.keys())
    #list_strs = ["{}", "{}", r"$\delta \mu = {}$", r"$\Delta_{{\mathrm{{SC}}}} = {}$", r"$l_{{\mathrm{{p}}}} = {}$", r"$\tau = {}$", r"$\tau_{{\mathrm{{i}}}} = {} \tau$", r"$\tau_{{\mathrm{{m}}}} = {} \tau$", r"$\alpha = {}$", r"$r_{{\mathrm{{n}}}} = {}$", r"$A_{{\mathrm{{psd}}}} = {}$", r"$\omega_{{\mathrm{{l}}}} = {}$", r"$\omega_{{\mathrm{{h}}}} = {}$", r"$\omega = {}$", r"$\phi = {}$"]
    #list_strs = ["{}", "{}", r"$\delta \mu = {}$", r"$\Delta_{{\mathrm{{SC}}}} = {}$", r"$l_{{\mathrm{{p}}}} = {}$", r"$\tau = {}$", r"$\tau_{{\mathrm{{i}}}} = {:.2f}$", r"$\tau_{{\mathrm{{m}}}} = {:.2f}$", r"$\alpha = {}$", r"$r_{{\mathrm{{n}}}} = {}$", r"$A_{{\mathrm{{psd}}}} = {}$", r"$\omega_{{\mathrm{{l}}}} = {}$", r"$\omega_{{\mathrm{{h}}}} = {}$", r"$\omega = {}$", r"$\phi = {}$"]
    list_strs = ["{}", "{}", r"$\delta \mu = {}$", r"$\Delta_{{\mathrm{{SC}}}} = {}$", r"$l_{{\mathrm{{p}}}} = {}$", r"$\tau = {:.2f}$", r"$\tau_{{\mathrm{{i}}}}/\tau = {:.3f}$", r"$\tau_{{\mathrm{{m}}}}/\tau = {:.3f}$", r"$\alpha = {:.3f}$", r"$f_{{s}} = {:.3f}$", r"$r_{{\mathrm{{n}}}} = {}$", r"$A_{{\mathrm{{psd}}}} = {}$", r"$\omega_{{\mathrm{{l}}}} = {}$", r"$\omega_{{\mathrm{{h}}}} = {}$", r"$\omega = {:.3}$", r"$\phi = {}$"]
    dict_all_str = dict(zip(list_keys, list_strs))

    # Omit the parameter corresponding to vary_param_choice
    if vary_param_choice == "TAU":
        del dict_all["tau"]
        del dict_all_str["tau"]
        
    if vary_param_choice == "NR":
        del dict_all["noise_ratio"]
        del dict_all_str["noise_ratio"]
        
    if vary_param_choice == "W":
        del dict_all["freq_sin"]
        del dict_all_str["freq_sin"]
        
    if vary_param_choice == "PH":
        del dict_all["phase_sin"]
        del dict_all_str["phase_sin"]
        
    if vary_param_choice == "STIMES":
        del dict_all["s_time_init"]
        del dict_all_str["s_time_init"]
    
    if vary_param_choice == "STIMEM":
        del dict_all["s_time_mid"]
        del dict_all_str["s_time_mid"]
    
    if vary_param_choice == "SLOPE":
        del dict_all["slope_mid"]
        del dict_all_str["slope_mid"]
    
    # Omit certain parameters depending on str_tuning_choice and/or str_noise_choice 
    list_tuningtype_tri_range = ["TRI", "TRIB", "TRIC", "TRID", "TRID2", "TRIE", "SNP", "SNPB"] # Keep STIMES, STIMESM
    list_tuningtype_tri_slope = ["TRI", "TRID", "TRID2"] # Keep SLOPE
    list_tuningtype_tri_factor = ["TRIE"] # Keep scale factor
    list_noisetype_nr = ["WHT", "WHTC", "1F1", "SIN", "SINAV"] # Keep noise_ratio
    list_noisetype_ampl = ["WHTCB", "1F1B"] # Keep amplitude
    list_noisetype_cutoffs = ["WHTC", "WHTCB", "1F1", "1F1B"] # Keep cutoffs
    list_noisetype_freq = ["SIN", "SINAV"] # Keep frequency
    list_noisetype_phase = ["SIN"] # Keep phase
    
    if any(x in dict_all["tuning"] for x in list_tuningtype_tri_range) == False:
        del dict_all["s_time_init"] 
        del dict_all["s_time_mid"]
        del dict_all_str["s_time_init"] 
        del dict_all_str["s_time_mid"]
        
    if any(x in dict_all["tuning"] for x in list_tuningtype_tri_slope) == False:
        del dict_all["slope_mid"]
        del dict_all_str["slope_mid"]
        
    if any(x in dict_all["tuning"] for x in list_tuningtype_tri_factor) == False:
        del dict_all["scale_factor"]
        del dict_all_str["scale_factor"]

    if any(x in dict_all["noise"] for x in list_noisetype_nr) == False:
        del dict_all["noise_ratio"]
        del dict_all_str["noise_ratio"]
    
    if any(x in dict_all["noise"] for x in list_noisetype_ampl) == False:
        del dict_all["noise_psd_ampl"] 
        del dict_all_str["noise_psd_ampl"]
        
    if any(x in dict_all["noise"] for x in list_noisetype_cutoffs) == False:
        del dict_all["w_cutoff_low"] 
        del dict_all["w_cutoff_high"]
        del dict_all_str["w_cutoff_low"] 
        del dict_all_str["w_cutoff_high"]
        
    if any(x in dict_all["noise"] for x in list_noisetype_freq) == False:
        del dict_all["freq_sin"] 
        del dict_all_str["freq_sin"] 
        
    if any(x in dict_all["noise"] for x in list_noisetype_phase) == False:
        del dict_all["phase_sin"]
        del dict_all_str["phase_sin"]
    
    # Use the static parameters to define the title labels
    list_str_title = []
    
    # Use the dynamic parameters to define the legend labels for each plot
    list_str_legend = [[] for _ in range(nbr_datasets)]
    
    # Distinguish between static and dynamic parameters; fill the above lists
    for key in dict_all.keys():
        # Static parameters if True, dynamic parameters otherwise
        if np.all(dict_all[key] == dict_all[key][0]) == True or all(x in [dict_all[key][0]] for x in dict_all[key]):
            str_key_param = dict_all_str[key].format(dict_all[key][0])
            list_str_title.append(str_key_param)
        else:
            for i in range(0, nbr_datasets):
                str_key_param = dict_all_str[key].format(dict_all[key][i])
                list_str_legend[i].append(str_key_param)
    
    # Form plot title
    str_plotTitle = ", ".join(list_str_title) + ", $N_{{\mathrm{{min}}}} = {}$".format(nbr_realizations_min)
    
    if flag_noNmin == True:
        str_plotTitle = str_plotTitle.replace(", $N_{{\mathrm{{min}}}} = {}$".format(nbr_realizations_min), "")
    
    # Add a line break as close as possible to half-way through str_plotTitle
    # -> Use commas to find index of split
    loc_commas = [x for x in range(len(str_plotTitle)) if str_plotTitle.startswith(", ", x)]
    ind_loc_comma_half = np.argmin(np.abs(len(str_plotTitle)//2 - np.array(loc_commas)))
    # -> Add a line break to string
    str_plotTitle = str_plotTitle[:loc_commas[ind_loc_comma_half]+1] + "\n" + str_plotTitle[loc_commas[ind_loc_comma_half]+1:]
    
    return str_plotTitle, list_str_legend    