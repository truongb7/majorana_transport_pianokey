"""
Kitaev chain: Piano key simulations
Functions

A suite of functions related to numerically simulating piano key transport in a Kitaev chain
"""

# ------- #
# Modules #
# ------- #

import argparse as ap
import numpy as np
import os as os
import pandas as pd
from pfapack import pfaffian as pf
import scipy.fft as sc_fft
import scipy.linalg as sc_linalg
import tl_functions as tl

# -------- #
# Overhead #
# -------- #

# Takes a list of strings ["[a,b]", "[c]" ...] and converts the inner stringed lists to actual lists, where a, b, c, ... are integers
def transProb_strToList(str_list):
    list_ind = []
    for str_inner in str_list:
        str_inner_inter = str_inner.replace("[", "")
        str_inner_inter = str_inner_inter.replace("]", "")
        list_inner = [int(ind) for ind in str_inner_inter.split(",")]
        list_ind.append(list_inner)
    return list_ind

# Argument parser, allows for optional command line arguments of parameters
# Returns all parameters updated after command line arguments have been provided
# -> It is important that the destination key for each argument matches the corresponding dictionary keys up to the prefixes
def argument_parser(dict_machine, dict_operations_main, dict_operations_fullsim, dict_operations_onesim, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other, dict_directories):
    
    # Command line arguments and parameters
    parser = ap.ArgumentParser(description="Specifications and parameters for piano key protocols")
    # -> Machine specifications
    parser.add_argument('--mc_mach', type=str, dest="mc_machine")
    # -> Operations specifications, main
    parser.add_argument('--op_m_clus', type=str, dest='op_m_cluster')
    parser.add_argument('--op_m_osim', type=str, dest='op_m_onesim')
    parser.add_argument('--op_m_avg', type=str, dest='op_m_average')
    # -> Operations specifications, fullsim
    parser.add_argument('--op_f_diab', type=str, dest='op_f_diabErr')
    parser.add_argument('--op_f_diab_cc', type=str, dest='op_f_diabErr_cc')
    parser.add_argument('--op_f_diab_ct', type=str, dest='op_f_diabErr_ct')
    parser.add_argument('--op_f_trpr', type=str, dest='op_f_transProb')
    parser.add_argument('--op_f_trpr_cc', type=str, dest='op_f_transProb_cc')
    parser.add_argument('--op_f_min', type=str, dest='op_f_mingap')
    parser.add_argument('--op_f_vel', type=str, dest='op_f_velocity')
    parser.add_argument('--op_f_minsep', type=str, dest='op_f_mingap_sep')
    # -> Operations specifications, onesim
    parser.add_argument('--op_s_diab', type=str, dest='op_s_diabErr')
    parser.add_argument('--op_s_diab_cc', type=str, dest='op_s_diabErr_cc')
    parser.add_argument('--op_s_diab_ct', type=str, dest='op_s_diabErr_ct')
    parser.add_argument('--op_s_inen', type=str, dest='op_s_instEnergy')
    parser.add_argument('--op_s_inen_c', type=str, dest='op_s_instEnergy_c')
    parser.add_argument('--op_s_exen', type=str, dest='op_s_exptEnergy')
    parser.add_argument('--op_s_occp', type=str, dest='op_s_occupation')
    parser.add_argument('--op_s_mate', type=str, dest='op_s_matElement')
    parser.add_argument('--op_s_trpr', type=str, dest='op_s_transProbs')
    parser.add_argument('--op_s_trpr_cc', type=str, dest='op_s_transProbs_cc')
    parser.add_argument('--op_s_inev', type=str, dest='op_s_instEigVec')
    parser.add_argument('--op_s_evev', type=str, dest='op_s_evolEigVec')
    # -> Operations specifications, other
    parser.add_argument('--op_o_save', type=str, dest='op_o_savedata')
    parser.add_argument('--op_o_nfix', type=str, dest='op_o_Nfix')
    parser.add_argument('--op_o_tord', type=str, dest='op_o_timeoverride')
    parser.add_argument('--op_o_svd', type=str, dest='op_o_SVD')
    parser.add_argument('--op_o_svde', type=str, dest='op_o_SVD_END')
    # -> Protocol specifications
    parser.add_argument('--pr_tune', type=str, dest='pr_tuning')
    parser.add_argument('--pr_disr', type=str, dest='pr_disorder')
    parser.add_argument('--pr_nois', type=str, dest='pr_noise')
    # -> Protocol parameters
    # ---> System
    parser.add_argument('--pa_L', type=int, dest='pa_L')
    parser.add_argument('--pa_R', type=int, dest='pa_R')
    parser.add_argument('--pa_delt', type=float, dest='pa_Delta')
    parser.add_argument('--pa_w', type=float, dest='pa_w')
    parser.add_argument('--pa_mulf', type=float, dest='pa_muLeft')
    parser.add_argument('--pa_murs', type=float, dest='pa_muRightStart')
    parser.add_argument('--pa_mure', type=float, dest='pa_muRightEnd')
    parser.add_argument('--pa_lp', type=int, dest='pa_lp')
    parser.add_argument('--pa_nstp', type=int, dest='pa_n_steps')
    # ---> Tuning function
    parser.add_argument('--pa_sti', type=float, dest='pa_s_time_init')
    parser.add_argument('--pa_stm', type=float, dest='pa_s_time_mid')
    parser.add_argument('--pa_spm', type=float, dest='pa_slope_mid')
    parser.add_argument('--pa_scl', type=float, dest='pa_scale_factor')
    # ---> Time
    parser.add_argument('--pa_tau', type=float, dest='pa_tau')
    parser.add_argument('--pa_dt', type=float, dest='pa_dt')
    parser.add_argument('--pa_nthr', type=int, dest='pa_Nthresh')
    parser.add_argument('--pa_nfix', type=int, dest='pa_Nfix')
    # ---> Number of realizations
    parser.add_argument('--pa_nbrl', type=int, dest='pa_nbr_realizations')
    # -> Disorder
    parser.add_argument('--pd_r', type=float, dest='pd_disorder_ratio')
    parser.add_argument('--pd_xi', type=float, dest='pd_length_corr')
    # -> Noise
    parser.add_argument('--pn_r', type=float, dest='pn_noise_ratio')
    parser.add_argument('--pn_A', type=float, dest='pn_noise_psd_ampl')
    parser.add_argument('--pn_wcl', type=float, dest='pn_w_cutoff_low')
    parser.add_argument('--pn_wch', type=float, dest='pn_w_cutoff_high')
    parser.add_argument('--pn_tau', type=float, dest='pn_tau_noise')
    parser.add_argument('--pn_dt', type=float, dest='pn_dt_noise')
    parser.add_argument('--pn_w_sin', type=float, dest='pn_freq_sin')
    parser.add_argument('--pn_p_sin', type=float, dest='pn_phase_sin')
    # -> Calculation quantity specifications, full simulations
    parser.add_argument('--cq_f_trpr', nargs="*", type=str, dest='cq_f_transProbs')
    parser.add_argument('--cq_f_trpr_cc', nargs="*", type=str, dest='cq_f_transProbs_cc')
    # -> Calculation quantity specifications, single simulations
    parser.add_argument('--cq_s_inen', nargs="*", type=int, dest='cq_s_instEnerg')
    parser.add_argument('--cq_s_inen_c', nargs="*", type=int, dest='cq_s_instEnerg_c')
    parser.add_argument('--cq_s_exen', nargs="*", type=int, dest='cq_s_exptEnerg')
    parser.add_argument('--cq_s_occp', nargs="*", type=int, dest='cq_s_occupation')
    parser.add_argument('--cq_s_trpr', nargs="*", type=str, dest='cq_s_transProbs')
    parser.add_argument('--cq_s_trpr_cc', nargs="*", type=str, dest='cq_s_transProbs_cc')
    parser.add_argument('--cq_s_inev', nargs="*", type=int, dest='cq_s_instEigVec')
    parser.add_argument('--cq_s_evev', nargs="*", type=int, dest='cq_s_evolEigVec')
    # -> Other parameters
    parser.add_argument('--po_nsamp', type=int, dest='po_NSamp')
    parser.add_argument('--po_desig', type=str, dest='po_str_desig')
    parser.add_argument('--po_desig2', type=str, dest='po_str_desig2')
    # -> Directories
    parser.add_argument('--dr_main', type=str, dest='dr_data')
    parser.add_argument('--dr_avg', type=str, dest='dr_data_average')
    parser.add_argument('--dr_one', type=str, dest='dr_data_onesim')
    parser.add_argument('--dr_oavg', type=str, dest='dr_data_onesim_average')
    parser.add_argument('--dr_min', type=str, dest='dr_data_mingap')
    
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
    list_dicts = [dict_machine, dict_operations_main, dict_operations_fullsim, dict_operations_onesim, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other, dict_directories]
    
    # Loop through each dictionary. Replace the values in the original dictionary with the command line input if provided
    # -> List of dictionary prefixes; must match order of list_dicts
    list_dict_prefix = ["mc_", "op_m_", "op_f_", "op_s_", "op_o_", "pr_", "pa_", "pd_", "pn_", "cq_f_", "cq_s_", "po_", "dr_"]
    
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
                
    # For lists of lists, such as for transProb(s), need to convert stringed lists to actual lists of lists
    for key in ['transProbs', 'transProbs_cc']:
        if dict_args_cmdline["cq_f_" + key] != None:
            list_ind_f = transProb_strToList(dict_args_cmdline["cq_f_" + key])
            dict_fullsim[key] = list_ind_f
        if dict_args_cmdline["cq_s_" + key] != None:
            list_ind_s = transProb_strToList(dict_args_cmdline["cq_s_" + key])
            dict_onesim[key] = list_ind_s
                
    # Return updated dictionaries
    return list_dicts

# Parameter string creator, for data file names
# -> Input: dictionaries of parameters and specifications
# -> Output: parameter string
def str_params(dict_operations_main, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=False, flag_noTau=False):
    
    # Unpack parameters from dictionaries  
    # -> Operations, main
    flag_cluster = dict_operations_main["cluster"]
    # -> Operations, other
    flag_timeoverride = dict_operations_other["timeoverride"]
    # -> Specifications
    str_tuning_choice = dict_specifications["tuning"]
    str_disorder_choice = dict_specifications["disorder"]
    str_noise_choice = dict_specifications["noise"]
    # -> Other
    NSamp = dict_other["NSamp"]
    str_desig = dict_other["str_desig"]
    
    # Dictionary of relevant parameters
    dict_params_rel = {}
    dict_params_rel.update(dict_protocol)
    dict_params_rel.update(dict_disorder)
    dict_params_rel.update(dict_noise)
     
    # Dictionary of text strings corresponding to numerical parameter (order is important here!)
    dict_textstr = {"tau":"tI", "n_steps":"n", "L":"L", "R":"R", "lp":"lp", "Delta":"D", "w":"w", "muLeft":"uL", "muRightStart":"uRI", "muRightEnd":"uRF", "s_time_init":"sTI", "s_time_mid":"sTM", "slope_mid":"sM", "scale_factor":"sF", "disorder_ratio":"mD", "length_corr":"cL", "noise_ratio":"mN", "noise_psd_ampl":"aN", "w_cutoff_low":"wCL", "w_cutoff_high":"wCH", "freq_sin":"w", "phase_sin":"pH"}
    
    # If desired, remove certain strings and parameters
    if flag_noTau == True:
        del dict_params_rel["tau"]
        del dict_textstr["tau"]
    
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
    elif str_tuning_choice == "TRIE":
        str_evol = "_TRIE" 
    elif str_tuning_choice == "TRID2":
        str_evol = "_TRID2"
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
        
    # -> Assign appropriate string depending on disorder type
    if str_disorder_choice == "UCUN":
        str_disorder = "_UCUN"
    elif str_disorder_choice == "UCND":
        str_disorder = "_UCND"
    elif str_disorder_choice == "UCUNB":
        str_disorder = "_UCUNB"
    elif str_disorder_choice == "GCUN":
        str_disorder = "_GCUN"
    elif str_disorder_choice == "GCND":
        str_disorder = "_GCND"
    elif str_disorder_choice == "GCNDB":
        str_disorder = "_GCNDB"
    elif str_disorder_choice == "ECND":
        str_disorder = "_ECND"
    elif str_disorder_choice == "ECNDB":
        str_disorder = "_ECNDB"
    elif str_disorder_choice == "SCND":
        str_disorder = "_SCND"
    elif str_disorder_choice == "SCNDB":
        str_disorder = "_SCNDB"
    else:
        str_disorder = ""

    # -> Assign appropriate string depending on noise type
    if str_noise_choice == "1F1":
        str_noise = "_1F1"
    elif str_noise_choice == "1F1B":
        str_noise = "_1F1B"
    elif str_noise_choice == "WHT":
        str_noise = "_WHT"
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
        
    # -> If cluster mode is on, clear str_evol, str_disorder, str_noise
    if flag_cluster == True:
        str_evol = ""
        str_disorder = ""
        str_noise = ""
        
    # -> Assign appropriate string designator if time override is active
    if flag_timeoverride == True:
        str_timeoverride = "_T"
    else:
        str_timeoverride = ""
        
    # Build text strings corresponding to numerical protocol parameters
    str_protocol = ""
    # Omit certain keys from dict_textstr depending on the string/boolean designations
    # -> Tuning functions
    list_omit_textstr_tuning = []
    if str_tuning_choice in ["LIN", "SMOOTH", "FQD", "SHARP"]:
        list_omit_textstr_tuning = ["s_time_init", "s_time_mid", "slope_mid", "scale_factor"]
    if str_tuning_choice in ["TRIB", "TRIC", "SNP", "SNPB"]:
        list_omit_textstr_tuning = ["slope_mid", "scale_factor"]
    if str_tuning_choice in ["TRIE"]:
        list_omit_textstr_tuning = ["slope_mid"]
    # -> Disorder
    list_omit_textstr_disorder = []
    if str_disorder_choice == "NONE":
        list_omit_textstr_disorder = ["disorder_ratio", "length_corr"]
    if str_disorder_choice in ["UCUN", "UCND", "UCUNB"]:
        list_omit_textstr_disorder = ["length_corr"]
    # -> Noise
    list_omit_textstr_noise = []
    if str_noise_choice == "NONE":
        list_omit_textstr_noise = ["noise_ratio", "noise_psd_ampl", "w_cutoff_low", "w_cutoff_high", "freq_sin", "phase_sin"]
    if str_noise_choice in ["WHT", "WHTA"]:
        list_omit_textstr_noise = ["noise_psd_ampl", "w_cutoff_low", "w_cutoff_high", "freq_sin", "phase_sin"]
    if str_noise_choice in ["1F1", "1F1A", "WHTC"]:
        list_omit_textstr_noise  = ["noise_psd_ampl", "freq_sin", "phase_sin"]
    if str_noise_choice in ["1F1B", "WHTCB"]:
        list_omit_textstr_noise  = ["noise_ratio", "freq_sin", "phase_sin"]
    if str_noise_choice == "SIN":
        list_omit_textstr_noise  = ["noise_psd_ampl", "w_cutoff_low", "w_cutoff_high"]
    if str_noise_choice == "SINAV":
        list_omit_textstr_noise  = ["noise_psd_ampl", "w_cutoff_low", "w_cutoff_high", "phase_sin"]
    # Combine the exemption lists
    list_omit_textstr = list_omit_textstr_tuning + list_omit_textstr_disorder + list_omit_textstr_noise
    
    # List of numerical values of relevant parameters
    arr_value_par = []
        
    # Build text strings
    for key in dict_textstr.keys():
        if key in list_omit_textstr:
            continue
        value_par = dict_params_rel[key]
        arr_value_par.append(value_par)
        str_value_par = str(value_par).replace('.', '')
        str_par = "_" + dict_textstr[key] + str_value_par
        str_protocol = str_protocol + str_par
        
    # Complete text string of parameters and designations
    str_params = str_desig + str_evol + str_timeoverride + str_disorder + str_noise + str_NSamp + str_protocol
    
    # Turn list of numerical values into an array
    arr_value_par = np.array(arr_value_par)

    return str_params, arr_value_par

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
    list_str_cols = ["timeoverride", "L", "R", "Delta", "w", "muLeft", "muRightStart", "muRightEnd", "lp", "n_steps", "tau"]
    
    # -> Add columns depending on tuning function choice
    str_tuning_choice = dict_specifications["tuning"]
    if str_tuning_choice in ["TRI", "TRID", "TRID2"]:
        list_str_cols.extend(["s_time_init", "s_time_mid", "slope_mid"])
    if str_tuning_choice in ["TRIB", "TRIC", "SNP", "SNPB"]:
        list_str_cols.extend(["s_time_init", "s_time_mid"])
    if str_tuning_choice in ["TRIE"]:
        list_str_cols.extend(["s_time_init", "s_time_mid", "scale_factor"])
    # -> Add columns depending on disorder choice
    str_disorder_choice = dict_specifications["disorder"]
    if str_disorder_choice in ["UCUN", "UCND", "UCUNB"]:
        list_str_cols.append("disorder_ratio")
    if str_disorder_choice in ["GCUN", "GCND", "GCNDB", "ECND", "ECNDB", "SCND", "SCNDB"]:       
        list_str_cols.append("disorder_ratio")
        list_str_cols.append("length_corr")
        
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
def create_csvcolumns_data(dict_operations_fullsim, dict_fullsim, flag_avg=False):
    
    # Unpack relevant parameters
    list_ind_transProb = dict_fullsim["transProbs"]
    list_ind_transProb_cc = dict_fullsim["transProbs_cc"]
    
    # Establish base-line, data columns
    list_str_cols = []

    # -> Add columns depending on data choice
    if dict_operations_fullsim["diabErr"] == True:
        list_str_cols.append("diabErr")
    if dict_operations_fullsim["diabErr_cc"] == True:
        list_str_cols.append("diabErr_cc")
    if dict_operations_fullsim["diabErr_ct"] == True:
        list_str_cols.append("diabErr_ct")
    if dict_operations_fullsim["transProb"] == True:
        for ele in list_ind_transProb:
            ele.sort()
            str_ele = list(map(str, ele))
            list_str_cols.append("transProb_" + "_".join(str_ele))
    if dict_operations_fullsim["transProb_cc"] == True:
        for ele in list_ind_transProb_cc:
            ele.sort()
            str_ele = list(map(str, ele))
            list_str_cols.append("transProb_cc_" + "_".join(str_ele))
                
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

# Establish data (OTHER) columns of full simulation .csv files
def create_csvcolumns_data_other(dict_operations_fullsim, flag_avg=False):
    
    # Establish base-line, data columns
    list_str_cols = []

    # -> Add columns depending on data choice
    if dict_operations_fullsim["mingap"] == True:
        list_str_cols.append("mingap")
        list_str_cols.append("mingap_loc")
    if dict_operations_fullsim["velocity"] == True:
        list_str_cols.append("velocity")
    if dict_operations_fullsim["mingap_sep"] == True:
        list_str_cols.append("mingap")
                
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
def create_csvcolumns_data_onesim(dict_operations_onesim, dict_onesim, flag_avg=False):
    
    # Unpack relevant parameters
    list_ind_exptEnerg = dict_onesim["exptEnerg"]
    list_ind_occupation = dict_onesim["occupation"]
    list_ind_transProb = dict_onesim["transProbs"]
    list_ind_transProb_cc = dict_onesim["transProbs_cc"]
    
    # Establish base-line, data columns
    list_str_cols = []

    # -> Add columns depending on data choice
    if dict_operations_onesim["diabErr"] == True:
        list_str_cols.append("diabErr")
    if dict_operations_onesim["diabErr_cc"] == True:
        list_str_cols.append("diabErr_cc")
    if dict_operations_onesim["diabErr_ct"] == True:
        list_str_cols.append("diabErr_ct")
    if dict_operations_onesim["matElement"] == True:
        list_str_cols.append("matElement")
    if dict_operations_onesim["exptEnergy"] == True:
        for index in list_ind_exptEnerg:
            list_str_cols.append("exptEnergy_{}".format(index))
    if dict_operations_onesim["occupation"] == True:
        for index in list_ind_occupation:
            list_str_cols.append("occupation_{}".format(index))
    if dict_operations_onesim["transProbs"] == True:
        for ele in list_ind_transProb:
            ele.sort()
            str_ele = list(map(str, ele))
            list_str_cols.append("transProbs_" + "_".join(str_ele))
    if dict_operations_onesim["transProbs_cc"] == True:
        for ele in list_ind_transProb_cc:
            ele.sort()
            str_ele = list(map(str, ele))
            list_str_cols.append("transProbs_cc_" + "_".join(str_ele))
    
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
def create_csvfile(dict_specifications, dict_operations_fullsim, dict_fullsim, dirname, filename, flag_avg=False, flag_desig=False):

    # Establish non-data columns in .csv
    list_str_cols = create_csvcolumns_nodata(dict_specifications, flag_desig=flag_desig)
        
    # Establish data columns in .csv
    list_str_cols_data = create_csvcolumns_data(dict_operations_fullsim, dict_fullsim, flag_avg=flag_avg)

    # Check that folders exist; if a folder does not exist, create it
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    # Check that .csv files exists in folder; if a file does not exist, create it
    if os.path.exists("{}/{}.csv".format(dirname, filename)) == False:
        df_empty = pd.DataFrame(columns=list_str_cols + list_str_cols_data)
        # Save to file
        df_empty.to_csv("{}/{}.csv".format(dirname, filename), index=False)
        
# Creates a .csv file where data (OTHER) may be stored for full simulations
def create_csvfile_other(dict_specifications, dict_operations_fullsim, dirname, filename, flag_avg=False, flag_desig=False):

    # Establish non-data columns in .csv
    list_str_cols = create_csvcolumns_nodata(dict_specifications, flag_desig=flag_desig)
    
    # Remove columns that are on an exceptions list
    exception_nodata = ["tau"]
    list_str_cols = [cols for cols in list_str_cols if cols not in exception_nodata]
        
    # Establish data columns in .csv
    list_str_cols_data = create_csvcolumns_data_other(dict_operations_fullsim, flag_avg=flag_avg)

    # Check that folders exist; if a folder does not exist, create it
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    # Check that .csv files exists in folder; if a file does not exist, create it
    if os.path.exists("{}/{}.csv".format(dirname, filename)) == False:
        df_empty = pd.DataFrame(columns=list_str_cols + list_str_cols_data)
        # Save to file
        df_empty.to_csv("{}/{}.csv".format(dirname, filename), index=False)
        
# Creates a .csv file where data may be stored for single simulations
def create_csvfile_onesim(dict_operations_onesim, dict_onesim, dirname, filename, flag_avg=False, flag_desig=False):

    # Establish non-data columns in .csv
    list_str_cols = create_csvcolumns_nodata_onesim(flag_desig=flag_desig)

    # Establish data columns in .csv
    list_str_cols_data = create_csvcolumns_data_onesim(dict_operations_onesim, dict_onesim, flag_avg=flag_avg)

    # Check that folders exist; if a folder does not exist, create it
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    # Check that .csv files exists in folder; if a file does not exist, create it
    if os.path.exists("{}/{}.csv".format(dirname, filename)) == False:
        df_empty = pd.DataFrame(columns=list_str_cols + list_str_cols_data)
        # Save to file
        df_empty.to_csv("{}/{}.csv".format(dirname, filename), index=False)

# Create a multi-index that labels data in a .csv for full simulations
def create_csvmultiIndex(dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, flag_desig=False):
    
    # Establish non-data columns that are present in .csv which we use as the names of the multi-index
    list_str_cols = create_csvcolumns_nodata(dict_specifications, flag_desig=flag_desig)
    
    # Dictionary of relevant parameters
    dict_params_rel = {}
    dict_params_rel.update(dict_protocol)
    dict_params_rel.update(dict_disorder)
    dict_params_rel.update(dict_noise)
        
    # Create the multi-index as a list of lists
    index = []
    index.append([dict_operations_other["timeoverride"]])
    for name in list_str_cols:
        if name not in ["timeoverride"]:
            index.append([dict_params_rel[name]])
    multiIndex = pd.MultiIndex.from_arrays(index, names=tuple(list_str_cols))
    
    return multiIndex
        
# Generates a dictionary containing the column name (of data frame):value
def dict_paramset(dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_desig=False):
    
    # Unpack relevant parameters
    # -> Operations, other
    flag_timeoverride = dict_operations_other["timeoverride"]
    # -> Specifications
    str_tuning_choice = dict_specifications["tuning"]
    str_disorder_choice, str_noise_choice = dict_specifications["disorder"], dict_specifications["noise"]
    # -> Protocol
    L, R, Delta, w, muLeft, muRightStart, muRightEnd, lp, n_steps, tau = dict_protocol["L"], dict_protocol["R"], dict_protocol["Delta"], dict_protocol["w"], dict_protocol["muLeft"], dict_protocol["muRightStart"], dict_protocol["muRightEnd"], dict_protocol["lp"], dict_protocol["n_steps"], dict_protocol["tau"]
    # -> Disorder
    disorder_ratio, length_corr = dict_disorder["disorder_ratio"], dict_disorder["length_corr"]
    # -> Noise
    noise_ratio, noise_psd_ampl, w_cutoff_low, w_cutoff_high, freq_sin, phase_sin = dict_noise["noise_ratio"], dict_noise["noise_psd_ampl"], dict_noise["w_cutoff_low"], dict_noise["w_cutoff_high"], dict_noise["freq_sin"], dict_noise["phase_sin"]
    # -> Other
    str_desig = dict_other["str_desig"]
    
    # Establish variables to save to .csv
    dict_fileparams = {"timeoverride":[flag_timeoverride], "L":[L], "R":[R], "Delta":[Delta], "w":[w], "muLeft":[muLeft], "muRightStart":[muRightStart], "muRightEnd":[muRightEnd], "lp":[lp], "n_steps":[n_steps], "tau":[tau]}
    
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
    
    # -> Disorder
    if str_disorder_choice in ["UCUN", "UCND", "UCUNA", "UCUNB"]:
        dict_fileparams["disorder_ratio"] = [disorder_ratio]
    if str_disorder_choice in ["GCUN", "GCND", "GCNDB", "ECND", "ECNDB", "SCND", "SCNDB"]:
        dict_fileparams["disorder_ratio"] = [disorder_ratio]
        dict_fileparams["length_corr"] = [length_corr]
        
    # -> Noise parameters 
    if str_noise_choice in ["WHT", "WHTA"]:
        dict_fileparams["noise_ratio"] = [noise_ratio]
    if str_noise_choice in ["1F1", "1F1A", "WHTC"]:
        dict_fileparams["noise_ratio"] = [noise_ratio]
        dict_fileparams["w_cutoff_low"] = [w_cutoff_low]
        dict_fileparams["w_cutoff_high"] = [w_cutoff_high]
    if str_noise_choice in ["WHTCB", "1F1B"]:
        dict_fileparams["noise_psd_ampl"] = [noise_psd_ampl]
        dict_fileparams["w_cutoff_low"] = [w_cutoff_low]
        dict_fileparams["w_cutoff_high"] = [w_cutoff_high]
    if str_noise_choice == "SIN":
        dict_fileparams["noise_ratio"] = [noise_ratio]
        dict_fileparams["freq_sin"] = [freq_sin]
        dict_fileparams["phase_sin"] = [phase_sin]
    if str_noise_choice == "SINAV":
        dict_fileparams["noise_ratio"] = [noise_ratio]
        dict_fileparams["freq_sin"] = [freq_sin]
        
    if flag_desig == True:
        dict_fileparams["str_desig"] = [str_desig]
        
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
# Common #
# ------ #

# Average of averages
# -> Note that nbr_a and nbr_b must be integers; avg_a and avg_b are allowed to be arrays
def avg_averages(nbr_a, nbr_b, avg_a, avg_b):
    #return (nbr_a*avg_a + nbr_b*avg_b)/(nbr_a + nbr_b)
    return np.divide(nbr_a*avg_a + nbr_b*avg_b, nbr_a + nbr_b)

# Calculates a running average of the diabatic error (or, in general, any quantity that requires averaging)
# -> Inputs: new diabatic error value; dictionary containing diabatic error average, square-average, log-average, and square-log-average; number of realizations that these averages represent
# -> Output: averages but with the new diabatic error value included, number of realizations + 1 
def average_updater(dict_diabErr_avgs_1, dict_diabErr_avgs_2, nbr_realizations_avg_1, nbr_realizations_avg_2):
    
    # Averages and square averages
    diabErr_avg_upd = ((nbr_realizations_avg_1)*dict_diabErr_avgs_1["avg"] + (nbr_realizations_avg_2)*dict_diabErr_avgs_2["avg"])/(nbr_realizations_avg_1 + nbr_realizations_avg_2)
    diabErr_avg_sqr_upd = ((nbr_realizations_avg_1)*dict_diabErr_avgs_1["avg_sqr"] + (nbr_realizations_avg_2)*dict_diabErr_avgs_2["avg_sqr"])/(nbr_realizations_avg_1 + nbr_realizations_avg_2)
    
    # Log averages and square of log averages
    diabErr_logAvg_upd = ((nbr_realizations_avg_1)*dict_diabErr_avgs_1["logAvg"] + (nbr_realizations_avg_2)*dict_diabErr_avgs_2["logAvg"])/(nbr_realizations_avg_1 + nbr_realizations_avg_2)
    diabErr_logAvg_sqr_upd = ((nbr_realizations_avg_1)*dict_diabErr_avgs_1["logAvg_sqr"] + (nbr_realizations_avg_2)*dict_diabErr_avgs_2["logAvg_sqr"])/(nbr_realizations_avg_1 + nbr_realizations_avg_2)
    
    return {"avg":diabErr_avg_upd, "avg_sqr":diabErr_avg_sqr_upd, "logAvg":diabErr_logAvg_upd, "logAvg_sqr":diabErr_logAvg_sqr_upd}, nbr_realizations_avg_1 + nbr_realizations_avg_2

# Ensures that x- and y- data are equally spaced. This is done by comparing the x-data to some reference array
def data_equalizer(x_data, y_data, arr_ref):
    ind_x = [np.argmin(np.abs(arr_ref[x] - x_data)) for x in range(0, arr_ref.shape[0])]
    x_data_new = x_data[ind_x]
    y_data_new = y_data[ind_x]
    return x_data_new, y_data_new

# Format values to scientific notation for labels in plots
def fmt_scinotn(val):
    #float_str = "{0:.2g}".format(val)
    float_str = "{:.1e}".format(val)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

# Generate title and legend text strings for plots
def str_plot(nbr_datasets, dict_specifications, dict_protocol, dict_disorder, dict_noise, vary_param_choice, dict_rescale_values={}, dict_rescale_symbols={}, flag_noNmin=False, flag_disorder_replace_var=False, flag_disorder_replace_std=False, flag_sciNotation=False):
    
    # Save minimum number of realizations
    nbr_realizations_min = dict_protocol["nbr_realizations_min"]
    
    # Create a dictionaries with entries parameter:label
    dict_all = dict_specifications
    dict_all.update(dict_protocol)
    dict_all.update(dict_disorder)
    dict_all.update(dict_noise)
    del dict_all["nbr_realizations_min"]
    #dict_all["s_time_init"] = dict_all["s_time_init"]*dict_all["tau"]
    #dict_all["s_time_mid"] = dict_all["s_time_mid"]*dict_all["tau"]
    
    list_keys = list(dict_all.keys())
    list_strs_protocol = [r"{}", r"{}", r"{}", r"$L = {}$", r"$R = {}$", r"$\Delta = {}$", r"$w = {}$", r"$\mu_{{\mathrm{{L}}}} = {}$", r"$\mu_{{\mathrm{{R,s}}}} = {}$", r"$\mu_{{\mathrm{{R,e}}}} = {}$", r"$l_{{\mathrm{{p}}}} = {}$", r"$n = {}$", r"$\tau = {:.2f}$", r"$\tau_{{\mathrm{{i}}}}/\tau = {:.3f}$", r"$\tau_{{\mathrm{{m}}}}/\tau = {:.3f}$", r"$\alpha = {:.3f}$", r"$f_{{s}} = {:.3f}$"]
    #list_strs_protocol = [r"{}", r"{}", r"{}", r"$L = {}$", r"$R = {}$", r"$\Delta = {}$", r"$w = {}$", r"$\mu_{{\mathrm{{L}}}} = {}$", r"$\mu_{{\mathrm{{R,s}}}} = {}$", r"$\mu_{{\mathrm{{R,e}}}} = {}$", r"$l_{{\mathrm{{p}}}} = {}$", r"$n = {}$", r"$\tau = {:.3f}$", r"$\tau_{{\mathrm{{i}}}} = {:.2f}$", r"$\tau_{{\mathrm{{m}}}} = {:.2f}$", r"$\alpha = {}$"]
    #list_strs_protocol = [r"{}", r"{}", r"{}", r"$L = {}$", r"$R = {}$", r"$\Delta = {}$", r"$w = {}$", r"$\mu_{{\mathrm{{L}}}} = {}$", r"$\mu_{{\mathrm{{R,s}}}} = {}$", r"$\mu_{{\mathrm{{R,e}}}} = {}$", r"$l_{{\mathrm{{p}}}} = {}$", r"$n = {}$", r"$\tau/\tau_{{0,K}} = {}$"]
    list_strs_disorder = [r"$r_{{\mathrm{{d}}}} = {:.3f}$", r"$\xi = {:.1f}$"] 
    #list_strs_noise = [r"$r_{{\mathrm{{n}}}} = {}$", r"$A_{{\mathrm{{psd}}}} = {}$", r"$\omega_{{\mathrm{{l}}}} = {}$", r"$\omega_{{\mathrm{{h}}}} = {}$", r"$\omega = {:.3f}$", r"$\phi = {}$"]
    #list_strs_noise = [r"$r_{{\mathrm{{n}}}} = {}$", r"$A = {}$", r"$\omega_{{\mathrm{{l}}}} = {}$", r"$\omega_{{\mathrm{{h}}}} = {:.1f}$", r"$\omega = {:.3f}$", r"$\phi = {}$"]
    list_strs_noise = [r"$r_{{\mathrm{{n}}}} = {}$", r"$A_{{\mathrm{{w}}}} = {}$", r"$\omega_{{\mathrm{{l}}}} = {}$", r"$\omega_{{\mathrm{{h}}}} = {:.1f}$", r"$\omega = {:.1f}$", r"$\phi = {}$"]
    list_strs = list_strs_protocol + list_strs_disorder + list_strs_noise
    
    # Convert all tau's to real units
    #hbar = (6.582e-16)*1e3 # hbar, in units of meV s
    #tau_rescale = 1/hbar/1e9
    #dict_all["tau"] = dict_all["tau"]/tau_rescale
    #dict_all_str["tau"] = dict_all_str["tau"] + " ns"
    
    # If replacing disorder ratio with disorder variance is desired
    if flag_disorder_replace_var == True:
        del dict_all["disorder_ratio"]
        list_keys.remove("disorder_ratio")
        list_strs = [string.replace(r"$r_{{\mathrm{{d}}}} = {:.3f}$", r"$\sigma^2 = {:.3}$") for string in list_strs]
        if vary_param_choice == "DR":
            vary_param_choice = "DV"
            
    # If replacing disorder ratio with disorder standard deviation is desired
    if flag_disorder_replace_std == True:
        del dict_all["disorder_ratio"]
        list_keys.remove("disorder_ratio")
        list_strs = [string.replace(r"$r_{{\mathrm{{d}}}} = {:.3f}$", r"$\sigma = {:.2}$") for string in list_strs]
        if vary_param_choice == "DR":
            vary_param_choice = "DS"
        
    # Dictionary which connects the list of keys to the list of strings
    dict_all_str = dict(zip(list_keys, list_strs))
    
    # Rescale parameters if desired; adjust symbols accordingly
    if len(dict_rescale_values) != 0:
        for key in dict_rescale_values.keys():
            key_act = key
            if key == "disorder_ratio":
                if flag_disorder_replace_var == True:
                    key_act = "disorder_variance"
                elif flag_disorder_replace_std == True:
                    key_act = "disorder_std"
            dict_all[key_act] = np.divide(dict_all[key_act], dict_rescale_values[key])
            dict_all_str[key_act] = dict_all_str[key_act][:dict_all_str[key_act].rfind("=")-1] + "/" + dict_rescale_symbols[key] + dict_all_str[key_act][dict_all_str[key_act].rfind("=")-1:]
    
    # Omit the parameter corresponding to vary_param_choice
    if vary_param_choice == "TAU":
        del dict_all["tau"]
        del dict_all_str["tau"]
        
    if vary_param_choice == "DR":
        del dict_all["disorder_ratio"]
        del dict_all_str["disorder_ratio"]
        
    if vary_param_choice == "DV":
        del dict_all["disorder_variance"]
        del dict_all_str["disorder_variance"]
        
    if vary_param_choice == "DS":
        del dict_all["disorder_std"]
        del dict_all_str["disorder_std"]
        
    if vary_param_choice == "LC":
        del dict_all["length_corr"]
        del dict_all_str["length_corr"]
    
    if vary_param_choice == "NR":
        del dict_all["noise_ratio"]
        del dict_all_str["noise_ratio"]
        
    if vary_param_choice == "NA":
        del dict_all["noise_psd_ampl"]
        del dict_all_str["noise_psd_ampl"]
        
    if vary_param_choice == "WCH":
        del dict_all["w_cutoff_high"]
        del dict_all_str["w_cutoff_high"]
        
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
        
    # Omit certain parameters depending on str_tuning_choice
    list_tuningtype_tri_range = ["TRI", "TRIB", "TRIC", "TRID", "TRID2", "TRIE", "SNP", "SNPB"] # Keep STIMES, STIMESM
    list_tuningtype_tri_slope = ["TRI", "TRID", "TRID2"] # Keep SLOPE
    list_tuningtype_tri_factor = ["TRIE"] # Keep scale factor
    
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
    
    # Omit certain parameters depending on str_disorder_choice
    list_disordertype_dr = ["UCUN", "UCND", "UCUNA", "UCUNB", "GCND", "GCNDB", "ECND", "ECNDB", "SCND", "SCNDB"] # Keep disorder_ratio
    list_disordertype_xi = ["GCUN", "GCND", "GCNDB", "ECND", "ECNDB", "SCND", "SCNDB"] # Keep correlation/decay length

    if any(x in dict_all["disorder"] for x in list_disordertype_dr) == False:
        del dict_all["disorder_ratio"]
    
    if any(x in dict_all["disorder"] for x in list_disordertype_xi) == False:
        del dict_all["length_corr"] 
    
    # Omit certain parameters depending on str_noise_choice
    list_noisetype_nr = ["WHT", "WHTC", "1F1", "SIN", "SINAV"] # Keep noise_ratio
    list_noisetype_ampl = ["WHTCB", "1F1B"] # Keep amplitude
    list_noisetype_cutoffs = ["WHTC", "WHTCB", "1F1", "1F1B"] # Keep cutoffs
    list_noisetype_freq = ["SIN", "SINAV"] # Keep frequency
    list_noisetype_phase = ["SIN"] # Keep phase

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
                if flag_sciNotation == True:
                    str_key_param = dict_all_str[key].format(fmt_scinotn(dict_all[key][i]))
                else:
                    str_key_param = dict_all_str[key].format(dict_all[key][i])
                list_str_legend[i].append(str_key_param)
    
    # Form plot title
    str_plotTitle = ", ".join(list_str_title) + ", $N_{{\mathrm{{min}}}} = {}$".format(nbr_realizations_min)
    
    if flag_noNmin == True:
        str_plotTitle = str_plotTitle.replace(", $N_{{\mathrm{{min}}}} = {}$".format(nbr_realizations_min), "")
        
    # Add a line break after a specified number of characters
    char_break = 150
    char_total = len(str_plotTitle)
    str_plotTitle_temp = ""
    ind_min = 0
    ind_max = char_break
    while ind_min <= char_total:
        str_plotTitle_part = str_plotTitle[ind_min:ind_max]
        if ind_max >= char_total:
            ind_separate = str_plotTitle_part.rfind("$")
            str_rep = "$"
        else:
            ind_separate = str_plotTitle_part.rfind("$,")
            str_rep = "$,\n"
        str_plotTitle_part = str_plotTitle_part[:ind_separate] + str_rep
        str_plotTitle_temp = str_plotTitle_temp + str_plotTitle_part
        
        ind_min = ind_min + ind_separate + 3
        ind_max = ind_max + ind_separate + 3
        
    str_plotTitle = str_plotTitle_temp
    
    """
    # Add a line break as close as possible to half-way through str_plotTitle
    # -> Use commas to find index of split
    loc_commas = [x for x in range(len(str_plotTitle)) if str_plotTitle.startswith(", ", x)]
    ind_loc_comma_half = np.argmin(np.abs(len(str_plotTitle)//7.0 - np.array(loc_commas)))
    # -> Add a line break to string
    str_plotTitle = str_plotTitle[:loc_commas[ind_loc_comma_half]+1] + "\n" + str_plotTitle[loc_commas[ind_loc_comma_half]+1:]
    """
    
    return str_plotTitle, list_str_legend    

# Load two-level system data
# -> Only one set of data is loaded corresponding to the first set of parameters
def load_tls(dict_specifications_all_mb, dict_protocol_all_mb, dict_disorder_all_mb, dict_noise_all_mb, vary_param_choice, dirname, filename_manual, dataType, flag_filemanual=False):
    
    # List of noise protocols where averaging is not present
    list_noise_noAvg = ["NONE", "SIN"]
    
    # ------------ #
    # Dictionaries #
    # ------------ #
    
    # Specifications
    str_tuning_choice = [dict_specifications_all_mb["tuning"][0]]
    str_noise_choice = [dict_specifications_all_mb["noise"][0]]
    
    # Protocol
    delta_mu = [np.round(0.5*np.abs(dict_protocol_all_mb["muRightStart"][0] - dict_protocol_all_mb["muRightEnd"][0]), 6)]
    Delta_SC = [dict_protocol_all_mb["Delta"][0]]
    lp = [dict_protocol_all_mb["lp"][0]]
    tau = [dict_protocol_all_mb["tau"][0]]
    s_time_init = [dict_protocol_all_mb["s_time_init"][0]]
    s_time_mid = [dict_protocol_all_mb["s_time_mid"][0]]
    slope_mid = [dict_protocol_all_mb["slope_mid"][0]]
    scale_factor = [dict_protocol_all_mb["scale_factor"][0]]
    noise_ratio = [dict_noise_all_mb["noise_ratio"][0]]
    nbr_realizations_min = dict_protocol_all_mb["nbr_realizations_min"]
    noise_psd_ampl = [dict_noise_all_mb["noise_psd_ampl"][0]]
    w_cutoff_low = [dict_noise_all_mb["w_cutoff_low"][0]]
    w_cutoff_high = [dict_noise_all_mb["w_cutoff_high"][0]]
    freq_sin = [dict_noise_all_mb["freq_sin"][0]]
    phase_sin = [dict_noise_all_mb["phase_sin"][0]]
    
    dict_specifications_all = {"tuning":str_tuning_choice, "noise":str_noise_choice}
    dict_protocol_all = {"delta_mu":delta_mu, "Delta_SC":Delta_SC, "lp":lp, "tau":tau, "s_time_init":s_time_init, "s_time_mid":s_time_mid, "slope_mid":slope_mid, "scale_factor":scale_factor, "noise_ratio":noise_ratio, "nbr_realizations_min":nbr_realizations_min, "noise_psd_ampl":noise_psd_ampl, "w_cutoff_low":w_cutoff_low, "w_cutoff_high":w_cutoff_high, "freq_sin":freq_sin, "phase_sin":phase_sin}
    
    #print(dict_specifications_all)

    # Dictionaries for varying parameters
    dict_param_choice = {"TAU":"tau", "STIMES":"s_time_init",  "STIMEM":"s_time_mid",  "SLOPE":"slope_mid", "NR":"noise_ratio", "NA":"noise_psd_ampl", "WCL":"w_cutoff_low", "WCH":"w_cutoff_high", "W":"freq_sin", "PH":"phase_sin"}

    # Adjust specifications and parameters based on number of data sets
    nbr_datasets = 1
    keys_exception = ["nbr_realizations_min"]
    keys_noadjust = []
    for dict_all in [dict_specifications_all, dict_protocol_all]:
        for key in dict_all.keys():
            if key in keys_exception:
                continue
            if len(dict_all[key]) > 1:
                nbr_datasets = len(dict_all[key])
                keys_noadjust.append(key)

    for key in dict_specifications_all.keys():
        if key in keys_exception or key in keys_noadjust:
            continue
        dict_specifications_all[key] = [dict_specifications_all[key][0]]*nbr_datasets
        
    for key in dict_protocol_all.keys():
        if key in keys_exception or key in keys_noadjust:
            continue
        dict_protocol_all[key] = np.full(nbr_datasets, dict_protocol_all[key][0])

    # ------------------------- #
    # Main data file management #
    # ------------------------- #

    list_str_main_core = []
    # Loop through all data sets and construct the core strings of the main data files in the format TUNING_NOISE
    for ind_data in range(0, nbr_datasets):
        str_main_core = dict_specifications_all["tuning"][ind_data] + "_" + dict_specifications_all["noise"][ind_data]
        if str_main_core in list_str_main_core:
            continue
        else:
            list_str_main_core.append(str_main_core)
            
    # Loop through all files in data_main, identify all those that correspond to list_strmain_core, and load the most updated ones
    dict_data_filenames = {}

    if flag_filemanual == False:
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
    
    count_clean = 0
    # Loop over number of data sets
    for ind_data in range(0, nbr_datasets + count_clean):
        
        ind_data_act = ind_data
        df_data_main = dict_data_files[dict_specifications_all["tuning"][ind_data_act] + "_" + dict_specifications_all["noise"][ind_data_act]]
        df_main = df_data_main
        
        # Establish dictionaries    
        dict_operations = {"cluster":False, "onesim":False, "average":False, "savedata":False, "Nfix":False, "timeoverride":False}
        dict_specifications = {key:dict_specifications_all[key][ind_data_act] for key in dict_specifications_all.keys()}
        dict_protocol = {key:dict_protocol_all[key][ind_data_act] for key in dict_protocol_all.keys() if key not in keys_exception}
          
        # Establish parameter columns and set index of df_main to parameter columns
        cols_nodata = tl.create_csvcolumns_nodata(dict_specifications)    
        cols_nodata.remove(dict_param_choice[vary_param_choice])
        df_main.set_index(cols_nodata, inplace=True)
        
        # Establish the multi-index for parameters with varied parameter accounted for
        multiIndex_pars = tl.create_csvmultiIndex(dict_operations, dict_specifications, dict_protocol)
        multiIndex_pars = multiIndex_pars.droplevel([dict_param_choice[vary_param_choice]])
        
        # Find all data corresponding to multi-index with varied parameter accounted for; reset index
        df_data_set = df_main.loc[multiIndex_pars]
        df_data_set.reset_index(inplace=True)
        df_main.reset_index(inplace=True)
        
        # In the case of noise, keep only the rows of data where the number of realizations >= nbr_realizations_min
        if dict_specifications["noise"] not in list_noise_noAvg:
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
        
    return list_df_data_set

# ------ #
# System #
# ------ #

# Characteristic time for the Kitaev chain as predicted by Landau-Zener theory (see Bauer's paper)
# Returns the characteristic time
# mu_c_ is the critical value of chemical potential. For the Kitaev chain, this can be taken to be the hopping w
def tau_LZ(mu_c_, muRight_, R_, Delta_):
    return 2.0*np.abs(mu_c_ - muRight_)*(R_/(Delta_*np.pi))**2

# Convert tau/tauLZ into tau
def tauR_to_tau(tauR, lp, w, Delta, muRightEnd):
    tauLZ = tau_LZ(w, muRightEnd, lp, Delta)
    return tauR*tauLZ

# Convert tau into tau/tauLZ
def tau_to_tauR(tau, lp, w, Delta, muRightEnd):
    tauLZ = tau_LZ(w, muRightEnd, lp, Delta)
    return tau/tauLZ

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

# Construct single-particle Kitaev Hamiltonian in the standard Majorana basis
# Returns matrix for single-particle Kitaev Hamiltonian
# L_ is the number of fermionic sites in chain
# mu_, w_, Delta_ are 1D arrays of length L each
def ham_kit(L_, mu_, w_, Delta_):
    # Definitions
    H_kit_mat = np.zeros((2*L_, 2*L_), dtype='complex') # Matrix for Hamiltonian
    # Variables
    a = (1j/4.0)*(Delta_ - w_)
    b = (1j/4.0)*(Delta_ + w_)
    c = mu_/2.0
    # Loop over each chain site and fill elements of matrix (m,n) where n>m 
    for m in range(0, L_):
        H_kit_mat[2*m, (2*m)+1] = -1j*c[m] # Fill 'c' elements
        if m < L_-1:
            H_kit_mat[2*m, (2*m)+3] = a[m] # Fill 'a' elements
            H_kit_mat[(2*m)+1, (2*m)+2] = b[m] # Fill 'b' elements
    # Make the complete matrix by adding its own complex conjugate
    H_kit_mat = H_kit_mat + (H_kit_mat.T).conj()
    return H_kit_mat

# Transforms any single-particle operator in the standard Majorana basis into the standard electron basis
# Returns mat_maj_ in the standard electron basis
# op_maj_ is an operator in the standard Majorana basis
def maj_to_ee(mat_maj_):
    # Create the transformation matrix
    mat_transform_oneblock = 0.5*np.array([[1,1],[-1j,1j]])
    mat_transform = np.kron(np.eye(mat_maj_.shape[0]//2), mat_transform_oneblock)
    # Transform mat_maj_
    mat_e = mat_transform.conj().T @ mat_maj_ @ mat_transform
    return mat_e

# Diagonalize the Kitaev Hamiltonian in the Majorana basis
# -> Returns the sorted eigenvalues and eigenvectors
def diag_ham_kit(L, mu, w, Delta):
    # Create the Hamiltonian
    matH = ham_kit(L, mu, w, Delta)
    # Diagonalize
    eigvals, eigvecs = sc_linalg.eigh(matH)
    # Sort above in ascending order
    ind_sort_eig = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[ind_sort_eig])
    eigvecs = eigvecs[:,ind_sort_eig]
    return eigvals, eigvecs
    
# Calculate the transformation O which brings a skew-symmetric matrix A into a special block-diagonal form
# Input:
# -> Skew-symmetric matrix A
# Output:
# -> Orthogonal transformation O
# Notes:
# -> The calculation of O is based on the results of M. Wimmer, Algorithm 923 ... ACM Trans. Math. Softw. 38, 30 (2012), see Eq. A1 to A5 therein
def calc_matO(matA):
     
    # Define the permutation matrix 
    matPermute = np.zeros((matA.shape[0], matA.shape[0]), dtype='float')

    # -> Loop over rows of matPermute
    for i in range(0, matA.shape[0]):
        if i < matA.shape[0]//2:
            j = 2*i
        else:
            j = 2*i - matA.shape[0] + 1
        matPermute[j,i] = 1
        
    # Q unitary such that A = QTQ^T, T is tridiagonal matrix
    matT, matQ = pf.skew_tridiagonalize(matA)

    # Extract J from T = P(0 & J^T // -J & 0) P^T
    matJ = ((matPermute.T @ matT @ matPermute)[:matA.shape[0]//2, matA.shape[0]//2:]).T

    # Perform an singlular value decomposition on J
    matV, matSigma, matW = np.linalg.svd(matJ)

    # Construct the transformation O which block-diagonalizes A
    matO = matQ @ matPermute @ sc_linalg.block_diag(matW.T, matV) @ matPermute.T

    return matO

# Covariance matrix in the eigenbasis corresponding to an excited state which contains two quasiparticles, one of which corresponds to the (near)-zero Majorana modes
# Input: Left chain size (L), right chain size (R), array indicating which single particle states should be occupied (ind_occup)
# Output: Covariance matrix in the eigenbasis, both even and odd variant
def calc_matMO(L, R, ind_occup):
    
    # Occupation numbers in eigenbasis (default: all sp-states unoccupied)
    arr_occup = np.zeros(L+R, dtype=int)
    
    # Fill indicated sp-states using ind_occup 
    arr_occup[ind_occup] = 1
    
    # Create the covariance matrix for the ground state in the eigenbasis
    matMO = np.kron(np.eye(L+R), np.array([[0,1],[-1,0]]))
    
    # Create the covariance matrix for the excited state by flipping signs of blocks in matMO for the ground state
    # Note that the first occupation number corresponds to the last block, the second number corresponds to the second last block, etc
    for i in range(0, len(arr_occup)):
        if i == 0:
            if arr_occup[i] == 1:
                matMO[-2:,-2:] = -matMO[-2:,-2:]
            else:
                continue
        else:
            if arr_occup[i] == 1:
                matMO[-2-2*i:-2*i,-2-2*i:-2*i] = -matMO[-2-2*i:-2*i,-2-2*i:-2*i]
            else:
                continue
            
    # Create the covariance matrix of opposite parity by flipping the block corresponding to the Majorana zero modes
    matMO_pflip = np.copy(matMO)
    matMO_pflip[-2:,-2:] = -matMO_pflip[-2:,-2:]
    
    # Determine the parities and define matMO_even and matMO_odd accordingly
    parity_MO = (pf.pfaffian(matMO)).real
    parity_MO_pflip = (pf.pfaffian(matMO_pflip)).real
    
    if parity_MO > parity_MO_pflip:
        matMO_even, matMO_odd = matMO, matMO_pflip
    else:
        matMO_even, matMO_odd = matMO_pflip, matMO
        
    return matMO_even, matMO_odd

# Covariance matrix in the original basis corresponding to an excited state which contains two quasiparticles, one of which corresponds to the (near)-zero Majorana modes
# Input: covariance matrices (MO[0, 1]) in the eigenbasis, orthogonal transformation for A (O)
# Output: Covariance matrix in the original basis, both even and odd variant
def calc_matM(matMO, matO):
    
    # Compute the covariance matrices in the original basis
    matM = matO @ matMO[0] @ matO.T
    matM_pflip = matO @ matMO[1] @ matO.T

    # Determine the parities and define matM_even and matM_odd accordingly
    parity_M = (pf.pfaffian(matM)).real
    parity_M_pflip = (pf.pfaffian(matM_pflip)).real

    if parity_M > parity_M_pflip:
        matM_even, matM_odd = matM, matM_pflip
    else:
        matM_even, matM_odd = matM_pflip, matM
        
    return matM_even, matM_odd

# Chemical potential as a function of time
# Returns a 1D array for the chemical potential
# s_ is a 1D array for the dimensionless time index, taking values from 0 to 1
# muStart_ and muEnd_ are the starting and ending chemical potentials
def mu_stime(s_, muStart_, muEnd_):
    time_func = np.sin((np.pi*s_/2))**2
    #time_func = s_
    return (1 - time_func)*muStart_ + (time_func)*muEnd_

# Chemical potential as a function of time
# Returns a 1D array for the chemical potential
# s_ is a 1D array for the dimensionless time index, taking values from 0 to 1
# muStart_ and muEnd_ are the starting and ending chemical potentials
def mu_stime_lin(s_, muStart_, muEnd_):
    time_func = s_
    #time_func = s_
    return (1 - time_func)*muStart_ + (time_func)*muEnd_

# Chemical potential as a function of time (FQD)
# Returns a 1D array for the chemical potential
# s_ is a 1D array for the dimensionless time index, taking values from 0 to 1
# muStart_ and muEnd_ are the starting and ending chemical potentials
def mu_stime_fqd(s, tau, DeltaLZ, w, muStart, muEnd):
    tuningFunc = tuningFunc_fqd(s*tau, tau, DeltaLZ, w, muStart, muEnd)
    #time_func = s_
    return (1 - tuningFunc)*muStart + (tuningFunc)*muEnd

# Tuning function for FQD
def tuningFunc_fqd(t, tau, DeltaLZ, w, muStart, muEnd):
    Omega = 2*DeltaLZ
    e0i = muStart - w
    e0f = muEnd - w
    alphai = e0i/Omega**2/np.sqrt(e0i**2 + Omega**2)
    alphaf = e0f/Omega**2/np.sqrt(e0f**2 + Omega**2)
    delta = Omega/tau*(alphaf - alphai)
    epsilont = (Omega**3)*(delta*t/Omega + alphai)/np.sqrt(1 - (Omega**4)*(delta*t/Omega + alphai)**2)
    tuningFunc = (1/(muEnd - muStart))*(epsilont - muStart + w)
    return tuningFunc

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
    
    coeffCubic = sc_linalg.solve(matCubic, matVec)
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
    coeffCubic = sc_linalg.solve(matCubic, matVec)
    
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

# Load minimum gap data from file
def load_mingaps(dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, dirname_mingap_managed):        
    # Establish file name
    filename_mingap_managed, arr_params_mingap_managed = str_params({"cluster":False}, {"timeoverride":False}, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=False, flag_noTau=True)
    # Load data
    try:
        with np.load("{}/mingap{}.npz".format(dirname_mingap_managed, filename_mingap_managed), allow_pickle=True) as file:
            arr_mingaps_all = np.copy(file["data"])
            #print("{}/mingap{}.npz".format(dirname_mingap_managed, filename_mingap_managed))
    except:
        arr_mingaps_all = np.zeros(0)
    # Return
    return arr_mingaps_all

# Analytical expression for the diabatic error of a single piano key
# -> Sine tuning
def exprAnaly_diabErr_smooth(delta_mu, Delta_LZ, tau):
    r = delta_mu/(2.0*Delta_LZ)
    term_exp = np.exp(-Delta_LZ/r*tau)
    term_power = 6*(Delta_LZ*tau)**(-4)*(r**2)/(1+r**2)**4
    return term_exp + term_power

# Semi-analytical expression for the diabatic error of a single piano key incorporating actual minimum gap data
# -> Adapted for plotting script pk_plot_diabErr.py
# -> Includes two modes for using the actual minimum gap data:
# ---> ("dynamic") The minimum gaps are loaded each time a "non-core" parameter changes
# ---> ("static") The minimum gaps corresponding to "core" parameters is loaded once and used for all calculations
def expr_diabErr_minGapAct(dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, vary_param_choice, dict_param_choice, nbr_realizations, arr_varpar_mingap, dirname_mingap_managed, mode="dynamic"):
    
    # Arrays
    arr_diabErr_avg = np.zeros(arr_varpar_mingap.shape[0])
    arr_diabErr_log = np.zeros(arr_varpar_mingap.shape[0])
    
    # Static mode only: load minimum gap data once
    if mode == "static":
        filename_mingap_managed, arr_params_mingap_managed = str_params({"cluster":False}, {"timeoverride":False}, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=False, flag_noTau=True)
        try:
            with np.load("{}/mingap{}.npz".format(dirname_mingap_managed, filename_mingap_managed), allow_pickle=True) as file:
                arr_mingaps_all_static = np.copy(file["data"])
                #print("{}/mingap{}.npz".format(dirname_mingap_managed, filename_mingap_managed))
        except:
            arr_mingaps_all_static = np.zeros(0)
    
    # Loop over varying parameter
    for ind_varpar in range(0, arr_varpar_mingap.shape[0]):
        
        # Copies of dictionaries
        dict_protocol_mg = dict(dict_protocol)
        dict_disorder_mg = dict(dict_disorder)
        dict_noise_mg = dict(dict_noise)
        
        # Search for varying parameter in dictionaries and replace with array
        for dict_sel in [dict_protocol_mg, dict_disorder_mg, dict_noise_mg]:
            if dict_param_choice[vary_param_choice] in list(dict_sel.keys()):
                dict_sel[dict_param_choice[vary_param_choice]] = arr_varpar_mingap[ind_varpar]
            else:
                continue
    
        # Dynamic mode only: load minimum gap data
        if mode == "dynamic":
            filename_mingap_managed, arr_params_mingap_managed = str_params({"cluster":False}, {"timeoverride":False}, dict_specifications, dict_protocol_mg, dict_disorder_mg, dict_noise_mg, dict_other, flag_NSamp=False, flag_noTau=True)
            try:
                with np.load("{}/mingap{}.npz".format(dirname_mingap_managed, filename_mingap_managed), allow_pickle=True) as file:
                    arr_mingaps_all = np.copy(file["data"])
                    #print("{}/mingap{}.npz".format(dirname_mingap_managed, filename_mingap_managed))
            except:
                arr_mingaps_all = np.zeros(0)
                continue
        elif mode == "static":
            arr_mingaps_all = arr_mingaps_all_static
        else:
            arr_mingaps_all = np.zeros(0)
            
        # Parameters
        delta_mu_mg = 0.5*np.abs(dict_protocol_mg["muRightEnd"] - dict_protocol_mg["muRightStart"])
        tau_mg = dict_protocol_mg["tau"]
        
        if vary_param_choice == "TAU":
            tau_mg = arr_varpar_mingap[ind_varpar]
            
        # Dynamic mode only: scramble arr_mingaps_all
        if mode == "dynamic":
            np.random.shuffle(arr_mingaps_all)
        
        Delta_LZ_mg = arr_mingaps_all[:nbr_realizations]
        
        # Calculate averages and store
        arr_diabErr_avg[ind_varpar] = np.average(exprAnaly_diabErr_smooth(delta_mu_mg, Delta_LZ_mg, tau_mg))
        arr_diabErr_log[ind_varpar] = np.average(np.log(exprAnaly_diabErr_smooth(delta_mu_mg, Delta_LZ_mg, tau_mg))) 
        
    # Set all zero-values to nan, indicating that these values do not exist
    arr_diabErr_avg[arr_diabErr_avg==0.0] = np.nan
    arr_diabErr_log[arr_diabErr_log==0.0] = np.nan
    
    return arr_diabErr_avg, arr_diabErr_log

# Square modulus of two many-body states in terms of their covariance matrices
# Returns the square modulus of the overlap between both states
# Ma_ and Mb_ are the covariance matrices of both states - both covariances matrices must be of the same size
def sqrmod_overlap_covar(Ma_, Mb_):
    size_chain = Ma_.shape[0]/2
    return np.abs((2**(float(-size_chain))*pf.pfaffian(Ma_ + Mb_)))

# Calculate the expectation value of the number operator using Majorana eigenvectors
# Input:
# -> Covariance matrix of a state
# -> Inst. single-particle eigenvector (fermion basis) corresponding to the MZMs
# Output:
# -> Expectation value of number operator
def occupation_number_mzm(mat_covar, vec_MZM_1, vec_MZM_2):
    # Calculate the occupation number expectation value via eigvec_MZM_c* @ mat_covar @ eigvec_MZM_c
    nbr_occupation = 0.5 + 0.5*(vec_MZM_1 @ mat_covar @ vec_MZM_2)
    return np.real(nbr_occupation[0,0])

# Continuity of real/imaginary parts of eigenvectors
# Input: real/imag. parts of current eigenvector in fermionic basis, real/imag. parts of a previous eigenvector in fermionic basis (assume all normalized)
# Output: real/imagin. parts of eigenvector which are consistent in continuity and sign compared to those of previous eigenvector (assume all normalized)
def eigvec_reim_continuity(eigvec_real, eigvec_imag, eigvec_real_prev, eigvec_imag_prev):
    
    # Ensure continuity of current eigenvectors compared to previous by checking all combinations of overlaps
    # -> Overlaps
    overlap_RR = eigvec_real @ eigvec_real_prev
    overlap_RI = eigvec_real @ eigvec_imag_prev
    overlap_IR = eigvec_imag @ eigvec_real_prev
    overlap_II = eigvec_imag @ eigvec_imag_prev
            
    # -> If overlap RR is larger than RI, eigenvectors are already consistent and no exchange is necessary
    # -> Otherwise, the real and imaginary parts to swap places
    if np.abs(overlap_RR) >= np.abs(overlap_RI):
        eigvec_R_new = eigvec_real
        eigvec_I_new = eigvec_imag
    else:
        eigvec_R_new = eigvec_imag
        eigvec_I_new = eigvec_real 
        
    # Ensure sign continuity of eigenvectors
    # -> It is possible that the overlaps could produce a negative sign. When this happens, the eigenvector needs to be complex conjugated which translates into flipping the sign of the imaginary part  
    if np.abs(overlap_RR) >= np.abs(overlap_RI):
        if np.sign(overlap_RR)*np.sign(overlap_II) < 0.0:
            eigvec_I_new = -eigvec_I_new
    else:
        if np.sign(overlap_RI)*np.sign(overlap_IR) < 0.0:
            eigvec_R_new = -eigvec_R_new
        
    return eigvec_R_new, eigvec_I_new

# --------- #
# Protocols #
# --------- #

# Simulate a piano key press on a Kitaev chain
def simulate_pianokey(L, R, arr_w, arr_Delta, arr_mu, arr_mu_clean, site_pk_start, site_pk_end, ind_pk, arr_mu_pk_s, arr_muD, arr_muN_pk_s, dt, N_sim_n, NSamp, matU_prev, matM_init, matM_clean_init, dict_operations_main, dict_operations_onesim, dict_onesim, flag_firstkey=False, flag_SVD=False, flag_printprogress=False):
    
    # ---------------- #
    # Testing purposes #
    # ---------------- #
    
    """
    # Testing arr_mu
    arr_mu_t_test = np.zeros(L+R).reshape(1,-1)
    """
    
    # ------------- #
    # Preliminaries #
    # ------------- #
    
    dict_results_onesim = {}
    dict_results_onesim_spec = {}
    dict_results_onesim_spec_clean = {}
    dict_results_onesim_eigvecs = {}
    dict_results_onesim_evolvecs = {}
    
    # ----------------------- #
    # Time evolution operator #
    # ----------------------- #
    
    # Establish the time evolution operator from previous parts of protocol (this shouold be identity if at the beginning)
    matU = matU_prev
    
    # For the first key, include the initial time (t = 0)
    # For all other keys, exclude the initial time since it will already have been captured by the last key (i.e. the final time for the previous key is the initial time for the current key)
    if flag_firstkey == True:
        ind_init = 0
    else:
        ind_init = 1
        
    # Build the time evolution operator via discretization
    for ind_time in range(ind_init, N_sim_n+1):
        
        # Print ind_time
        if flag_printprogress == True:
            if (ind_time+1)%(N_sim_n//10) == 0:
                print("Step:", ind_time+1, "/", N_sim_n)
        
        # Change chemical potential on piano key
        arr_mu[site_pk_start:site_pk_end] = arr_mu_pk_s[ind_time] + arr_muN_pk_s[ind_time] + arr_muD[site_pk_start:site_pk_end]
        
        # Instantaneous single-particle Hamiltonian
        matH = ham_kit(L+R, arr_mu, arr_w, arr_Delta)
        
        # ''Cross sectional'' time evolution operator via matrix exponentiation
        matU_slice = sc_linalg.expm(-2*1j*dt*matH)
          
        # Update the total time evolution operator
        if ind_time != 0:
            matU = matU_slice @ matU
            #matTimeBack = matTimeBack @ matTime_slice
        
        # If desired, perform an SVD on the time evolution operator to remove errors
        if flag_SVD == True:
            unitary_measure = np.linalg.norm(matU.conj().T @ matU - np.identity(2*(L+R), dtype='complex'))
            #unitary_measure_B = np.linalg.norm(matUBack.conj().T @ matUBack - np.identity(2*(L+R), dtype='complex'))
            if unitary_measure > 1e-14:
                SVD = np.linalg.svd(matU)
                matU = SVD[0] @ SVD[2]
            """
            if unitary_measure_B > 1e-14:
                SVD_B = np.linalg.svd(matUBack)
                matUBack = SVD_B[0] @ SVD_B[2]
            """
            
        # ---------------------------- #
        # Single simulation management #
        # ---------------------------- #
        
        if dict_operations_main["onesim"] == True:
        
            # Calculate single simulation quantities only at certain intervals of t, depending on NSamp
            if ind_time%((N_sim_n+1)//NSamp) == 0 or ind_time == ind_init or ind_time == N_sim_n:
                flag_updateArr = True
            else:
                flag_updateArr = False
                
            # Calculate quantities when flag_updateArr == True
            if flag_updateArr == True:
                
                # ----------------------------- #
                # Set up intermediate variables #
                # ----------------------------- #
                
                # Dictionary of intermediate variables, arrays, vectors, etc
                #dict_intervars = {"eigvalH":0, "eigvecH":0, "matU_upd":0, "matM":0}
                dict_intervars = {}
                
                # Perform an SVD on the time evolution operator to remove errors
                try:
                    mats_SVD = np.linalg.svd(matU)
                    matU = mats_SVD[0] @ mats_SVD[2]
                except:
                    pass
                
                # Covariance matrix of instantaneous ground state of total Hamiltonian
                if True in [dict_operations_onesim["diabErr"], dict_operations_onesim["diabErr_ct"], dict_operations_onesim["occupation"]]:
                    
                    # Extract the real, anti-symmetric matrix A defined through H = iA
                    matA = (-1j*matH).real
                    
                    # Create covariance matrices for the many-body ground states (even and odd)
                    # -> Orthogonal transformations
                    matO = calc_matO(matA)
                    # -> Covariance matrix in the eigenbases
                    matMO = calc_matMO(L, R, [])
                    # -> Covariance matrices in the original basis
                    matM_even, matM_odd = calc_matM(matMO, matO)
                    
                    # -> Store only the even result
                    dict_intervars["matM"] = matM_even
                    
                # Covariance matrix of instantaneous ground state of clean Hamiltonian
                if True in [dict_operations_onesim["diabErr_cc"]]:
                    
                    # Establish Hamiltonian
                    arr_mu_clean[site_pk_start:site_pk_end] = arr_mu_pk_s[ind_time]
                    matH_clean = ham_kit(L+R, arr_mu_clean, arr_w, arr_Delta)
                    
                    # Extract the real, anti-symmetric matrix A defined through H = iA
                    matA_clean = (-1j*matH_clean).real
                    
                    # Create covariance matrices for the many-body ground states (even and odd)
                    # -> Orthogonal transformations
                    matO_clean = calc_matO(matA_clean)
                    # -> Covariance matrix in the eigenbases
                    matMO_clean = calc_matMO(L, R, [])
                    # -> Covariance matrices in the original basis
                    matM_even_clean, matM_odd_clean = calc_matM(matMO_clean, matO_clean)
                    
                    # -> Store only the even result
                    dict_intervars["matM_clean"] = matM_even_clean
                
                # Diagonalize the total Hamiltonian
                if True in [dict_operations_onesim["instEnergy"], dict_operations_onesim["instEigVec"], dict_operations_onesim["occupation"]]:
                    
                    # -> Diagonalize matH
                    eigvalH, eigvecH = sc_linalg.eigh(matH)
                    
                    # -> Sort the eigenvalues in ascending order; sort the eigenvectors accordingly
                    ind_eigvalH_sorted = np.argsort(np.real(eigvalH))
                    eigvalH = np.real(eigvalH[ind_eigvalH_sorted])
                    eigvecH = eigvecH[:,ind_eigvalH_sorted]
                    
                    # -> Store
                    dict_intervars["eigvalH"] = eigvalH
                    dict_intervars["eigvecH"] = eigvecH
                    
                # Diagonalize the clean Hamiltonian
                if True in [dict_operations_onesim["instEnergy_c"]]:
                    
                    # Establish Hamiltonian
                    arr_mu_clean[site_pk_start:site_pk_end] = arr_mu_pk_s[ind_time]
                    matH_clean = ham_kit(L+R, arr_mu_clean, arr_w, arr_Delta)
                    
                    # -> Diagonalize matH
                    eigvalH_clean, eigvecH_clean = sc_linalg.eigh(matH_clean)
                    
                    # -> Sort the eigenvalues in ascending order; sort the eigenvectors accordingly
                    ind_eigvalH_clean_sorted = np.argsort(np.real(eigvalH_clean))
                    eigvalH_clean = np.real(eigvalH_clean[ind_eigvalH_clean_sorted])
                    eigvecH_clean = eigvecH_clean[:,ind_eigvalH_clean_sorted]
                    
                    # -> Store
                    dict_intervars["eigvalH_clean"] = eigvalH_clean
                    dict_intervars["eigvecH_clean"] = eigvecH_clean
                    
                # -------------------- #
                # Calculate quantities #
                # -------------------- #
                
                # Time 
                time_t = np.round(ind_time*dt + ind_pk*(dt*N_sim_n), 8)
                if ind_time == ind_init:
                    dict_results_onesim["time"] = np.array([time_t])
                    dict_results_onesim_spec["time"] = np.array([time_t])
                    dict_results_onesim_spec_clean["time"] = np.array([time_t])
                    dict_results_onesim_eigvecs["time"] = np.array([time_t])
                    dict_results_onesim_evolvecs["time"] = np.array([time_t])
                else:
                    dict_results_onesim["time"] = np.append(dict_results_onesim["time"], time_t)
                    dict_results_onesim_spec["time"] = np.append(dict_results_onesim_spec["time"], time_t)
                    dict_results_onesim_spec_clean["time"] = np.append(dict_results_onesim_spec_clean["time"], time_t)
                    dict_results_onesim_eigvecs["time"] = np.append(dict_results_onesim_eigvecs["time"], time_t)
                    dict_results_onesim_evolvecs["time"] = np.append(dict_results_onesim_evolvecs["time"], time_t)
                    
                # Noise
                if ind_time == ind_init:
                    dict_results_onesim["noise"] = np.array([arr_muN_pk_s[ind_time]])
                else:
                    dict_results_onesim["noise"] = np.append(dict_results_onesim["noise"], arr_muN_pk_s[ind_time])
                    
                # Diabatic error (TT : init total - compare total, "default")
                if dict_operations_onesim["diabErr"] == True:
                    
                    # Evolve covariance matrix of initial, instantaneous ground state of total Hamiltonian
                    matM_evol = matU @ matM_init @ matU.T.conj()
                    # Square of the overlap between the instantaneous MB ground state and time-evolved MB ground state
                    sqrmod = sqrmod_overlap_covar(matM_evol, dict_intervars["matM"])
                    # Calculate the diabatic error
                    diabErr_t = np.abs(1 - np.abs(sqrmod))
                    # Storage
                    if ind_time == ind_init:
                        dict_results_onesim["diabErr"] = np.array([diabErr_t])
                    else:
                        dict_results_onesim["diabErr"] = np.append(dict_results_onesim["diabErr"], diabErr_t)
                        
                # Diabatic error (CC : init clean - compare clean)
                if dict_operations_onesim["diabErr_cc"] == True:
                    
                    # Evolve covariance matrix of initial, instantaneous ground state of clean Hamiltonian
                    matM_clean_evol = matU @ matM_clean_init @ matU.T.conj()
                    # Square of the overlap between the instantaneous MB ground state and time-evolved MB ground state
                    sqrmod_clean = sqrmod_overlap_covar(matM_clean_evol, dict_intervars["matM_clean"])
                    # Calculate the diabatic error
                    diabErr_t_cc = np.abs(1 - np.abs(sqrmod_clean))
                    # Storage
                    if ind_time == ind_init:
                        dict_results_onesim["diabErr_cc"] = np.array([diabErr_t_cc])
                    else:
                        dict_results_onesim["diabErr_cc"] = np.append(dict_results_onesim["diabErr_cc"], diabErr_t_cc)

                # Diabatic error (CT : init clean - compare total)
                if dict_operations_onesim["diabErr_ct"] == True:
                    
                    # Evolve covariance matrix of initial, instantaneous ground state of clean Hamiltonian
                    matM_clean_evol = matU @ matM_clean_init @ matU.T.conj()
                    # Square of the overlap between the instantaneous MB ground state (T) and time-evolved MB ground state (C)
                    sqrmod_ct = sqrmod_overlap_covar(matM_clean_evol, dict_intervars["matM"])
                    # Calculate the diabatic error
                    diabErr_t_ct = np.abs(1 - np.abs(sqrmod_ct))
                    # Storage
                    if ind_time == ind_init:
                        dict_results_onesim["diabErr_ct"] = np.array([diabErr_t_ct])
                    else:
                        dict_results_onesim["diabErr_ct"] = np.append(dict_results_onesim["diabErr_ct"], diabErr_t_ct)        
    
                # Transition probabilities (TT : init total - compare total, "default")
                if dict_operations_onesim["transProbs"] == True:
                    
                    # Time evolve covariance matrix of the initial ground state of total Hamiltonian
                    matM_evol = matU @ matM_init @ matU.T.conj()
                    # Extract the real, anti-symmetric matrix A defined through H = iA
                    matA = (-1j*matH).real
                    # Orthogonal transformation for A
                    matO = calc_matO(matA)
                    
                    # Loop over chosen many-body excited states
                    for ind in dict_onesim["transProbs"]: 
                        
                        # Covariance matrix of excited state in the eigenbasis
                        matMO_exc = calc_matMO(L, R, ind)
                        # Above covariance matrices in the original basis
                        matM_exc_even, matM_exc_odd = calc_matM(matMO_exc, matO)
                        # Take the even result
                        maxM_exc = matM_exc_even
                        # Transition probability
                        dict_transProbs_t = sqrmod_overlap_covar(matM_evol, maxM_exc)
                        
                        # Dictionary key for each transition probability
                        ind.sort()
                        str_ind = list(map(str, ind))
                        key_transProb = "transProbs_" + "_".join(str_ind)
                        
                        # Storage
                        if ind_time == ind_init:
                            dict_results_onesim[key_transProb] = np.array([dict_transProbs_t])
                        else:
                            dict_results_onesim[key_transProb] = np.append(dict_results_onesim[key_transProb], dict_transProbs_t)
                            
                # Transition probabilities (CC : init clean - compare clean)
                if dict_operations_onesim["transProbs_cc"] == True:
                    
                    # Time evolve covariance matrix of the initial ground state of clean Hamiltonian
                    matM_clean_evol = matU @ matM_clean_init @ matU.T.conj()
                    
                    # Establish Hamiltonian
                    arr_mu_clean[site_pk_start:site_pk_end] = arr_mu_pk_s[ind_time]
                    matH_clean = ham_kit(L+R, arr_mu_clean, arr_w, arr_Delta)
                    # Extract the real, anti-symmetric matrix A defined through H = iA
                    matA_clean = (-1j*matH_clean).real
                    # Orthogonal transformation for A
                    matO_clean = calc_matO(matA_clean)
                    
                    # Loop over chosen many-body excited states
                    for ind in dict_onesim["transProbs_cc"]: 
                        
                        # Covariance matrix of excited state in the eigenbasis
                        matMO_clean_exc = calc_matMO(L, R, ind)
                        # Above covariance matrices in the original basis
                        matM_clean_exc_even, matM_clean_exc_odd = calc_matM(matMO_clean_exc, matO_clean)
                        # Take the even result
                        maxM_clean_exc = matM_clean_exc_even
                        # Transition probability
                        dict_transProbs_t_cc = sqrmod_overlap_covar(matM_clean_evol, maxM_clean_exc)
                        
                        # Dictionary key for each transition probability
                        ind.sort()
                        str_ind = list(map(str, ind))
                        key_transProb = "transProbs_cc_" + "_".join(str_ind)
                        
                        # Storage
                        if ind_time == ind_init:
                            dict_results_onesim[key_transProb] = np.array([dict_transProbs_t_cc])
                        else:
                            dict_results_onesim[key_transProb] = np.append(dict_results_onesim[key_transProb], dict_transProbs_t_cc)
                 
                """
                # Expectation value of total Hamiltonian wrt. time-evolved eigenstate of initial total Hamiltonian
                if dict_operations_onesim["exptEnergy"] == True:
                    
                    # Loop over chosen eigenvectors/energies
                    for key in dict_exptEnergy.keys():
                        evolEigVec = matU @ dict_exptEnergy[key] 
                        dict_exptEnergy_t = evolEigVec.T.conj() @ matH @ evolEigVec
                        
                        # -> Storage
                        if ind_time == ind_init:
                            dict_results_onesim["exptEnergy"][key] = np.array([dict_exptEnergy_t])
                        else:
                            dict_results_onesim["exptEnergy"][key] = np.append(dict_results_onesim["exptEnergy"][key], dict_exptEnergy_t)
                """
                    
                # Occupation numbers
                if dict_operations_onesim["occupation"] == True:  
                    
                    if ind_time == ind_init:
                        dict_eigvec_occup_real_prev = {}
                        dict_eigvec_occup_imag_prev = {}
                    
                    # -> Time evolve the covariance matrix of the initial ground state
                    matM_evol = matU @ matM_init @ matU.T.conj()
                    
                    # -> Loop through occupation numbers 
                    for ind in dict_onesim["occupation"]:
                        
                        # ---> Establish corresponding instantaneous eigenvector and normalize the real/imaginary parts
                        eigvec_occup = dict_intervars["eigvecH"][:,L+R+ind]
                        eigvec_occup_real = eigvec_occup.real/np.linalg.norm(eigvec_occup.real)
                        eigvec_occup_imag = eigvec_occup.imag/np.linalg.norm(eigvec_occup.imag)
                        
                        # ---> At the beginning of the piano key press, use the above to calculate the occupation number
                        # ---> Otherwise, compare with previous eigenvectors to ensure consistency/continuity, then calculate occupation number
                        if ind_time == ind_init:
                            dict_occupation_t = occupation_number_mzm(matM_evol, eigvec_occup_real, eigvec_occup_imag)
                            dict_occupation_t_swap = occupation_number_mzm(matM_evol, eigvec_occup_imag, eigvec_occup_real)
                            dict_eigvec_occup_real_prev[str(ind)] = np.copy(eigvec_occup_real)
                            dict_eigvec_occup_imag_prev[str(ind)] = np.copy(eigvec_occup_imag)
                        else:
                            eigvec_occup_real_act, eigvec_occup_imag_act = eigvec_reim_continuity(eigvec_occup_real, eigvec_occup_imag, dict_eigvec_occup_real_prev[str(ind)], dict_eigvec_occup_imag_prev[str(ind)])
                            dict_occupation_t = occupation_number_mzm(matM_evol, eigvec_occup_real_act, eigvec_occup_imag_act)
                            dict_occupation_t_swap = occupation_number_mzm(matM_evol, eigvec_occup_imag_act, eigvec_occup_real_act)
                            dict_eigvec_occup_real_prev[str(ind)] = np.copy(eigvec_occup_real_act)
                            dict_eigvec_occup_imag_prev[str(ind)] = np.copy(eigvec_occup_imag_act)
                            
                        # We assume that the state initially contains no particles, and so the occupation numbers must necessarily be initially zero. In defining the number operator, it might be necessary to flip the real and imaginary parts in the calculationto ensure that this is the case
                        # The following takes the lower value between dict_occupation_t and dict_occupation_t_swap
                        if dict_occupation_t > dict_occupation_t_swap:
                            dict_occupation_t = dict_occupation_t_swap
                            
                        # ---> Storage
                        if ind_time == ind_init:
                            dict_results_onesim["occupation_{}".format(ind)] = np.array([dict_occupation_t])
                        else:
                            dict_results_onesim["occupation_{}".format(ind)] = np.append(dict_results_onesim["occupation_{}".format(ind)], dict_occupation_t)
                
                # Instantaneous spectrum of total Hamiltonian
                if dict_operations_onesim["instEnergy"] == True:
                    instSpec_t = dict_intervars["eigvalH"]
                    # Storage
                    instSpec_t_reshape = np.expand_dims(instSpec_t, 1)
                    if ind_time == ind_init:
                        dict_results_onesim_spec["instEnergy"] = instSpec_t_reshape
                    else:
                        dict_results_onesim_spec["instEnergy"] = np.column_stack((dict_results_onesim_spec["instEnergy"], instSpec_t_reshape))
                        
                # Instantaneous spectrum of clean Hamiltonian
                if dict_operations_onesim["instEnergy_c"] == True:
                    instSpec_t_c = dict_intervars["eigvalH_clean"]
                    # Storage
                    instSpec_t_c_reshape = np.expand_dims(instSpec_t_c, 1)
                    if ind_time == ind_init:
                        dict_results_onesim_spec_clean["instEnergy"] = instSpec_t_c_reshape
                    else:
                        dict_results_onesim_spec_clean["instEnergy"] = np.column_stack((dict_results_onesim_spec_clean["instEnergy"], instSpec_t_c_reshape))
                
                # Instantaneous eigenvectors
                if dict_operations_onesim["instEigVec"] == True:
                    # Loop over chosen eigenvectors
                    for ind in dict_onesim["instEigVec"]:
                        dict_instEigVecs_t = dict_intervars["eigvecH"][:,L+R+ind]
                        # Storage
                        instEigVecs_t_reshape = np.expand_dims(dict_instEigVecs_t, 1)
                        if ind_time == ind_init:
                            dict_results_onesim_eigvecs["instEigVec_{}".format(ind)] = instEigVecs_t_reshape
                        else:
                            dict_results_onesim_eigvecs["instEigVec_{}".format(ind)] = np.column_stack((dict_results_onesim_eigvecs["instEigVec_{}".format(ind)], instEigVecs_t_reshape))
                
                """
                # Time-evolved eigenvectors
                if dict_operations_onesim["evolEigVec"] == True:
                    # Loop over chosen eigenvectors
                    for key in dict_evolEigVec.keys():
                        dict_evolEigVecs_t = matU @ dict_evolEigVec[key] 
                        # -> Storage
                        evolEigVecs_t_reshape = np.expand_dims(dict_evolEigVecs_t, 1)
                        if ind_time == ind_init:
                            dict_results_onesim["evolEigVec"][key] = evolEigVecs_t_reshape
                        else:
                            dict_results_onesim["evolEigVec"][key] = np.column_stack((dict_results_onesim["evolEigVec"][key], evolEigVecs_t_reshape))
                """ 
                
    return {"matU":matU, "onesim":dict_results_onesim, "onesim_spec":dict_results_onesim_spec, "onesim_spec_clean":dict_results_onesim_spec_clean, "onesim_eigvecs":dict_results_onesim_eigvecs, "onesim_evolvecs":dict_results_onesim_evolvecs}
    
# Calculate the single-particle time evolution operator for the piano key protocol performed with multiple piano keys ('steps')
# Notes and features:
# -> Standard Majorana basis
# -> The following is calculated during a single simulation:
# -> -> Time evolution operator (forward and backwards)
# Inputs: 
# -> System parameters (r = piano key size) 
# -> Time parameters (tn = total time for nth piano key, n = piano key index)
# -> matU: Most updated time evolution operator of protocol
# -> matM_init: Covariance matrix of the initial ground state (assume even)
# -> list_ind_instEigVecs: list of indices of the instantaneous eigenvectors that will be calculated,  
# -> list_ind_occupation: list of indices of the occupation numbers that will be calculated
# -> list_ind_transProb: list of indices of the excited states for which transition probabilities will be calculated
# -> dict_evolEigVec: dictionary with index:initial eigenvector 
# -> dict_exptEnergy: dictionary with index:initial eigenvector, used to calculate expectation value of H(t)
# -> flag_backward allows for a reverse piano-key protocol to be simulated
def simulate_pianokey_old(L, R, r, Delta, w, muLeftVal, muRightStart, muRightEnd, tn, N, n, arr_muRight_s, arr_muD, NSamp, matU, matM_init, list_ind_instEigVecs, list_ind_occupation, list_ind_transProb, dict_evolEigVec, dict_exptEnergy, dict_operations_onesim, flag_onesim=False, flag_backward=False, flag_SVD=True, flag_printprogress=True):
    
    # ------------- #
    # Initial setup # 
    # ------------- #
    
    # Parameters
    arr_Delta = np.full(L+R, Delta)
    arr_w = np.full(L+R, w)
    arr_muLeft = np.full(L, muLeftVal)
    
    # Initialize dictionary (and sub-dictionaries therein) for full simulation results
    dict_results = {}
    
    # Initialize dictionary (and sub-dictionaries therein) for single simulation results
    dict_results_onesim = {"time":0}
    
    for key in dict_operations_onesim.keys():
        dict_results_onesim[key] = 0
    
    if dict_operations_onesim["instEigVec"] == True:
        dict_results_onesim["instEigVec"] = {}
        for ind in list_ind_instEigVecs:
            dict_results_onesim["instEigVec"][str(ind)] = 0
            
    if dict_operations_onesim["transProbs"] == True:
        dict_results_onesim["transProbs"] = {}
        for ind in list_ind_transProb:
            #str_ind = "".join(map(str, ind))
            str_ind = ",".join(map(str, ind))
            dict_results_onesim["transProbs"][str_ind] = 0
            
    if dict_operations_onesim["occupation"] == True:
        dict_results_onesim["occupation"] = {}
        for ind in list_ind_occupation:
            dict_results_onesim["occupation"][str(ind)] = 0
    
    if dict_operations_onesim["exptEnergy"] == True:
        dict_results_onesim["exptEnergy"] = {}
        for key in dict_exptEnergy.keys():
            dict_results_onesim["exptEnergy"][key] = 0
            
    if dict_operations_onesim["evolEigVec"] == True:
        dict_results_onesim["evolEigVec"] = {}
        for key in dict_evolEigVec.keys():
            dict_results_onesim["evolEigVec"][key] = 0
    
    # --------------------------- #
    # Time evolution of piano key #
    # --------------------------- #
    
    # Difference between time steps
    dt = tn/N
    
    # Initialize the time evolution operator as an identity matrix
    matTime = np.eye(2*(L+R), dtype='complex')
    
    # Initialize the *backward* time evolution operator as an identity matrix
    matTimeBack = np.eye(2*(L+R), dtype='complex')
    
    # Build the time evolution operator via discretization
    for i in range(0, N):
        
        # Print i
        if flag_printprogress == True:
            if (i+1)%(N//10) == 0:
                print("Step:", i+1, "/", N)
        
        # Chemical potentials
        arr_muRight = np.full(R, muRightStart) # Chemical potential on the right side of the chain
        arr_muRight[:(n*r)] = muRightEnd # On the nth step, ensure that the previous (n-1)*r elements of arr_muRight are fixed (these piano keys have already been changed)
        
        # If we are going forwards, we press down on a piano key; if we go backwards, we lift a piano key
        if flag_backward == False:
            arr_muRight[n*r:(n+1)*r] = arr_muRight_s[i] # On the nth step, ensure that the (n*r)th to ((n+1)*r)th elements of arr_muRight changes with time (changing the nth piano key)
        else:
            arr_muRight[n*r:(n+1)*r] = np.flip(arr_muRight_s)[i] # On the nth step, ensure that the nth to (n+1)th elements of arr_muRight changes with time (changing the nth piano key)
        
        # Array of chemical potentials everywhere
        arr_mu = np.concatenate((arr_muLeft, arr_muRight))
        
        # Add disorder to chemical potential
        arr_mu = arr_mu + arr_muD
        
        # Compute the single-particle Hamiltonian
        matH = ham_kit(L+R, arr_mu, arr_w, arr_Delta)
        
        # Compute the ''cross sectional'' time evolution operator via matrix exponentiation
        matTime_slice = sc_linalg.expm(-2*1j*dt*matH)
          
        # Update the total time evolution operator
        matTime = matTime_slice @ matTime
        
        # Update the *backward* total time evolution operator
        matTimeBack = matTimeBack @ matTime_slice
        
        # If desired, use an SVD on the time evolution operator to bring them as close as possible to its true counterparts
        if flag_SVD == True:
            unitary_measure = np.linalg.norm(matTime.conj().T @ matTime - np.identity(2*(L+R), dtype='complex'))
            unitary_measure_B = np.linalg.norm(matTimeBack.conj().T @ matTimeBack - np.identity(2*(L+R), dtype='complex'))
            if unitary_measure > 1e-14:
                SVD = np.linalg.svd(matTime)
                matTime = SVD[0] @ SVD[2]
            if unitary_measure_B > 1e-14:
                SVD_B = np.linalg.svd(matTimeBack)
                matTimeBack = SVD_B[0] @ SVD_B[2]
                
        # ---------------------------- #
        # Single simulation management #
        # ---------------------------- #
        
        if flag_onesim == True:
        
            # Calculate single simulation quantities only at certain intervals of t, depending on NSamp
            if i%(N//NSamp) == 0 or i == N-1:
                flag_updateArr = True
            else:
                flag_updateArr = False
                
            # Calculate quantities when flag_updateArr == True
            if flag_updateArr == True:
                
                # ----------------------------- #
                # Set up intermediate variables #
                # ----------------------------- #
                
                # Dictionary of intermediate variables, arrays, vectors, etc
                dict_intervars = {"eigvalH":0, "eigvecH":0, "matU_upd":0, "matM":0}
                
                # Diagonalize the instantaneous Hamiltonian
                if True in [dict_operations_onesim["instEnergy"], dict_operations_onesim["instEigVec"], dict_operations_onesim["occupation"]]:
                    
                    # -> Diagonalize matH
                    eigvalH, eigvecH = sc_linalg.eigh(matH)
                    
                    # -> Sort the eigenvalues in ascending order; sort the eigenvectors accordingly
                    ind_eigvalH_sorted = np.argsort(np.real(eigvalH))
                    eigvalH = np.real(eigvalH[ind_eigvalH_sorted])
                    eigvecH = eigvecH[:,ind_eigvalH_sorted]
                    
                    # -> Store
                    dict_intervars["eigvalH"] = eigvalH
                    dict_intervars["eigvecH"] = eigvecH
                    
                # Update the time evolution operator up to time "t"
                if True in [dict_operations_onesim["diabErr"], dict_operations_onesim["transProbs"], dict_operations_onesim["occupation"], dict_operations_onesim["evolEigVec"], dict_operations_onesim["exptEnergy"]]:
                    
                    # -> Update matU
                    matU_upd = matTime @ matU
                    
                    # -> Use SVD method to remove numerical errors 
                    try:
                        mats_SVD = np.linalg.svd(matU_upd)
                        matU_upd = mats_SVD[0] @ mats_SVD[2]
                    except:
                        pass
                    
                    # -> Store
                    dict_intervars["matU_upd"] = matU_upd
                    
                # Calculate the covariance matrix of the instantaneous ground state
                if True in [dict_operations_onesim["diabErr"], dict_operations_onesim["occupation"], dict_operations_onesim["transProbs"]]:
                    
                    # Extract the real, anti-symmetric matrix A defined through H = iA
                    matA = (-1j*matH).real
                    
                    # Create covariance matrices for the many-body ground states (even and odd)
                    # -> Orthogonal transformations
                    matO = calc_matO(matA)
                    # -> Covariance matrix in the eigenbases
                    matMO = calc_matMO(L, R, [])
                    # -> Covariance matrices in the original basis
                    matM_even, matM_odd = calc_matM(matMO, matO)
                    
                    # -> Store only the even matM
                    dict_intervars["matM"] = matM_even
                    
                # -------------------- #
                # Calculate quantities #
                # -------------------- #
                
                # Time 
                time_t = tn*(i/(N-1) + n)
                
                # -> Storage
                if i == 0:
                    dict_results_onesim["time"] = np.array([time_t])
                else:
                    dict_results_onesim["time"] = np.append(dict_results_onesim["time"], time_t)
                
                # Instantaneous spectrum 
                if dict_operations_onesim["instEnergy"] == True:
                    instSpec_t = dict_intervars["eigvalH"]
                    
                    # -> Storage
                    instSpec_t_reshape = np.expand_dims(instSpec_t, 1)
                    if i == 0:
                        dict_results_onesim["instEnergy"] = instSpec_t_reshape
                    else:
                        dict_results_onesim["instEnergy"] = np.column_stack((dict_results_onesim["instEnergy"], instSpec_t_reshape))
                
                # Instantaneous eigenvectors
                if dict_operations_onesim["instEigVec"] == True:
                    # -> Loop over chosen eigenvectors
                    for ind in list_ind_instEigVecs:
                        dict_instEigVecs_t = dict_intervars["eigvecH"][:,L+R+ind]
                        
                        # -> Storage
                        instEigVecs_t_reshape = np.expand_dims(dict_instEigVecs_t, 1)
                        if i == 0:
                            dict_results_onesim["instEigVec"][str(ind)] = instEigVecs_t_reshape
                        else:
                            dict_results_onesim["instEigVec"][str(ind)] = np.column_stack((dict_results_onesim["instEigVec"][str(ind)], instEigVecs_t_reshape))
                            
                # Diabatic error
                if dict_operations_onesim["diabErr"] == True:
                    # -> Time evolve the covariance matrix of the initial ground state
                    matM_evol = dict_intervars["matU_upd"] @ matM_init @ dict_intervars["matU_upd"].T.conj()
                    # -> Square of the overlap between the instantaneous MB ground state and time-evolved MB ground state
                    sqrmod = sqrmod_overlap_covar(matM_evol, dict_intervars["matM"])
                    # -> Calculate the diabatic error
                    diabErr_t = np.abs(1 - np.abs(sqrmod))
                    
                    # -> Storage
                    if i == 0:
                        dict_results_onesim["diabErr"] = np.array([diabErr_t])
                    else:
                        dict_results_onesim["diabErr"] = np.append(dict_results_onesim["diabErr"], diabErr_t)
    
                # Transition probabilities
                if dict_operations_onesim["transProbs"] == True:
                    
                    # -> Time evolve the covariance matrix of the initial ground state
                    matM_evol = dict_intervars["matU_upd"] @ matM_init @ dict_intervars["matU_upd"].T.conj()
                    # Extract the real, anti-symmetric matrix A defined through H = iA
                    matA = (-1j*matH).real
                    # Orthogonal transformation for A
                    matO = calc_matO(matA)
                    
                    # -> Loop over chosen many body excited states
                    for ind in list_ind_transProb: 
                        # ---> Covariance matrix of excited state in the eigenbasis
                        matMO_exc = calc_matMO(L, R, ind)
                        # ---> Above covariance matrices in the original basis
                        matM_exc_even, matM_exc_odd = calc_matM(matMO_exc, matO)
                        # ---> Take the even result
                        maxM_exc = matM_exc_even
                        # ---> Transition probability
                        #str_ind = "".join(map(str, ind))
                        str_ind = ",".join(map(str, ind))
                        dict_transProbs_t = sqrmod_overlap_covar(matM_evol, maxM_exc)
                        
                        # ---> Storage
                        if i == 0:
                            dict_results_onesim["transProbs"][str_ind] = np.array([dict_transProbs_t])
                        else:
                            dict_results_onesim["transProbs"][str_ind] = np.append(dict_results_onesim["transProbs"][str_ind], dict_transProbs_t)
                        
                # Time-evolved eigenvectors
                if dict_operations_onesim["evolEigVec"] == True:
                    # -> Loop over chosen eigenvectors
                    for key in dict_evolEigVec.keys():
                        dict_evolEigVecs_t = dict_intervars["matU_upd"] @ dict_evolEigVec[key] 
                        
                        # -> Storage
                        evolEigVecs_t_reshape = np.expand_dims(dict_evolEigVecs_t, 1)
                        if i == 0:
                            dict_results_onesim["evolEigVec"][key] = evolEigVecs_t_reshape
                        else:
                            dict_results_onesim["evolEigVec"][key] = np.column_stack((dict_results_onesim["evolEigVec"][key], evolEigVecs_t_reshape))
                        
                # Expectation value of instantaneous Hamiltonian
                if dict_operations_onesim["exptEnergy"] == True:
                    # -> Loop over chosen eigenvectors/energies
                    for key in dict_exptEnergy.keys():
                        evolEigVec = dict_intervars["matU_upd"] @ dict_exptEnergy[key] 
                        dict_exptEnergy_t = evolEigVec.T.conj() @ matH @ evolEigVec
                        
                        # -> Storage
                        if i == 0:
                            dict_results_onesim["exptEnergy"][key] = np.array([dict_exptEnergy_t])
                        else:
                            dict_results_onesim["exptEnergy"][key] = np.append(dict_results_onesim["exptEnergy"][key], dict_exptEnergy_t)
                    
                # Occupation numbers
                if dict_operations_onesim["occupation"] == True:  
                    
                    if i == 0:
                        dict_eigvec_occup_real_prev = {}
                        dict_eigvec_occup_imag_prev = {}
                    
                    # -> Time evolve the covariance matrix of the initial ground state
                    matM_evol = dict_intervars["matU_upd"] @ matM_init @ dict_intervars["matU_upd"].T.conj()
                    
                    # -> Loop through occupation numbers 
                    for ind in list_ind_occupation:
                        
                        # ---> Establish corresponding instantaneous eigenvector and normalize the real/imaginary parts
                        eigvec_occup = dict_intervars["eigvecH"][:,L+R+ind]
                        eigvec_occup_real = eigvec_occup.real/np.linalg.norm(eigvec_occup.real)
                        eigvec_occup_imag = eigvec_occup.imag/np.linalg.norm(eigvec_occup.imag)
                        
                        # ---> At the beginning of the piano key press, use the above to calculate the occupation number
                        # ---> Otherwise, compare with previous eigenvectors to ensure consistency/continuity, then calculate occupation number
                        if i == 0:
                            dict_occupation_t = occupation_number_mzm(matM_evol, eigvec_occup_real, eigvec_occup_imag)
                            dict_occupation_t_swap = occupation_number_mzm(matM_evol, eigvec_occup_imag, eigvec_occup_real)
                            dict_eigvec_occup_real_prev[str(ind)] = np.copy(eigvec_occup_real)
                            dict_eigvec_occup_imag_prev[str(ind)] = np.copy(eigvec_occup_imag)
                        else:
                            eigvec_occup_real_act, eigvec_occup_imag_act = eigvec_reim_continuity(eigvec_occup_real, eigvec_occup_imag, dict_eigvec_occup_real_prev[str(ind)], dict_eigvec_occup_imag_prev[str(ind)])
                            dict_occupation_t = occupation_number_mzm(matM_evol, eigvec_occup_real_act, eigvec_occup_imag_act)
                            dict_occupation_t_swap = occupation_number_mzm(matM_evol, eigvec_occup_imag_act, eigvec_occup_real_act)
                            dict_eigvec_occup_real_prev[str(ind)] = np.copy(eigvec_occup_real_act)
                            dict_eigvec_occup_imag_prev[str(ind)] = np.copy(eigvec_occup_imag_act)
                            
                        # We assume that the state initially contains no particles, and so the occupation numbers must necessarily be initially zero. In defining the number operator, it might be necessary to flip the real and imaginary parts in the calculationto ensure that this is the case
                        # The following takes the lower value between dict_occupation_t and dict_occupation_t_swap
                        if dict_occupation_t > dict_occupation_t_swap:
                            dict_occupation_t = dict_occupation_t_swap
                            
                        # ---> Storage
                        if i == 0:
                            dict_results_onesim["occupation"][str(ind)] = np.array([dict_occupation_t])
                        else:
                            dict_results_onesim["occupation"][str(ind)] = np.append(dict_results_onesim["occupation"][str(ind)], dict_occupation_t)
                    
    # ----------------------------------- #
    # Data management at end of key press #
    # ----------------------------------- #
    
    dict_results["matTime"] = matTime
    dict_results["matTimeBack"] = matTimeBack
    
    # Return dictionary
    return {"main":dict_results, "onesim":dict_results_onesim}

# -------- #
# Disorder #
# -------- #

# Create disorder potentials which are Gaussian-correlated (GC), picked from a uniform distribution (ND), uses SVD method
def generate_muD_GCUN(size, arr_mean, width, length_corr):
    
    # Define the standard deviation for a uniform distribution
    std = np.sqrt(1/12)*width
    
    # Initialize covariance matrix
    mat_covar_random = np.zeros((size, size))
    
    # Construct covariance matrix
    if length_corr != 0.0:
        for i in range(0, size):
            for j in range(0, size):
                mat_covar_random[i,j] = (std**2)*np.exp(-np.abs(i-j)**2/2/length_corr**2)
    else:
        for i in range(0, size):
            mat_covar_random[i,i] = (std**2)
            
    # Perform an SVD decomposition on the covariance matrix
    mat_U, mat_S, mat_VH = np.linalg.svd(mat_covar_random)
    
    # Create an uncorrelated uniform distribution with zero mean and unit standard deviation
    random_uncorr = np.random.uniform(low=-np.sqrt(12)/2, high=np.sqrt(12)/2, size=size)
    
    # Create a correlated distribution from the above
    random_corr = (mat_U @ np.diag(np.sqrt(mat_S))) @ random_uncorr + arr_mean
    return random_corr

# Create disorder potentials which are Gaussian-correlated (GC), picked from a multivariate normal distribution (ND)
def generate_muD_GCND(size, arr_mean, arr_std, length_corr):
    
    # Initialize covariance matrix
    mat_covar_random = np.zeros((size, size))
    
    # Construct covariance matrix
    if length_corr != 0.0:
        for i in range(0, size):
            for j in range(0, size):
                mat_covar_random[i,j] = (arr_std[i]**2)*np.exp(-np.abs(i-j)**2/2/length_corr**2)
    else:
        for i in range(0, size):
            mat_covar_random[i,i] = (arr_std[i]**2)
            
    # Return 1D array of disorder potentials
    return np.random.multivariate_normal(arr_mean, mat_covar_random)

# Create disorder potentials which are exponentially-correlated (EC), picked from a multivariate normal distribution (ND)
def generate_muD_ECND(size, arr_mean, arr_std, length_decay):
    
    # Initialize covariance matrix
    mat_covar_random = np.zeros((size, size))
    
    # Construct covariance matrix
    if length_decay != 0.0:
        for i in range(0, size):
            for j in range(0, size):
                mat_covar_random[i,j] = (arr_std[i]**2)*np.exp(-np.abs(i-j)/length_decay)
    else:
        for i in range(0, size):
            mat_covar_random[i,i] = (arr_std[i]**2)
            
    # Return 1D array of disorder potentials
    return np.random.multivariate_normal(arr_mean, mat_covar_random)

# Create disorder potentials which are simple-correlated (SC), picked initially from a normal distribution (ND). Correlations are manually set
def generate_muD_SCND(size, mean, std, length_corr_):
    # length_corr must be an integer -- ff it isn't, integer-fy it
    length_corr = int(length_corr_)
    # Number of correlated sections in the chain
    nbr_section_corr = int(np.ceil((size)/length_corr))
    # Build the array of disorder values
    arr_muD = np.zeros(size, dtype='float')
    for ind_section in range(0, nbr_section_corr):
        if ind_section != nbr_section_corr-1:
            arr_muD[ind_section*length_corr:(ind_section+1)*length_corr] = np.random.normal(loc=mean, scale=std) 
        else:
            arr_muD[ind_section*length_corr:] = np.random.normal(loc=mean, scale=std) 
    return arr_muD

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

# ---------------------------------------------- #
# Analytical expressions in the case of disorder #
# ---------------------------------------------- #

# Kinetic energy of Kitaev chain
def kinenerg(k, mu, w):
    return -mu - w*np.cos(k)

# Bulk energy of Kitaev chain
def bulkenerg(k, mu, w, DeltaSC):
    return np.sqrt(kinenerg(k, mu, w)**2 + DeltaSC**2*np.sin(k)**2)

# Bulk energy of Kitaev chain; for numerical solutions
# -> Used to numerically solve for k given E
def bulkenerg_solve(k, mu, w, DeltaSC, E):
    return np.sqrt(kinenerg(k, mu, w)**2 + DeltaSC**2*np.sin(k)**2) - E

# u(k) from the Bogoliubov transformation in momentum space with translational invariance
# -> Chosen to be real
def uK(k, mu, w, DeltaSC):
    return np.sqrt(0.5*(1 + kinenerg(k, mu, w)/bulkenerg(k, mu, w, DeltaSC)))

# v(k) from the Bogoliubov transformation in momentum space with translational invariance
# -> Contains an overall phase since uK is chosen to be real
def vK(k, mu, w, DeltaSC, flag_absolute_uv=False):
    if flag_absolute_uv == False:
        return 1j*np.sign(np.sin(k))*np.sqrt(0.5*(1 - kinenerg(k, mu, w)/bulkenerg(k, mu, w, DeltaSC)))
    else:
        return np.sqrt(0.5*(1 - kinenerg(k, mu, w)/bulkenerg(k, mu, w, DeltaSC)))

# Average of square of first order correction to minimum gap
# -> Based on first attempt using perturbation theory and is technically incorrect
# -> Piano key with translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections unaccounted for
def expr_mingap_1stCorrSqr_avg_A(mu, w, DeltaSC, N, k1, k_arr, disorder_type, sigma, xi, phase_chi=0.0):
    
    # Coherence factors uK1, vK1
    uK1_sqr = np.abs(uK(k1, mu, w, DeltaSC))**2
    vK1_sqr = np.abs(vK(k1, mu, w, DeltaSC))**2
    
    # Sum over coherence factors vK
    vK_sum = np.sum(np.abs(vK(k_arr, mu, w, DeltaSC))**2)
    
    # Calculate sum over covariance matrix elements depending on disorder type
    if disorder_type in ["UCND", "UCUN"]:
        covar_sum = (sigma**2)*N
    else:
        covar_sum = (sigma**2)*N
    
    # Evaluate correction term
    correction_1stCorrSqr_avg = (1/N**2)*covar_sum*(uK1_sqr - vK1_sqr + vK_sum)**2

    return correction_1stCorrSqr_avg

# Average of square of first order correction to minimum gap
# -> Based on second attempt using perturbation theory and corrects the errors from the first attempt
# -> Piano key with translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections accounted for
# -> Verified using single-particle calculation
def expr_mingap_1stCorrSqr_avg_B(mu, w, DeltaSC, N, k1, disorder_type, sigma, xi, phase_chi=0.0):
    
    # Coherence factors uK1, vK1
    uK1_sqr = np.abs(uK(k1, mu, w, DeltaSC))**2
    vK1_sqr = np.abs(vK(k1, mu, w, DeltaSC))**2
    
    # Calculate sum over covariance matrix elements depending on disorder type
    if disorder_type in ["UCND", "UCUN"]:
        covar_sum = (sigma**2)*N
    else:
        covar_sum = (sigma**2)*N
    
    # Evaluate correction term
    correction_1stCorrSqr_avg = (1/N**2)*covar_sum*(uK1_sqr - vK1_sqr)**2

    return correction_1stCorrSqr_avg

# Average of square of first order correction to minimum gap
# -> Based on third attempt using perturbation theory
# -> Full chain, no translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections accounted for
# -> Coherence factors calculated numerically, exactly by default
def expr_mingap_1stCorrSqr_avg_C(L, R, lp, muLeft, muRight, w, Delta, n1, disorder_type, sigma, xi):
    
    # Set up the single-particle clean Hamiltonian in the Majorana basis
    arr_mu = np.concatenate((np.full(L, muLeft), np.full(R, muRight)))
    arr_w = np.full(L+R, w)
    arr_Delta = np.full(L+R, Delta)
    matH_maj = ham_kit(L+R, arr_mu, arr_w, arr_Delta)
    
    # Convert Hamiltonian to the electron basis (taking into account appropriate factors if necessary)
    matH_ele = maj_to_ee(2*matH_maj)

    # Diagonalize clean Hamiltonian
    eigvals, eigvecs = sc_linalg.eigh(matH_ele)

    # Sort above in ascending order
    ind_sort_eig = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[ind_sort_eig])
    eigvecs = eigvecs[:,ind_sort_eig]
    
    # Extract the electron and hole components and use them to calculate correction
    uj_sqr = np.abs(eigvecs[::2,L+R+n1])**2
    vj_sqr = np.abs(eigvecs[1::2,L+R+n1])**2
    
    # Choose expression depending on disorder type
    if disorder_type in ["UCND", "UCUN"]:
        eh_diffSqr_sum = np.sum((uj_sqr - vj_sqr)**2)
        correction_1stCorrSqr_avg = sigma**2*eh_diffSqr_sum
        
    if disorder_type in ["GCND", "GCUN"]:
        
        if type(sigma) == np.ndarray:
            correction_1stCorrSqr_avg = np.zeros(sigma.shape[0])
        if type(xi) == np.ndarray:
            correction_1stCorrSqr_avg = np.zeros(xi.shape[0])
            
        for ind in range(0, correction_1stCorrSqr_avg.shape[0]):
            
            sigma_s = sigma
            xi_s = xi
            
            # Varying parameters
            if type(sigma) == np.ndarray:
                sigma_s = sigma[ind]
            if type(xi) == np.ndarray:
                xi_s = xi[ind]
            
            for j in range(0, L+R):
                for jp in range(0, L+R):
                    if xi_s != 0.0:
                        correction_1stCorrSqr_avg[ind] += (sigma_s**2)*np.exp(-(j-jp)**2/2/xi_s**2)*(uj_sqr[j] - vj_sqr[j])*(uj_sqr[jp] - vj_sqr[jp])
                    else:
                        if j == jp:
                            correction_1stCorrSqr_avg[ind] += (sigma_s**2)*(uj_sqr[j] - vj_sqr[j])*(uj_sqr[jp] - vj_sqr[jp])
            
    return correction_1stCorrSqr_avg

# Average of square of first order correction to minimum gap
# -> Based on third attempt using perturbation theory
# -> Full chain, no translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections accounted for
# -> Coherence factors calculated approximately, given by analytical expressions
def expr_mingap_1stCorrSqr_avg_C_approx(L, lp, muRight, w, Delta, k1, phase_u, phase_v, disorder_type, sigma, xi, phase_chi=0.0):
    
    # Choose expression depending on disorder type
    if disorder_type in ["UCND", "UCUN"]:
        
        # Variables
        arr_n = np.arange(L, L+lp)
            
        # Approximate expressions for uj_sqr and vj_sqr
        uK1_sqr = np.abs(uK(k1, muRight, w, Delta))**2
        vK1_sqr = np.abs(vK(k1, muRight, w, Delta))**2
        uj_sqr = (2/lp)*uK1_sqr*np.cos(k1*arr_n + phase_u)**2
        vj_sqr = (2/lp)*vK1_sqr*np.cos(k1*arr_n + phase_v)**2
        
        # Evaluate correction term
        eh_diffSqr_sum = np.sum((uj_sqr - vj_sqr)**2)
        correction_1stCorrSqr_avg = sigma**2*eh_diffSqr_sum
        
    return correction_1stCorrSqr_avg 

# Average of second order correction to minimum gap
# -> Based on first attempt using perturbation theory and is technically incorrect
# -> Piano key with translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections unaccounted for
def expr_mingap_2ndCorr_avg_A(mu, w, DeltaSC, N, k1, k_arr, disorder_type, sigma, xi, phase_chi=0.0):
    
    # Remove k1 from arr_k 
    k_arr = k_arr[k_arr != k1]
    
    # Remove (effectively) -k1 from arr_k
    k_arr = k_arr[k_arr != 4*np.pi - k1]
    
    # Coherence factors uK1, vK1
    uK1 = uK(k1, mu, w, DeltaSC)
    vK1 = vK(k1, mu, w, DeltaSC)
    
    # Coherence factors uKj and vKj
    uKj = uK(k_arr, mu, w, DeltaSC)
    vKj = vK(k_arr, mu, w, DeltaSC)
    
    # Energies
    e_k1 = bulkenerg(k1, mu, w, DeltaSC)
    e_kj = bulkenerg(k_arr, mu, w, DeltaSC)
    
    # Calculate sum over states which depends on the disorder type
    if disorder_type in ["UCND", "UCUN"]: 
        numer = np.abs(uK1*np.conj(uKj) - vK1*np.conj(vKj))**2
        denom = e_k1 - e_kj
        correction_2ndCorr_avg = (sigma**2/N)*np.sum(np.divide(numer, denom))

    return correction_2ndCorr_avg

# Average of second order correction to minimum gap
# -> Based on second attempt using perturbation theory and corrects the errors from the first attempt
# -> Piano key with translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections accounted for
# -> Verified using single-particle calculation
def expr_mingap_2ndCorr_avg_B(mu, w, DeltaSC, N, k1, k_arr, disorder_type, sigma, xi, phase_chi=0.0):
    
    # Remove k1 from arr_k 
    k_arr = k_arr[k_arr != k1]
    
    # Remove (effectively) -k1 from arr_k
    k_arr = k_arr[k_arr != 4*np.pi - k1]
    
    # Coherence factors uK1, vK1
    uK1 = uK(k1, mu, w, DeltaSC)
    vK1 = vK(k1, mu, w, DeltaSC)
    
    # Coherence factors uKj and vKj
    uKj = uK(k_arr, mu, w, DeltaSC)
    vKj = vK(k_arr, mu, w, DeltaSC)
    
    # Energies
    e_k1 = bulkenerg(k1, mu, w, DeltaSC)
    e_kj = bulkenerg(k_arr, mu, w, DeltaSC)
    
    # Calculate sum over states which depends on the disorder type
    # -> Watch the complex conjugations
    if disorder_type in ["UCND", "UCUN"]: 
        numer_posE = np.abs(uK1*np.conj(uKj) - vK1*np.conj(vKj))**2
        numer_negE = np.abs(uK1*vKj - vK1*uKj)**2
        denom_posE = e_k1 - e_kj
        denom_negE = e_k1 + e_kj
        correction_2ndCorr_avg = (sigma**2/N)*(np.sum(np.divide(numer_posE,denom_posE)) + np.sum(numer_negE/denom_negE))

    return correction_2ndCorr_avg

# Average of second order correction to minimum gap
# -> Based on third attempt using perturbation theory
# -> Full chain, no translational invariance
# -> Calculated using many-body formalism
# -> Many-body ground state energy corrections accounted for
# -> Coherence factors calculated numerically, exactly by default
def expr_mingap_2ndCorr_avg_C(L, R, lp, muLeft, muRight, w, Delta, n1, disorder_type, sigma, xi):
    
    # Set up the single-particle clean Hamiltonian in the Majorana basis
    arr_mu = np.concatenate((np.full(L, muLeft), np.full(R, muRight)))
    arr_w = np.full(L+R, w)
    arr_Delta = np.full(L+R, Delta)
    matH_maj = ham_kit(L+R, arr_mu, arr_w, arr_Delta)
    
    # Convert Hamiltonian to the electron basis (taking into account appropriate factors if necessary)
    matH_ele = maj_to_ee(2*matH_maj)

    # Diagonalize clean Hamiltonian
    eigvals, eigvecs = sc_linalg.eigh(matH_ele)

    # Sort above in ascending order
    ind_sort_eig = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[ind_sort_eig])
    eigvecs = eigvecs[:,ind_sort_eig]
    
    # Extract the electron and hole parts from the state n1
    uj1 = eigvecs[::2,L+R+n1]
    vj1 = eigvecs[1::2,L+R+n1]
    
    # Extract the energy of state n1
    e1 = eigvals[L+R+n1]
    
    # Initialize correction term
    correction_2ndCorr_avg = 0.0
    
    # Choose expression depending on disorder type
    if disorder_type in ["UCND", "UCUN"]:
        # Sum over each positive energy state
        for n in range(0, L+R):
            if n == n1:
                continue
            else:
                # Electron and hole parts from state n
                ujn = eigvecs[::2,L+R+n]
                vjn = eigvecs[1::2,L+R+n]
                # Energy
                en = eigvals[L+R+n]
                # Numerators and denominators
                numer_posE = np.sum(np.abs(np.conj(ujn)*uj1 - vj1*np.conj(vjn))**2)
                numer_negE = np.sum(np.abs(uj1*vjn - ujn*vj1)**2)
                denom_posE = e1 - en
                denom_negE = e1 + en
                # Add to correction term
                correction_2ndCorr_avg += numer_posE/denom_posE + numer_negE/denom_negE      
        correction_2ndCorr_avg = (sigma**2)*correction_2ndCorr_avg
        
    if disorder_type in ["GCND", "GCUN"]:
        
        if type(sigma) == np.ndarray:
            correction_2ndCorr_avg = np.zeros(sigma.shape[0], dtype='complex')
        if type(xi) == np.ndarray:
            correction_2ndCorr_avg = np.zeros(xi.shape[0], dtype='complex')
            
        for ind in range(0, correction_2ndCorr_avg.shape[0]):
            
            #print(ind)
            
            sigma_s = sigma
            xi_s = xi
            
            # Varying parameters
            if type(sigma) == np.ndarray:
                sigma_s = sigma[ind]
            if type(xi) == np.ndarray:
                xi_s = xi[ind]
            
            # Sum over each positive energy state
            for n in range(0, L+R):
                
                if n == n1:
                    continue
                
                else:
                    # Electron and hole parts from state n
                    ujn = eigvecs[::2,L+R+n]
                    vjn = eigvecs[1::2,L+R+n]
                    
                    # Construct auxiliary vectors to evaluate numerators
                    vec_numA = uj1*np.conj(ujn) - vj1*np.conj(vjn)
                    vec_numB = uj1*vjn - vj1*ujn
                    
                    # Construct numerators (optimized)
                    if xi_s != 0.0:
                        numerA = np.sum([np.sum((sigma_s**2)*np.exp(-(j-np.arange(0,L+R))**2/2/xi_s**2)*vec_numA[j]*np.conj(vec_numA)) for j in np.arange(0,L+R)])
                        numerB = np.sum([np.sum((sigma_s**2)*np.exp(-(j-np.arange(0,L+R))**2/2/xi_s**2)*vec_numB[j]*np.conj(vec_numB)) for j in np.arange(0,L+R)])
                    else:
                        numerA = np.sum([(sigma_s**2)*vec_numA[j]*np.conj(vec_numA[j]) for j in np.arange(0,L+R)])
                        numerB = np.sum([(sigma_s**2)*vec_numB[j]*np.conj(vec_numB[j]) for j in np.arange(0,L+R)])
                    
                    """
                    # Construct numerators 
                    numerA = 0.0
                    numerB = 0.0
                    for j in range(0, L+R):
                        for jp in range(0, L+R):
                            if xi_s != 0.0:
                                numerA += (sigma_s**2)*np.exp(-(j-jp)**2/2/xi_s**2)*vec_numA[j]*np.conj(vec_numA[jp])
                                numerB += (sigma_s**2)*np.exp(-(j-jp)**2/2/xi_s**2)*vec_numB[j]*np.conj(vec_numB[jp])
                            else:
                                if j == jp:
                                    numerA += (sigma_s**2)*vec_numA[j]*np.conj(vec_numA[jp])
                                    numerB += (sigma_s**2)*vec_numB[j]*np.conj(vec_numB[jp])
                    """
                
                    # Energy
                    en = eigvals[L+R+n]
                    
                    # Add to correction term
                    correction_2ndCorr_avg[ind] += numerA/(e1 - en) + numerB/(e1 + en)
            
    return np.real(correction_2ndCorr_avg)

# Square of first order correction to the minimum gap
# -> See notes disorderN, pg. 45
# -> Disorder-averaged
def mingap_corr_1st_sqr_avg(mu, w, DeltaSC, lp, k1, sigma, type_disorder):
    
    # Momentum space terms
    term_momen = (np.abs(uK(k1, mu, w, DeltaSC))**2 - np.abs(vK(k1, mu, w, DeltaSC))**2)**2
    
    # Correlator
    if type_disorder in ["UCND","UCUN"]:
        term_real = lp*sigma**2
    # -> Placeholder for now
    else:
        term_real = lp*sigma**2
    
    # Full expression
    #return 1/lp*term_real*term_momen
    return 1/(lp**2)*term_real*term_momen

# Square of first order correction to the minimum gap
# -> See notes disorderN, pg. 76
# -> Disorder-averaged
def mingap_corr_1st_sqr_avg_v2(mu, w, DeltaSC, L, lp, k1, uk1_phase, vk1_phase, sigma, type_disorder):
    
    # Positions on the piano key
    x = np.arange(L, L+lp)
    
    # Momentum space terms
    uk1_sqr = np.abs(uK(k1, mu, w, DeltaSC))**2
    vk1_sqr = np.abs(vK(k1, mu, w, DeltaSC))**2
    
    # Each term in sum
    termToSum = (uk1_sqr*np.cos(k1*x + uk1_phase)**2 - vk1_sqr*np.cos(k1*x + vk1_phase)**2)**2
    
    # Sum
    correction = np.sum(termToSum)
    
    # Correlations
    if type_disorder in ["UCND","UCUN"]:
        prefactor_corr = sigma**2
    # -> Placeholder for now
    else:
        prefactor_corr = sigma**2
        
    return 4/lp**2*prefactor_corr*correction

# Second order correction to the minimum gap
# -> See notes disorderN, pg. 58
# -> Disorder-averaged
def mingap_corr_2nd_avg(mu, w, DeltaSC, lp, k1, arr_k, sigma, type_disorder):
    
    # Set up variables
    #n = np.arange(1, lp+1)
    arr_toSum = np.zeros(arr_k.shape[0])
    
    # Loop over k-values
    for ind in range(0, arr_k.shape[0]):
        
        # Momentum management
        kj = arr_k[ind]
        # -> Skip current iteration if momenta match
        if np.round(kj, 7) == np.round(k1, 7):
            #arr_denom_toSum[ind] = 1.0
            arr_toSum[ind] = 0.0
            continue
        
        """
        # Average of Fourier transform of disorder potential with K = k1 - k_alpha and K' = - k1 - k_alpha
        if type_disorder in ["UCND","UCUN"]:
            arr_disorder_ft_sqr = sigma**2
            arr_disorder_ft_neg_sqr  = sigma**2
        # -> Placeholder for now
        else:
            arr_disorder_ft_sqr = sigma**2
            arr_disorder_ft_neg_sqr  = sigma**2
        """
        arr_disorder_ft_sqr = 1.0
        arr_disorder_ft_neg_sqr  = 1.0
        
        # Terms involving coherence factors
        uu_vv_sqr = np.abs(uK(k1, mu, w, DeltaSC)*uK(kj, mu, w, DeltaSC) - vK(k1, mu, w, DeltaSC)*vK(kj, mu, w, DeltaSC))**2
        uv_uv_sqr = np.abs(uK(k1, mu, w, DeltaSC)*vK(kj, mu, w, DeltaSC) - vK(k1, mu, w, DeltaSC)*uK(kj, mu, w, DeltaSC))**2
        
        # Energy differences
        energy_diff = bulkenerg(kj, mu, w, DeltaSC) - bulkenerg(k1, mu, w, DeltaSC)
        energy_sum = bulkenerg(kj, mu, w, DeltaSC) + bulkenerg(k1, mu, w, DeltaSC)
        
        # Term in sum with energy differences
        term_neg = np.divide((arr_disorder_ft_sqr)*(uu_vv_sqr), energy_diff)
        
        # Term in sum with energy sums
        term_pos = np.divide((arr_disorder_ft_neg_sqr)*(uv_uv_sqr), energy_sum)
        
        # All together
        arr_toSum[ind] = -term_neg + term_pos
    
    # Prefactors
    prefactor = 1/lp*sigma**2
    
    #return -prefactor*np.sum(arr_toSum)
    return prefactor*np.sum(arr_toSum)

