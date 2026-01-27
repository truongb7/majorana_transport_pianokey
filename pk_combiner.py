"""
Kitaev chain: Piano key simulations
Combiner

Features:
    - Loads (or creates) a "main" file and loads a list of "secondary" files
    - The secondary file is used to update the contents of the main file
    - Averaging of data sets are performed where applicable
"""

# ------- #
# Modules #
# ------- #

import argparse as ap
import numpy as np
import pandas as pd
import os as os
import time as tm
import pk_functions as pk

# Combines data contained in secondary files with those within a main file
def data_combiner(dict_operations, dict_specifications, dict_files, list_secondary_filenames, list_dataType_all=["diabErr", "diabErr_cc", "diabErr_ct", "transProb", "transProb_cc", "mingap", "mingap_loc", "velocity"], flag_tmAppend=False, flag_replace=False):
    
    # ---------------- #
    # Unpack arguments #
    # ---------------- #
    
    # Operations
    flag_average = dict_operations["average"]
    flag_other = dict_operations["other"]
    
    # Specifications
    str_tuning_choice = dict_specifications["tuning"]
    str_disorder_choice = dict_specifications["disorder"]
    str_noise_choice = dict_specifications["noise"]
    
    # Directories
    # -> Note that for the dictionaries below, "main" is taken care of by the key "data_average" while "secondary" is taken care of the key "data". In other words:
    dirname_main = dict_files["dir_main"]
    dirname_secondary = dict_files["dir_secondary"]
    
    # Designators
    str_desig_main = dict_files["str_desig_main"]
    #str_desig_secondary = dict_files["str_desig_secondary"]
    
    # Other 
    list_str_avgQuants = ["_avg", "_avg_sqr", "_logAvg", "_logAvg_sqr"]

    # ---------- #
    # File names #
    # ---------- #
    
    # Append date and time if desired
    if flag_tmAppend == True:
        str_time = tm.strftime("_%Y%m%d" + "_" + "%H%M")
    else:
        str_time = ""
    
    # File names
    filename_main = str_desig_main + str_tuning_choice + "_" + str_disorder_choice + "_" + str_noise_choice
    #filename_secondary = str_desig_secondary + str_tuning_choice + "_" + str_noise_choice
    
    if flag_other == True:
        filename_main = filename_main + "_OT"
        
    filename_main = filename_main + str_time
    
    # -------------------------- #
    # Folder and file management #
    # -------------------------- #    
        
    # Create main file if it does not exist
    if flag_other == True:
        dict_operations_fullsim = {"diabErr":False, "diabErr_cc":False, "diabErr_ct":False, "transProb":False, "transProb_cc":False, "mingap":True, "velocity":True}
        pk.create_csvfile_other(dict_specifications, dict_operations_fullsim, dirname_main, filename_main, flag_avg=flag_average)
    else:
        dict_operations_fullsim = {"diabErr":True, "diabErr_cc":False, "diabErr_ct":False, "transProb":False, "transProb_cc":False}
        dict_fullsim = {"transProbs":[[0]], "transProbs_cc":[[0]]}
        pk.create_csvfile(dict_specifications, dict_operations_fullsim, dict_fullsim, dirname_main, filename_main, flag_avg=flag_average)
        
    # -------------- #
    # Load main file #
    # -------------- #
    
    df_main = pd.read_csv("{}/{}.csv".format(dirname_main, filename_main))
    
    # Obtain all columns of df_main
    list_cols_all = list(df_main.columns)
    
    # Obtain columns which contain no data; this will be used for indexing
    list_cols_nodata = pk.create_csvcolumns_nodata(dict_specifications)
    
    if flag_other == True:
        list_cols_nodata = [cols for cols in list_cols_nodata if cols not in ["tau"]]
    
    # Obtain columns which contain data
    list_cols_data = [col for col in list_cols_all if col not in list_cols_nodata]
    
    # Set the index of df_main to list_cols_nodata
    df_main.set_index(list_cols_nodata, inplace=True)
    
    # --------------------- #
    # Update main data file #
    # --------------------- #
    
    count = 1
    
    # Loop over secondary files
    for filename_secondary in list_secondary_filenames:
        
        print("File {} of {}".format(count, len(list_secondary_filenames)))
        count += 1
    
        # Load secondary file and change its index to list_cols_nodata
        df_secondary = pd.read_csv("{}/{}".format(dirname_secondary, filename_secondary))
        df_secondary.set_index(list_cols_nodata, inplace=True)
        list_cols_data_sec = list(df_secondary.columns) 
        
        # Ensure that main and secondary files have the same columns
        cols_toAdd = [col for col in list_cols_data_sec if col not in list_cols_data]
        if cols_toAdd != []:
            df_main = pk.df_column_adder(df_main, cols_toAdd)
            
        # Comparing multi-indices to distinguish between data which is already present in the main file and data which is not
        arr_indCompare = df_secondary.index.isin(df_main.index)
        ind_shared = df_secondary.index[arr_indCompare]
        ind_new = df_secondary.index[~arr_indCompare]
            
        # When averaging of data is not required, update data which is already present and append new data
        if flag_average == False:
            # Data which already exists in main file
            if len(ind_shared) != 0:
                # Loop through each data type; remove indices which have NaN in secondary; update main
                for col_dataType in list_cols_data_sec:
                    ind_shared_noNan = df_secondary.loc[ind_shared, col_dataType].dropna().index
                    df_main.loc[ind_shared_noNan, col_dataType] = df_secondary.loc[ind_shared_noNan, col_dataType]        
            # Data which does not yet exist in main file   
            if len(ind_new) != 0:
                df_main = pd.concat([df_main, df_secondary.loc[ind_new]])
                
        # Otherwise, perform average; update data which is already present and append new data
        else:
            # Data which already exists in main file
            if len(ind_shared) != 0:
                # Establish which data types are present by searching for columns with "[dataType]_count" in the secondary file
                list_dataType_avail = [dataType for dataType in list_dataType_all if dataType + "_count" in list_cols_data_sec]
                # Loop through each data type; remove indices which have NaN in secondary (use _count); update main
                for dataType in list_dataType_avail:
                    # Identify columns corresponding to the data type (except for count)
                    cols_dataType = [dataType + subname for subname in list_str_avgQuants]
                    # Drop NaNs by looking at dataType + "_count" in secondary file, update shared index
                    ind_shared_noNan = df_secondary.loc[ind_shared, dataType + "_count"].dropna().index
                    # Set up data to be averaged and number of realizations (counts) for each
                    # -> In df_main.loc[ind_shared_noNan, cols_dataType], replace all NaNs with zeros
                    data_main = np.nan_to_num(df_main.loc[ind_shared_noNan, cols_dataType])
                    data_sec = df_secondary.loc[ind_shared_noNan, cols_dataType]
                    count_main = np.nan_to_num(df_main.loc[ind_shared_noNan, dataType + "_count"].values.reshape(-1,1))
                    count_sec = df_secondary.loc[ind_shared_noNan, dataType + "_count"].values.reshape(-1,1)
                    # Perform averages, calculate number of realizations, and update all values
                    if flag_replace == False:
                        df_main.loc[ind_shared_noNan, cols_dataType] = pk.avg_averages(count_main, count_sec, data_main, data_sec)
                        df_main.loc[ind_shared_noNan, dataType + "_count"] = count_main + count_sec 
                    # Otherwise, replace all values completely
                    else:
                        df_main.loc[ind_shared_noNan, cols_dataType] = data_sec
                        df_main.loc[ind_shared_noNan, dataType + "_count"] = count_sec 
                        
            # Data which does not yet exist in main file   
            if len(ind_new) != 0:
                df_main = pd.concat([df_main, df_secondary.loc[ind_new]])
          
    # Restore original indexing of df_main
    df_main.reset_index(inplace=True)
                                
    # Save updated main data to file
    df_main.to_csv("{}/{}.csv".format(dirname_main, filename_main), index=False)
    #df_data_combined.to_csv("{}/{}.csv".format(dirname_main, filename_main), index=False)       

if __name__ == "__main__":
    
    # ------------------------ #
    # Operation specifications #
    # ------------------------ #

    # Features
    flag_cluster = False # When cluster mode is active, scan through all output .csv files within designated folders and feed each one into the combiner
    flag_average = True # Enable averaging. This should be turned on for results with noise
    flag_time = False # Append a date/time string to end of resultant main data file
    flag_delete = False # Delete all secondary files after combining
    flag_other = False # Combine other data corresponding to full simulations
    flag_replace = True # If an existing data file is found, its data is replaced as opposed to being appended to

    # ----------------------- #
    # Protocol specifications #
    # ----------------------- #

    # Tuning function
    # -> LIN: Linear
    # -> SMOOTH: Smooth (sin^2)
    # -> FQD: Fast-QUAD (adapted from Felix's/Bill C.'s paper)
    # -> SHARP: Sharp. At critical point, the energy changes sharply as ~|t-t*|
    # -> TRIB: Tri-region, B-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a cubic function to interpolate between the regions
    # -> TRIC: Tri-region, C-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a Gaussian kernel to smoothen tuning function
    # -> TRID: Tri-region, D-variant. Custom tuning function. Uses a Gaussian kernel to smoothen tuning function
    # -> TRIE: Tri-region, E-variant. Same as D-variant, but introduces a scale factor f = slope_mid*s_time_m. s_time_m is eliminated as a parameter in favour of this scale factor
    str_tuning_choice = "SMOOTH"
    
    # Disorder
    # -> NONE: No disorder
    # -> UCUN: Uncorrelated, uniformly distributed
    # -> UCND: Uncorrelated, normally/Gaussian distributed
    # -> GCUN: Gaussian correlated, uniformly distributed
    # -> GCND: Gaussian correlated, normally/Gaussian distributed
    # -> ECND: Exponentially correlated, normally/Gaussian distributed
    # -> SCND: Simply correlated, normally/Gaussian distributed
    str_disorder_choice = "NONE"

    # Noise
    # -> NONE: Default, no noise
    # -> WHT: White noise
    # -> WHTC: White noise; with cutoff frequencies, define noise power
    # -> WHTCB: White noise; with cutoff frequencies, define psd amplitude
    # -> 1F1: 1/f noise; with cutoff frequencies, define noise power
    # -> 1F1B: 1/f noise; with cutoff frequencies, define psd amplitude
    # -> SIN: Single mode noise; specified amplitude, phase, and frequency
    # -> SINAV: Single mode noise; same as SIN but results are phase averaged "on the fly"
    str_noise_choice = "WHTCB"
    
    # -------------------------- #
    # File names and directories #
    # -------------------------- #
    
    # Directories
    # -> Note that for the dictionaries below, "main" is taken care of by the key "data_average" while "secondary" is taken care of the key "data". In other words:
    # -> dict_directories["data_average"] = main
    # -> dict_directories["data"] = secondary
    dirname_main = "data_main"
    dirname_secondary = "data_main/cluster"
    
    # Designators
    str_desig_main = "CTL_"
    str_desig_secondary = ""
    
    # Secondary file list (names must end with .csv extension)
    list_secondary_filenames_user = ["CDR_SMOOTH_NONE_WHTCB_20250207_0957.csv"]
    #list_secondary_filenames_user = [str_tuning_choice + "_" + str_disorder_choice + "_" + str_noise_choice + ".csv"]
    
    """
    END OF PARAMETER SPECIFICATIONS
    """
    
    # --------------------------------------- #
    # Argument parser for command line inputs #
    # --------------------------------------- #
    
    # Command line arguments and parameters
    parser = ap.ArgumentParser(description="Specifications and parameters for TLS combiner")
    parser.add_argument('--flag_cluster', type=str, dest='flag_cluster')
    parser.add_argument('--flag_average', type=str, dest='flag_average')
    parser.add_argument('--flag_time', type=str, dest='flag_time')
    parser.add_argument('--flag_delete', type=str, dest='flag_delete')
    parser.add_argument('--flag_other', type=str, dest='flag_other')
    parser.add_argument('--tuning', type=str, dest='tuning')
    parser.add_argument('--disorder', type=str, dest='disorder')
    parser.add_argument('--noise', type=str, dest='noise')
    parser.add_argument('--dirmain', type=str, dest='dirmain')
    parser.add_argument('--dirsec', type=str, dest='dirsec')
    parser.add_argument('--strdesig_main', type=str, dest='strdesig_main')
    parser.add_argument('--strdesig_sec', type=str, dest='strdesig_sec')
    parser.add_argument('--list_files_sec', nargs="*", type=str, dest='list_files_sec')
    dict_args_cmdline = vars(parser.parse_args())    
    
    dict_args_default = {'flag_cluster':flag_cluster, 'flag_average':flag_average, 'flag_time':flag_time, 'flag_delete':flag_delete, 'flag_other':flag_other, 'tuning':str_tuning_choice, 'disorder':str_disorder_choice, 'noise':str_noise_choice, 'dirmain':dirname_main, 'dirsec':dirname_secondary, 'strdesig_main':str_desig_main, 'strdesig_sec':str_desig_secondary, 'list_files_sec':list_secondary_filenames_user}
    
    for key in dict_args_default.keys():
        if dict_args_cmdline[key] != None:
            if dict_args_cmdline[key] == "true":
                dict_args_cmdline[key] = True
            if dict_args_cmdline[key] == "false":
                dict_args_cmdline[key] = False
            if dict_args_cmdline[key] == "NONE":
                if key not in ["tuning", "disorder", "noise"]:
                    dict_args_cmdline[key] = ""
            dict_args_default[key] = dict_args_cmdline[key]
    
    # Redefine arguments and parameters
    flag_cluster = dict_args_default["flag_cluster"]
    flag_average = dict_args_default["flag_average"]
    flag_time = dict_args_default["flag_time"]
    flag_delete = dict_args_default["flag_delete"]
    flag_other = dict_args_default["flag_other"]
    str_tuning_choice = dict_args_default["tuning"]
    str_disorder_choice = dict_args_default["disorder"]
    str_noise_choice = dict_args_default["noise"]
    dirname_main = dict_args_default["dirmain"]
    dirname_secondary = dict_args_default["dirsec"]
    str_desig_main = dict_args_default["strdesig_main"]
    str_desig_secondary = dict_args_default["strdesig_sec"]
    list_secondary_filenames_user = dict_args_default['list_files_sec']
    
    # ------------ #
    # Dictionaries #
    # ------------ #
    
    dict_operations = {"average":flag_average, "other":flag_other}
    dict_specifications = {"tuning":str_tuning_choice, "disorder":str_disorder_choice, "noise":str_noise_choice}
    dict_files = {"dir_main":dirname_main, "dir_secondary":dirname_secondary, "str_desig_main":str_desig_main, "str_desig_secondary":str_desig_secondary}
    
    # -------------------- #
    # Secondary data files #
    # -------------------- #
    
    # If cluster mode is off, allow the user to specific which secondary files should be combined
    if flag_cluster == False:
        list_secondary_filenames = list_secondary_filenames_user
        #filename_secondary = str_desig_secondary + str_tuning_choice + "_" + str_noise_choice
    # Otherwise, when cluster mode is on, search through the data folder and retrieve all ouputs of all jobs, as long as they follow by the prescribed format: STR_DESIG_TUNING_DISORDER_NOISE_ID[JOBID]_[TASKID]
    else:
        if flag_other == True:
            str_secfile_desig = str_desig_secondary + str_tuning_choice + "_" + str_disorder_choice + "_" + str_noise_choice + "_OT" + "_ID"
        else:
            str_secfile_desig = str_desig_secondary + str_tuning_choice + "_" + str_disorder_choice + "_" + str_noise_choice + "_ID"
        list_dataFiles_all = os.listdir(dirname_secondary)
        list_secondary_filenames = [dataFile for dataFile in list_dataFiles_all if str_secfile_desig in dataFile]
        
    # ------------ #
    # Run combiner #
    # ------------ #
    
    data_combiner(dict_operations, dict_specifications, dict_files, list_secondary_filenames, flag_tmAppend=flag_time, flag_replace=flag_replace)
    
    # ---------------------- #
    # Delete secondary files #
    # ---------------------- #
    
    if flag_cluster == True and flag_delete == True:
        for file_secondary in list_secondary_filenames:
            os.remove("{}/{}".format(dirname_secondary, file_secondary))
        