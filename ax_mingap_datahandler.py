"""
Auxiliary script: minimum gap data handler

Features:
    - Takes raw minimum gap data, usually from the cluster, and appends it to the current roster of data available
"""

# ------- #
# Modules #
# ------- #

import numpy as np
import os as os

# --------- #
# Functions #
# --------- #

# Takes as input a file string parameter and determines numerical values of parameters
# -> Meant to be a temporary function to fix data file names issues before changes were made to current scripts
# -> For example: '_n1_L60_R60_lp60_D06_w30_uL28_uRI32_uRF28_mD06_cL600.npy'
# -> Several simple assumptions are made: 
# ---> Integer parameters are converted to numerical values as is
# ---> Float parameters are converted to a float numerical value. This is done by assuming that the decimal needs to always follow the first digit. Of course this is not always true but for the parameters that used, it is
# ---> The exception to the above is the correlation length, which has its decimal before the last digit
def str_params_reversal(str_params):
    # Integer parameters
    list_pars_int = ["_n", "_L", "_R", "_lp"]
    # Float parameters
    list_pars_float = ["_D", "_w", "_uL", "_uRI", "_uRF", "_mD"]
    # Other
    list_pars_other = ["_cL"]
    # List of numerical values for each parameter
    arr_value_pars = []
    # Loop through integer parameters and append to list the numerical values    
    for pars_int in list_pars_int:
        if pars_int not in str_params:
            continue
        else:
            ind_val_begin = str_params.find(pars_int) + len(pars_int)
            ind_val_end = str_params.find("_", ind_val_begin)
            if ind_val_end == -1:
                ind_val_end = str_params.find(".", ind_val_begin)
            val = int(str_params[ind_val_begin:ind_val_end])
            arr_value_pars.append(val)
    # Loop through float parameters and do the same
    for pars_float in list_pars_float:
        if pars_float not in str_params:
            continue
        else:
            ind_val_begin = str_params.find(pars_float) + len(pars_float)
            ind_val_end = str_params.find("_", ind_val_begin)
            if ind_val_end == -1:
                ind_val_end = str_params.find(".", ind_val_begin)
            val_str = str_params[ind_val_begin:ind_val_end]
            val_str_decimal = "{}.{}".format(val_str[0], val_str[1:])
            val = float(val_str_decimal)
            arr_value_pars.append(val)    
    # Loop through other parameters and do the same
    for pars_other in list_pars_other:
        if pars_other not in str_params:
            continue
        else:
            ind_val_begin = str_params.find(pars_other) + len(pars_other)
            ind_val_end = str_params.find("_", ind_val_begin)
            if ind_val_end == -1:
                ind_val_end = str_params.find(".", ind_val_begin)
            val_str = str_params[ind_val_begin:ind_val_end]
            val_str_decimal = "{}.{}".format(val_str[:-1], val_str[-1])
            val = float(val_str_decimal)
            arr_value_pars.append(val)    
    return np.array(arr_value_pars)



# -------------------- #
# Flags and parameters #
# -------------------- #

# Flags
flag_replace = True # If an existing data file is found, its data is replaced as opposed to being appended to

# Protocol specifications
str_tuning = "SMT"
str_disorder = "UCND"
str_noise = "NONE"

# String inserted after the jobid, if desired
#str_custom = "SMT_UCND_"
str_custom = ""

# Directories
dir_rawdata = "data_mingap/raw"
dir_managedData = "data_mingap/managed"

"""
END OF PARAMETER SPECIFICATIONS
"""

# --------- #
# Variables #
# --------- #

# Handle the protocol specification string
str_protocolspec = str_tuning + "_" + str_disorder + "_" + str_noise + "_"
str_protocolspec = str_protocolspec.replace("NONE_", "")
count_underscore = str_protocolspec.count("_")

# --------------- #
# File management #
# --------------- #

# Loop over contents of folder containing raw data
# -> Note: an error will occur if a .npz file is empty
for datafile in os.listdir(dir_rawdata):
    # Skip files that are not of the protocol specifications 
    if str_protocolspec not in datafile:
        continue
    else:
        # Remove the jobid and arrayid from the file name to create a new file name
        ind_parsbegin = datafile.find("_1_" + str_protocolspec) + 3
        datafile_new = "mingap" + "_" + str_custom + datafile[ind_parsbegin:]
        # Load data and list of numerical values coresponding to each parameter
        with np.load("{}/{}".format(dir_rawdata, datafile), allow_pickle=True) as data_raw_temp:
            data_raw = np.copy(data_raw_temp['data'])
            arr_params = np.copy(data_raw_temp['params'])
            
        #arr_params = str_params_reversal("_" + datafile[ind_parsbegin:])
        # ***Change from .npy to npz later. Also change how saving, appending work with new .npz files
        #with np.load("{}/{}".format(dir_rawdata, datafile)) as data_raw_temp:
        #data_raw_temp = np.load("{}/{}".format(dir_rawdata, datafile), allow_pickle=True)
        #data_raw = np.copy(data_raw_temp)
            
        # If the data file doesn't already exist, create it
        if os.path.exists("{}/{}".format(dir_managedData, datafile_new)) == False:
            np.savez("{}/{}".format(dir_managedData, datafile_new), data=data_raw, params=arr_params)
        # Otherwise, load the existing data 
        else:
            if flag_replace == True:
                np.savez("{}/{}".format(dir_managedData, datafile_new), data=data_raw, params=arr_params)
            else:
                with np.load("{}/{}".format(dir_managedData, datafile_new), allow_pickle=True) as data_managed_temp:
                    data_managed = np.copy(data_managed_temp["data"])
                data_managed_combined = np.concatenate((data_managed, data_raw))
                np.savez("{}/{}".format(dir_managedData, datafile_new), data=data_managed_combined, params=arr_params)
            
        