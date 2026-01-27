"""
Auxiliary script: minimum gap averager

Features:
    - Takes minimum gap data corresponding to desired protocol specifications and calculates the average along with the average of the square
    - Saves all calculated quantities into a .csv file, similar to what is done for the diabatic errors
"""

# ------- #
# Modules #
# ------- #

import numpy as np
import os as os
import pandas as pd
import pk_functions as pk

# -------------------- #
# Flags and parameters #
# -------------------- #

# Flags
flag_replace = True # If an existing data is found in the .csv file, replace the data as opposed to skipping it

# Protocol specifications
str_tuning = "SMT"
str_disorder = "UCND"
str_noise = "NONE"

# Number of realizations
nbr_realizations = 1500

# Directories
dir_managedData = "data_mingap/managed"
dir_averagedData = "data_mingap/averaged"

"""
END OF PARAMETER SPECIFICATIONS
"""

# --------- #
# Variables #
# --------- #

# String attached to the tuning function (in case of mismatch)
if str_tuning == "SMT":
    str_tuning_act = "SMOOTH"
else:
    str_tuning_act = str_tuning
    
# Handle the protocol specification string
str_protocolspec = str_tuning + "_" + str_disorder + "_" + str_noise + "_"
str_protocolspec = str_protocolspec.replace("NONE_", "")
    
# File name
filename = str_tuning_act + "_" + str_disorder + "_" + str_noise
    
# ------------ #
# Dictionaries #
# ------------ #

dict_specifications = {"tuning":str_tuning_act, "disorder":str_disorder, "noise":str_noise}
dict_operations_fullsim = {"mingap":False, "velocity":False, "mingap_sep":True}

dict_textstr = {"tau":"tI", "n_steps":"n", "L":"L", "R":"R", "lp":"lp", "Delta":"D", "w":"w", "muLeft":"uL", "muRightStart":"uRI", "muRightEnd":"uRF", "s_time_init":"sTI", "s_time_mid":"sTM", "slope_mid":"sM", "scale_factor":"sF", "disorder_ratio":"mD", "length_corr":"cL", "noise_ratio":"mN", "noise_psd_ampl":"aN", "w_cutoff_low":"wCL", "w_cutoff_high":"wCH"}

# ------------------------ #
# Calculations and storage #
# ------------------------ #

# Create a .csv file if it does not already exist
pk.create_csvfile_other(dict_specifications, dict_operations_fullsim, dir_averagedData, filename, flag_avg=True, flag_desig=False)

# Load .csv
df_mingap = pd.read_csv("{}/{}.csv".format(dir_averagedData, filename))

# Columns of parameters
cols_parameters = pk.create_csvcolumns_nodata(dict_specifications)
cols_parameters.remove("tau")
cols_parameters.append("mingap_count")

# Set in the index of df_mingaps to be the above column of parameters
df_mingap.set_index(cols_parameters, inplace=True)

# Loop over contents of folder containing managed data
list_files_all = os.listdir(dir_managedData)
list_files_spec = [filename for filename in list_files_all if str_protocolspec in filename]
counter=0
for datafile in list_files_spec:
    counter=counter+1
    print("File:", counter, "/", len(list_files_spec))
    # Load parameters and data 
    with np.load("{}/{}".format(dir_managedData, datafile), allow_pickle=True) as file:
        arr_parameter = np.copy(file["params"])
        arr_data = np.copy(file["data"])
    # Use the parameter array to see if corresponding data already exists in .csv file
    # -> Correspond the numerical values to the string
    list_text_str_avail = [key for key in list(dict_textstr.keys()) if "_" + dict_textstr[key] in datafile]
    dict_str_pars_avail = dict(zip(list_text_str_avail, arr_parameter))
    # -> Reorder arr_parameter according to cols_parameters
    arr_params_reorder = [[dict_str_pars_avail[key]] for key in cols_parameters if key in list(dict_textstr.keys())]
    arr_params_reorder.insert(0, [False])
    arr_params_reorder.append([nbr_realizations])
    # -> Make a multi-index and use it to identify whether or not data is present in .csv file
    datafile_index = pd.MultiIndex.from_arrays(arr_params_reorder, names=tuple(cols_parameters))
    if datafile_index[0] not in df_mingap.index:
        # Calculate averages and store to a data frame
        df_datafile = pd.DataFrame([[np.average(arr_data[:nbr_realizations]), np.average(arr_data[:nbr_realizations]**2)]], index=datafile_index, columns=["mingap_avg", "mingap_avg_sqr"])
        df_mingap = pd.concat([df_mingap, df_datafile])
    else:
        if flag_replace == True:
            #print("Y")
            df_mingap.loc[datafile_index[0], "mingap_avg"] = np.average(arr_data[:nbr_realizations])
            df_mingap.loc[datafile_index[0], "mingap_avg_sqr"] = np.average(arr_data[:nbr_realizations]**2)
        else:
            continue
        
# Reset index of data frame and save to file
df_mingap.reset_index(inplace=True)
df_mingap.to_csv("{}/{}.csv".format(dir_averagedData, filename), index=False)
        

        
        



