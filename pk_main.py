"""
Kitaev chain: Piano key simulations
Main

Features:
    - Establishes specifications and parameters for protocol to be simulated
    - Runs the simulation(s) of the protocol requested 
    - Saves simulation results to appropriate .csv files and .npz files
"""

# ------- #
# Modules #
# ------- #

import numpy as np
import os as os
import pandas as pd
import pk_functions as pk
import pk_simulation as pk_sim
import time as tm

# --------------------------------------------------------- #
# Debugging/testing/benchmarking at the beginning of script #
# --------------------------------------------------------- #

# Begin timer 
time_start = tm.time()

# Machine parameters
# -> LCL: Personal PC
# -> LAP: Personal Laptop
# -> NVL: Narval
# -> BLG: Beluga
# -> GRM: Graham
# -> CDR: Cedar
str_machine_choice = "LCL"

# ------------------------ #
# Operation specifications #
# ------------------------ #

# Main features
flag_cluster = False # Cluster mode: all data files will have parameter string appended
flag_onesim = False # Calculates the diabatic error vs. time in a single simulation (note: this will break the loop after one iteration -- only the first tau value in the array will be used)

# Full simulation features
flag_fullsim_diabErr = True # Diabatic error, uses initial and compared states of total Hamiltonian (TT)
flag_fullsim_diabErr_cc = False # Diabatic error, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)
flag_fullsim_diabErr_ct = False # Diabatic error, uses initial state of clean Hamiltonian and compared stated of instantaneous total Hamiltonian (CT)
flag_fullsim_transProb = False # Transition probability, uses initial and compared states of total Hamiltonian (TT)
flag_fullsim_transProb_cc = False # Transition probability, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)
flag_fullsim_mingap = False # Calculate the minimum gap of a simulation
flag_fullsim_velocity = False # Calculate the effective, normalized Landau-Zener velocity

# Full simulation features, separate files
flag_fullsim_sep_mingap = True # Calculate the minimum gap of a simulation and save to corresponding .npy file. Subsequent simulations performed with the same parameters will *append* minimum gaps to this file as opposed to replacing them

# Single simulation features
flag_onesim_diabErr = False # Diabatic error vs. t, uses initial and compared states of total Hamiltonian (TT)
flag_onesim_diabErr_cc = False # Diabatic error vs. t, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)
flag_onesim_diabErr_ct = False # Diabatic error vs. t, uses initial state of clean Hamiltonian and compared stated of instantaneous total Hamiltonian (CT)

# (WIP, these will need to be saved to separate files)
flag_onesim_instEnergy = False # Instantaneous energy vs. t of total Hamiltonian
flag_onesim_instEnergy_c = False # Instantaneous energy vs. t of clean Hamiltonian
flag_onesim_exptEnergy = False # Expectation value of total Hamiltonian vs. t
flag_onesim_occupation = False # Occupation numbers vs. t
flag_onesim_matElement = False # Matrix elements vs. t of noise in clean inst. eigenbasis
flag_onesim_transProbs = False # Transition probabilities vs. t, uses initial and compared states of total Hamiltonian (TT)
flag_onesim_transProbs_cc = False # Transition probabilities vs. t, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)

# (WIP, these will need to be saved to separate files)
flag_onesim_instEigVec = False # Instantaneous eigenvectors vs. t
flag_onesim_evolEigVec = False # Time-evolved eigenvectors vs. t

# Data
flag_savedata = True # Save simulation results (no longer active --- saving is now automatic)

# Other
flag_Nfix = False # Fix the number of time steps
flag_timeoverride = False # The total times tau_init and tau_final are no longer relative to tau_rel
flag_SVD = False # Enable SVD method in calculation of unitary during the calculation
flag_SVD_END = True # Enable the SVD method at the end of the calculation
flag_print_info = True # Enable printing of protocol parameters and specifications
flag_print_time = True # Enable printing of run-time
flag_print_real = True # Enable printing of number of realizations
flag_benchmarks = False # Save real time/simulation time step to file for benchmarking purposes

# ----------------------- #
# Protocol specifications #
# ----------------------- #

# Tuning function
# -> LIN: Linear
# -> SMOOTH: Smooth (sin^2)
# -> FQD: Fast-QUAD (adapted from Felix's/Bill C.'s paper)
# -> SHARP: Sharp. At critical point, the energy changes sharply as ~|t-t*|
# -> TRI: Tri-region. Custom tuning function which interpolates (cubic-ly) between regions
# -> TRI: Tri-region. Custom tuning function which interpolates (cubic-ly) between regions
# -> TRIB: Tri-region, B-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a cubic function to interpolate between the regions
# -> TRIC: Tri-region, C-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a Gaussian kernel to smoothen tuning function
# -> TRID: Tri-region, D-variant. Custom tuning function. Uses a Gaussian kernel to smoothen tuning function
# -> TRID2: Tri-region, D-variant-double. Custom tuning function. Uses a Gaussian kernel to smoothen tuning function. This convolution is performed twice
# -> TRIE: Tri-region, E-variant. Same as D-variant, but introduces a scale factor f = slope_mid*s_time_m. s_time_m is eliminated as a parameter in favour of this scale factor
# -> SNP: Tri-region, sin^2. Custom tuning function. Similar to TRI with a flat middle region and sin^2 functions used in transition regions
# -> SNPB: Tri-region, sin^2, B-variant. Same as SNP with the tuning function smoothened using a convolution with a Gaussian
str_tuning_choice = "SMOOTH"

# Disorder
# -> NONE: No disorder
# -> UCUN: Uncorrelated, uniformly distributed
# -> UCND: Uncorrelated, normally/Gaussian distributed
# -> GCUN: Gaussian correlated, uniformly distributed
# -> GCND: Gaussian correlated, normally/Gaussian distributed
# -> ECND: Exponentially correlated, normally/Gaussian distributed
# -> SCND: Simply correlated, normally/Gaussian distributed
str_disorder_choice = "UCND"

# Noise
# -> NONE: Default, no noise
# -> WHT: White noise
# -> WHTC: White noise; with cutoff frequencies, define noise power
# -> WHTCB: White noise; with cutoff frequencies, define psd amplitude
# -> 1F1: 1/f noise; with cutoff frequencies, define noise power
# -> 1F1B: 1/f noise; with cutoff frequencies, define psd amplitude
# -> SIN: Single mode noise; specified amplitude, phase, and frequency
# -> SINAV: Single mode noise; same as SIN but results are phase averaged "on the fly"
# ---> Note: SINAV perform averaging in the same way as for typical noise and disorder
str_noise_choice = "NONE"

# ------------------- #
# Protocol parameters #
# ------------------- #

# System
L = 30
R = 30
Delta = 0.6
w = 3.0
muLeft = 2.8
muRightStart = 3.2
muRightEnd = 2.8

# Piano keys
lp = R # Total size of piano key (default = R)
n_steps = 1 # Number of piano key steps; note that lp/(n_steps) must be an integer

# Time
tau = 100.0 # Protocol time (must be a float!)
dt = 0.1 # Maximum time step size
Nthresh = 1500 # Minimum number of time steps (default = 1500)
Nfix = 1500 # (if flag_timeoverride == True) Fixed number of time steps

# Tuning functions
# --> For reference, the default slope for SMOOTH is pi/2 (~1.57) in units of tau
s_time_init = 0.0 # (TRI, TRIB, TRIC) Duration of the start/end regions in units of tau
s_time_mid = 0.0 # (TRI, TRIB, TRIC) Duration of the middle region in units of tau
slope_mid = 0.0 # (TRI, TRID) Slope of the middle region
scale_factor = 1.0 # (TRIE) scale factor = slope_mid*s_time_mid

# Averaging
nbr_realizations = 5 # Number of realizations

# Disorder
disorder_ratio = 0.1 # Disorder strength (default: in units of the chemical potential difference)
length_corr = 5.0 # (GCXX, SCXX) Correlation length, (ECXX) decay length

# Noise
noise_ratio = 0.01 # Noise strength (default: in units of the chemical potential difference)
noise_psd_ampl = 5e-4 # (-B variants only) Amplitude of the power spectral density (note: A = 1e-7 corresponds roughly to a variance of 1e-5 for 1/f)
w_cutoff_low = 1e-8 # Low frequency cutoff
w_cutoff_high = 0.85 # High frequency cutoff
tau_noise = 40000 # Total time of signal 
dt_noise = dt # Size of noise time step (default = dt)

# Noise (SIN)
# -> noise_ratio will be used to determine the amplitude
freq_sin = 0.08 # Frequency
phase_sin = 0.0 # Phase (vary between 0 and 2pi)

# ----------------------------------- #
# Calculation quantity specifications #
# ----------------------------------- #

list_ind_instEnerg = [0, 1] # Instantaneous energies of total Hamiltonian
list_ind_instEnerg_c = [0, 1] # Instantaneous energies of clean Hamiltonian

list_ind_exptEnerg = [0, 1] # Expectation values of H(t) to calculate
list_ind_occupation = [0, 1] # Occupation numbers to calculate
list_ind_transProb = [[1], [1, 2]] # Transition probabilities to calculate, uses initial and compared states of total Hamiltonian (TT)
list_ind_transProb_cc = [[1], [1, 2]] # Transition probabilities to calculate, uses initial and compared states of clean Hamiltonian (CC)

list_ind_instEigVec = [0, 1] # Instantaneous eigenvectors to calculate
list_ind_evolEigVec = [0, 1] # Time-evolved eigenvectors to calculate

# ---------------- #
# Other parameters #
# ---------------- #

NSamp = 500 # Number of points to be sampled and saved to file for single simulations
str_desig = "" # Name of any special designator attached to the beginning of data files (i.e. the cluster task ID of the job) 
str_desig2 = "" # Name of any second special designator
list_disorder_noAveraging = ["NONE"] # List of disorder protocols where averaging is not performed
list_noise_noAveraging = ["NONE", "SIN"] # List of noise protocols where averaging is not performed
list_str_avgQuants = ["_avg", "_avg_sqr", "_logAvg", "_logAvg_sqr"] # List of subquantities to calculate when averaging

# -------------------------- #
# Directories and file names #
# -------------------------- #

# Directories
dirname_data = "data"
dirname_data_average = "data_average"
dirname_data_onesim = "data_onesim"
dirname_data_onesim_average = "data_onesim_average"
dirname_data_mingap = "data_mingap"

"""
END OF PARAMETER SPECIFICATIONS
"""

# --------------------------------------------- #
# Dictionaries of specifications and parameters #
# --------------------------------------------- #

dict_machine = {"machine":str_machine_choice}
dict_operations_main = {"cluster":flag_cluster, "onesim":flag_onesim, "average":False}
dict_operations_fullsim = {"diabErr":flag_fullsim_diabErr, "diabErr_cc":flag_fullsim_diabErr_cc, "diabErr_ct":flag_fullsim_diabErr_ct, "transProb":flag_fullsim_transProb, "transProb_cc":flag_fullsim_transProb_cc, "mingap":flag_fullsim_mingap, "velocity":flag_fullsim_velocity, "mingap_sep":flag_fullsim_sep_mingap}
dict_operations_onesim = {"diabErr":flag_onesim_diabErr, "diabErr_cc":flag_onesim_diabErr_cc, "diabErr_ct":flag_onesim_diabErr_ct, "instEnergy":flag_onesim_instEnergy, "instEnergy_c":flag_onesim_instEnergy_c, "exptEnergy":flag_onesim_exptEnergy, "occupation":flag_onesim_occupation, "matElement":flag_onesim_matElement, "transProbs":flag_onesim_transProbs, "transProbs_cc":flag_onesim_transProbs_cc, "instEigVec":flag_onesim_instEigVec, "evolEigVec":flag_onesim_evolEigVec}
dict_operations_other = {"savedata":flag_savedata, "Nfix":flag_Nfix, "timeoverride":flag_timeoverride, "SVD":flag_SVD, "SVD_END":flag_SVD_END}
dict_specifications = {"tuning":str_tuning_choice, "disorder":str_disorder_choice, "noise":str_noise_choice}
dict_specifications_clean = {"tuning":str_tuning_choice, "disorder":"NONE", "noise":"NONE"}
dict_protocol = {"L":L, "R":R, "Delta":Delta, "w":w, "muLeft":muLeft, "muRightStart":muRightStart, "muRightEnd":muRightEnd, "lp":lp, "n_steps":n_steps, "tau":tau, "dt":dt, "s_time_init":s_time_init, "s_time_mid":s_time_mid, "slope_mid":slope_mid, "scale_factor":scale_factor, "Nthresh":Nthresh, "Nfix":Nfix, "nbr_realizations":nbr_realizations}
dict_disorder = {"disorder_ratio":disorder_ratio, "length_corr":length_corr}
dict_noise = {"noise_ratio":noise_ratio, "noise_psd_ampl":noise_psd_ampl, "w_cutoff_low":w_cutoff_low, "w_cutoff_high":w_cutoff_high, "tau_noise":tau_noise, "dt_noise":dt_noise, "freq_sin":freq_sin, "phase_sin":phase_sin}
dict_fullsim = {"transProbs":list_ind_transProb, "transProbs_cc":list_ind_transProb_cc}
dict_onesim = {"instEnerg":list_ind_instEnerg, "instEnerg_c":list_ind_instEnerg_c, "exptEnerg":list_ind_exptEnerg, "occupation":list_ind_occupation, "transProbs":list_ind_transProb, "transProbs_cc":list_ind_transProb_cc, "instEigVec":list_ind_instEigVec, "evolEigVec":list_ind_evolEigVec}
dict_other = {"NSamp":NSamp, "str_desig":str_desig, "str_desig2":str_desig2}
dict_directories = {"data":dirname_data, "data_average":dirname_data_average, "data_onesim":dirname_data_onesim, "data_onesim_average":dirname_data_onesim_average, "data_mingap":dirname_data_mingap}
dict_strings = {"EMPTY":0}

# Update dictionaries with command line inputs
dict_machine, dict_operations_main, dict_operations_fullsim, dict_operations_onesim, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other, dict_directories = pk.argument_parser(dict_machine, dict_operations_main, dict_operations_fullsim, dict_operations_onesim, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other, dict_directories)

# Text string of parameters and designations
str_params = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other)[0]
str_params_onesim = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=True)[0]
str_params_onesim_clean = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications_clean, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=True)[0]
str_params_onesim_spec = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_noTau=True)[0]
str_params_onesim_spec_clean = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications_clean, dict_protocol, dict_disorder, dict_noise, dict_other, flag_noTau=True)[0]

# Print all parameters
if flag_print_info == True:
    print("")
    for dictionary in [dict_operations_main, dict_operations_fullsim, dict_operations_onesim, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other, dict_directories]:
        print(dictionary)
    print("")

# File names
filename_data = dict_other["str_desig"] + dict_specifications["tuning"] + "_" + dict_specifications["disorder"] + "_" + dict_specifications["noise"]
filename_data_other = dict_other["str_desig"] + dict_specifications["tuning"] + "_" + dict_specifications["disorder"] + "_" + dict_specifications["noise"] + "_OT"
filename_data_onesim = str_params_onesim
filename_data_onesim_clean = str_params_onesim_clean

# Create a multi-index for selecting data in data frames
multiIndex_pars = pk.create_csvmultiIndex(dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise)
multiIndex_pars_other = multiIndex_pars.droplevel("tau")

# -------------------------- #
# Folder and file management #
# -------------------------- #

# (Outdated) If cluster mode is on, append str_params to dirname_data and dirname_data_average
# If cluster mode is on, append jobid_taskid to dirname_data and dirname_data_average
if dict_operations_main["cluster"] == True:
    #filename_data = dict_specifications["tuning"] + "_" + dict_specifications["noise"] + str_params
    filename_data = dict_specifications["tuning"] + "_" + dict_specifications["disorder"] + "_" + dict_specifications["noise"] + "_ID" + dict_other["str_desig"]
    filename_data_other = dict_specifications["tuning"] + "_" + dict_specifications["disorder"] + "_" + dict_specifications["noise"] + "_OT" + "_ID" + dict_other["str_desig"]
  
# Create folders and data files
# -> Full simulations
pk.create_csvfile(dict_specifications, dict_operations_fullsim, dict_fullsim, dict_directories["data"], filename_data, flag_avg=False)
if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:    
    pk.create_csvfile(dict_specifications, dict_operations_fullsim, dict_fullsim, dict_directories["data_average"], filename_data, flag_avg=True)
    
# -> Full simulations, other
pk.create_csvfile_other(dict_specifications, dict_operations_fullsim, dict_directories["data"], filename_data_other, flag_avg=False)
if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:    
    pk.create_csvfile_other(dict_specifications, dict_operations_fullsim, dict_directories["data_average"], filename_data_other, flag_avg=True)
    
# -> Full simulations, separate files
# --> Minimum gap
if dict_operations_fullsim["mingap_sep"] == True:
    
    # Check that folders exist; if a folder does not exist, create it
    if os.path.exists(dict_directories["data_mingap"]) == False:
        os.makedirs(dict_directories["data_mingap"])
        
    # Name and parameter list of data file
    filename_mingap_sep, arr_params = pk.str_params(dict_operations_main, dict_operations_other, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_other, flag_NSamp=False, flag_noTau=True)
    
    # Create an array of zeros equal to the number of realizations. The minimum gap values will be stored in this array
    arr_data_mingaps_sep = np.zeros(dict_protocol["nbr_realizations"], dtype=float)
        
    # Check that .npz file exists in folder; if a file does not exist, create it
    # -> Otherwise, load the file, append arr_data_mingaps_sep
    if os.path.exists("{}/mingap{}.npz".format(dict_directories["data_mingap"], filename_mingap_sep)) == False:
        np.savez("{}/mingap{}.npz".format(dict_directories["data_mingap"], filename_mingap_sep), data=arr_data_mingaps_sep, params=arr_params)
        mingaps_sep_indoff = 0
    else:
        with np.load("{}/mingap{}.npz".format(dict_directories["data_mingap"], filename_mingap_sep)) as file:
            arr_data_mingaps_sep_prev = file['data']
            mingaps_sep_indoff = arr_data_mingaps_sep_prev.shape[0]
            arr_data_mingaps_sep = np.concatenate((arr_data_mingaps_sep_prev, arr_data_mingaps_sep))

# -> Single simulations
if dict_operations_main["onesim"] == True:
    pk.create_csvfile_onesim(dict_operations_onesim, dict_onesim, dict_directories["data_onesim"], filename_data_onesim)
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        pk.create_csvfile_onesim(dict_operations_onesim, dict_onesim, dict_directories["data_onesim_average"], filename_data_onesim, flag_avg=True)
        
# --------------- #
# Load data files #
# --------------- #

# Establish data columns to be used
# -> Full simulations
cols_nodata = pk.create_csvcolumns_nodata(dict_specifications)
list_cols_fullsim = pk.create_csvcolumns_data(dict_operations_fullsim, dict_fullsim, flag_avg=False)
list_cols_fullsim_avg = pk.create_csvcolumns_data(dict_operations_fullsim, dict_fullsim, flag_avg=True)
# -> Full simulations, other
list_cols_fullsim_other = pk.create_csvcolumns_data_other(dict_operations_fullsim, flag_avg=False)
list_cols_fullsim_avg_other = pk.create_csvcolumns_data_other(dict_operations_fullsim, flag_avg=True)
# -> Single simulations
list_cols_onesim = pk.create_csvcolumns_data_onesim(dict_operations_onesim, dict_onesim, flag_avg=False)
list_cols_onesim_avg = pk.create_csvcolumns_data_onesim(dict_operations_onesim, dict_onesim, flag_avg=True)

# Full simulation data
# -> For each data frame, add columns if they are not already present
df_fullsim = pd.read_csv("{}/{}.csv".format(dict_directories["data"], filename_data))
df_fullsim = pk.df_column_adder(df_fullsim, list_cols_fullsim)
if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
    df_fullsim_avg = pd.read_csv("{}/{}.csv".format(dict_directories["data_average"], filename_data))
    df_fullsim_avg = pk.df_column_adder(df_fullsim_avg, list_cols_fullsim_avg)
    
# Full simulation data, other
# -> For each data frame, add columns if they are not already present
df_fullsim_other = pd.read_csv("{}/{}.csv".format(dict_directories["data"], filename_data_other))
df_fullsim_other = pk.df_column_adder(df_fullsim_other, list_cols_fullsim_other)
if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
    df_fullsim_avg_other = pd.read_csv("{}/{}.csv".format(dict_directories["data_average"], filename_data_other))
    df_fullsim_avg_other = pk.df_column_adder(df_fullsim_avg_other, list_cols_fullsim_avg_other)

# Single simulation data
# -> For each data frame, add columns into the data frames if they are not already present
if dict_operations_main["onesim"] == True:
    df_onesim = pd.read_csv("{}/{}.csv".format(dict_directories["data_onesim"], filename_data_onesim))
    df_onesim = pk.df_column_adder(df_onesim, list_cols_onesim)
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        df_onesim_avg = pd.read_csv("{}/{}.csv".format(dict_directories["data_onesim_average"], filename_data_onesim))
        df_onesim_avg = pk.df_column_adder(df_onesim_avg, list_cols_onesim_avg)
        
# Change the indexing of the data frames to the non-data columns
# -> Full simulations
cols_nodata = pk.create_csvcolumns_nodata(dict_specifications)
        
# --------- #
# Run files #
# --------- #

# For benchmarking
time_start_run = np.copy(tm.time())
    
# Loop over number of realizations
if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
    if dict_specifications["noise"] == "SINAV" and dict_specifications["disorder"] == "NONE":
        #nbr_phase = 150
        nbr_phase = 500
        arr_phase_sin = np.linspace(0, 2*np.pi, nbr_phase)
        nbr_max_loops = arr_phase_sin.shape[0]
    else:
        nbr_max_loops = dict_protocol["nbr_realizations"]
else:
    nbr_max_loops = 1

for ind_real in range(0, nbr_max_loops):
    
    # Print statements
    if flag_print_real == True:
        print("Realization number: " + str(ind_real+1))
        
    # Handle variables
    # -> In the case of protocols requiring on-the-fly phase averaging, change the phase for each realization
    if dict_specifications["noise"] == "SINAV":
        dict_noise["phase_sin"] = arr_phase_sin[ind_real]
        
    # Run simulation, obtain results
    results_sim = pk_sim.run_simulation(dict_operations_main, dict_operations_other, dict_operations_fullsim, dict_operations_onesim, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other) 
    
    # Change the indexing of the data frames to the non-data columns
    # -> Full simulations
    cols_nodata = pk.create_csvcolumns_nodata(dict_specifications)
    df_fullsim.set_index(cols_nodata, inplace=True)
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        df_fullsim_avg.set_index(cols_nodata, inplace=True)
    # -> Full simulations, other
    cols_nodata_other = [cols for cols in cols_nodata if cols not in ["tau"]]
    df_fullsim_other.set_index(cols_nodata_other, inplace=True)
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        df_fullsim_avg_other.set_index(cols_nodata_other, inplace=True)
    # -> Single simulations
    if dict_operations_main["onesim"] == True:
        df_onesim.set_index(["time"], inplace=True)
        if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
            df_onesim_avg.set_index(["time"], inplace=True)
    
    # Convert results to data frames
    df_fullsim_results = pd.DataFrame(data=results_sim["fullsim"], index=multiIndex_pars)
    df_fullsim_results_other = pd.DataFrame(data=results_sim["fullsim_other"], index=multiIndex_pars_other)
    if dict_operations_main["onesim"] == True:
        df_onesim_results = pd.DataFrame(data=results_sim["onesim"])
        df_onesim_results.set_index(["time"], inplace=True)
    
    # Indexing conveniences
    ind_sim = multiIndex_pars[0]
    ind_sim_other = multiIndex_pars_other[0]
    
    # For averaging of full simulation results: create a data frame that contains the value, value squared, log(value) and log(value) squared for each quantity calculated
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        df_fullsim_results_forAvg = pd.DataFrame(columns=list_cols_fullsim_avg, index=multiIndex_pars)
        for name_quantity in list_cols_fullsim:
            df_fullsim_results_forAvg.loc[ind_sim, name_quantity + "_avg"] = df_fullsim_results.loc[ind_sim, name_quantity]
            df_fullsim_results_forAvg.loc[ind_sim, name_quantity + "_avg_sqr"] = df_fullsim_results.loc[ind_sim, name_quantity]**2
            df_fullsim_results_forAvg.loc[ind_sim, name_quantity + "_logAvg"] = np.log(df_fullsim_results.loc[ind_sim, name_quantity])
            df_fullsim_results_forAvg.loc[ind_sim, name_quantity + "_logAvg_sqr"] = np.log(df_fullsim_results.loc[ind_sim, name_quantity])**2
            df_fullsim_results_forAvg.loc[ind_sim, name_quantity + "_count"] = int(1)
            
    # For averaging of full simulation results (OTHER): create a data frame that contains the value, value squared, log(value) and log(value) squared for each quantity calculated
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        df_fullsim_results_forAvg_other = pd.DataFrame(columns=list_cols_fullsim_avg_other, index=multiIndex_pars_other)
        for name_quantity in list_cols_fullsim_other:
            # In the case of the velocity, if the curve fit fails, continue the next iteration of the loop
            if results_sim["fullsim_other"][name_quantity] == "FAIL":
                continue
            df_fullsim_results_forAvg_other.loc[ind_sim_other, name_quantity + "_avg"] = df_fullsim_results_other.loc[ind_sim_other, name_quantity]
            df_fullsim_results_forAvg_other.loc[ind_sim_other, name_quantity + "_avg_sqr"] = df_fullsim_results_other.loc[ind_sim_other, name_quantity]**2
            df_fullsim_results_forAvg_other.loc[ind_sim_other, name_quantity + "_logAvg"] = np.log(df_fullsim_results_other.loc[ind_sim_other, name_quantity])
            df_fullsim_results_forAvg_other.loc[ind_sim_other, name_quantity + "_logAvg_sqr"] = np.log(df_fullsim_results_other.loc[ind_sim_other, name_quantity])**2
            df_fullsim_results_forAvg_other.loc[ind_sim_other, name_quantity + "_count"] = int(1)
            
    # For averaging of single simulation results: create a data frame that contains the value, value squared, log(value) and log(value) squared for each quantity calculated
    if dict_operations_main["onesim"] == True:
        if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
            df_onesim_results_forAvg = pd.DataFrame(columns=list_cols_onesim_avg, index=df_onesim.index)
            for name_quantity in list_cols_onesim:
                df_onesim_results_forAvg[name_quantity + "_avg"] = df_onesim_results.iloc[1:][name_quantity]
                df_onesim_results_forAvg[name_quantity + "_avg_sqr"] = df_onesim_results.iloc[1:][name_quantity]**2
                df_onesim_results_forAvg[name_quantity + "_logAvg"] = np.log(df_onesim_results.iloc[1:][name_quantity])
                df_onesim_results_forAvg[name_quantity + "_logAvg_sqr"] = np.log(df_onesim_results.iloc[1:][name_quantity])**2
                df_onesim_results_forAvg[name_quantity + "_count"] = int(1)
    
    # Save full results to data frame
    if ind_sim in df_fullsim.index:
        df_fullsim.loc[ind_sim, list_cols_fullsim] = df_fullsim_results.loc[ind_sim, list_cols_fullsim]
    else:
        df_fullsim = pd.concat([df_fullsim, df_fullsim_results])
        
    # Save full results (OTHER) to data frame
    if ind_sim_other in df_fullsim_other.index:
        df_fullsim_other.loc[ind_sim_other, list_cols_fullsim_other] = df_fullsim_results_other.loc[ind_sim_other, list_cols_fullsim_other]
    else:
        df_fullsim_other = pd.concat([df_fullsim_other, df_fullsim_results_other])
      
    # For averaging of full results: update averaged, full results and save to data frame
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        if ind_sim in df_fullsim_avg.index:
            for name_quantity in list_cols_fullsim:
                # Identitfy columns of subnames to be averaged
                cols_name = [name_quantity + subname for subname in list_str_avgQuants]
                # Set up data to be averaged and number of realizations for each
                fullsim_data = np.nan_to_num(df_fullsim_avg.loc[ind_sim, cols_name])
                fullsim_data_results = df_fullsim_results_forAvg.loc[ind_sim, cols_name]
                nbr_reals_data = np.nan_to_num(df_fullsim_avg.loc[ind_sim, name_quantity + "_count"])
                nbr_reals_data_results = df_fullsim_results_forAvg.loc[ind_sim, name_quantity + "_count"]
                # Perform average
                df_fullsim_avg.loc[ind_sim, cols_name] = pk.avg_averages(nbr_reals_data, nbr_reals_data_results, fullsim_data, fullsim_data_results)
                # Update counts/nbr of realizations
                df_fullsim_avg.loc[ind_sim, name_quantity + "_count"] = nbr_reals_data + nbr_reals_data_results        
        else:
            df_fullsim_avg = pd.concat([df_fullsim_avg, df_fullsim_results_forAvg])
            
    # For averaging of full results (OTHER): update averaged, full results and save to data frame
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        if ind_sim_other in df_fullsim_avg_other.index:
            for name_quantity in list_cols_fullsim_other:
                # In the case of the velocity, if the curve fit fails, continue the next iteration of the loop
                if results_sim["fullsim_other"][name_quantity] == "FAIL":
                    continue
                # Identitfy columns of subnames to be averaged
                cols_name = [name_quantity + subname for subname in list_str_avgQuants]
                # Set up data to be averaged and number of realizations for each
                fullsim_data_other = np.nan_to_num(df_fullsim_avg_other.loc[ind_sim_other, cols_name])
                fullsim_data_results_other = df_fullsim_results_forAvg_other.loc[ind_sim_other, cols_name]
                nbr_reals_data_other = np.nan_to_num(df_fullsim_avg_other.loc[ind_sim_other, name_quantity + "_count"])
                nbr_reals_data_results_other = df_fullsim_results_forAvg_other.loc[ind_sim_other, name_quantity + "_count"]
                # Perform average
                df_fullsim_avg_other.loc[ind_sim_other, cols_name] = pk.avg_averages(nbr_reals_data_other, nbr_reals_data_results_other, fullsim_data_other, fullsim_data_results_other)
                # Update counts/nbr of realizations
                df_fullsim_avg_other.loc[ind_sim_other, name_quantity + "_count"] = nbr_reals_data_other + nbr_reals_data_results_other        
        else:
            df_fullsim_avg_other = pd.concat([df_fullsim_avg_other, df_fullsim_results_forAvg_other])
        
    # Save single results to data frame
    if dict_operations_main["onesim"] == True:
        df_onesim[list_cols_onesim] = df_onesim_results[list_cols_onesim]
        #df_onesim = df_onesim[~df_onesim.index.duplicated(keep="last")]
    
        # For averaging of single results: update averaged, single results and save to data frame
        if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
            if len(df_onesim_results_forAvg.index) == len(df_onesim_avg.index):
                for name_quantity in list_cols_onesim:
                    if name_quantity not in ["noise"]:
                        # Identitfy columns of subnames to be averaged
                        cols_name = [name_quantity + subname for subname in list_str_avgQuants]
                        # Set up data to be averaged and number of realizations for each
                        onesim_data = np.nan_to_num(df_onesim_avg[cols_name].values)
                        onesim_data_results = df_onesim_results_forAvg[cols_name].values
                        nbr_reals_data = np.nan_to_num(df_onesim_avg[name_quantity + "_count"].values.reshape(-1,1))
                        nbr_reals_data_results = df_onesim_results_forAvg[name_quantity + "_count"].values.reshape(-1,1)
                        # Perform average
                        df_onesim_avg[cols_name] = pk.avg_averages(nbr_reals_data, nbr_reals_data_results, onesim_data, onesim_data_results)
                        # Update counts/nbr of realizations
                        df_onesim_avg[name_quantity + "_count"] = (nbr_reals_data + nbr_reals_data_results).reshape(-1) 
            else:
                df_onesim_avg[list_cols_onesim_avg] = df_onesim_results_forAvg[list_cols_onesim_avg]
                #df_onesim_avg = df_onesim_avg[~df_onesim_avg.index.duplicated(keep="last")]
            
    # ------------ #
    # Save to file #
    # ------------ #
    
    # Reset indices of all data frames and save to file
    df_fullsim.reset_index(inplace=True)
    df_fullsim.to_csv("{}/{}.csv".format(dict_directories["data"], filename_data), index=False)
    df_fullsim_other.reset_index(inplace=True)
    df_fullsim_other.to_csv("{}/{}.csv".format(dict_directories["data"], filename_data_other), index=False)
    if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
        df_fullsim_avg.reset_index(inplace=True)
        df_fullsim_avg.to_csv("{}/{}.csv".format(dict_directories["data_average"], filename_data), index=False)
        df_fullsim_avg_other.reset_index(inplace=True)
        df_fullsim_avg_other.to_csv("{}/{}.csv".format(dict_directories["data_average"], filename_data_other), index=False)
    if dict_operations_main["onesim"] == True:
        df_onesim.reset_index(inplace=True)
        df_onesim.to_csv("{}/{}.csv".format(dict_directories["data_onesim"], filename_data_onesim), index=False)
        if dict_specifications["disorder"] not in list_disorder_noAveraging or dict_specifications["noise"] not in list_noise_noAveraging:  
            df_onesim_avg.reset_index(inplace=True)
            df_onesim_avg.to_csv("{}/{}.csv".format(dict_directories["data_onesim_average"], filename_data_onesim), index=False)
            
    # Instantaneous spectrum
    if dict_operations_main["onesim"] == True:
        # -> These are saved in separate files given their nature and size
        if dict_operations_onesim["instEnergy"] == True:
            np.savez("{}/instEnergy{}.npz".format(dict_directories["data_onesim"], str_params_onesim_spec), spec=results_sim["onesim_spec"]["instEnergy"])
        if dict_operations_onesim["instEnergy_c"] == True:
            np.savez("{}/instEnergy{}.npz".format(dict_directories["data_onesim"], str_params_onesim_spec_clean), spec=results_sim["onesim_spec_clean"]["instEnergy"]) 
            
    # Minimum gap, separate file
    if dict_operations_fullsim["mingap_sep"] == True:
        arr_data_mingaps_sep[mingaps_sep_indoff + ind_real] = results_sim["fullsim_other"]["mingap"]
        np.savez("{}/mingap{}.npz".format(dict_directories["data_mingap"], filename_mingap_sep), data=arr_data_mingaps_sep[:mingaps_sep_indoff + ind_real + 1], params=arr_params)
     
    # ----- #
    # Timer #
    # ----- #
        
    # End timer
    time_end = tm.time()
    if flag_print_time == True:
        print("Runtime of simulation {}: ".format(ind_real+1), time_end - time_start, "s")
            
# ---------------------------------------------------- #
# Debugging/testing/benchemarking at the end of script #
# ---------------------------------------------------- #

# End timer
time_end = tm.time()
time_duration = np.copy(time_end - time_start_run)
print("Runtime of all simulations: ", time_end - time_start, "s")

# Save simulation benchmarks
if flag_benchmarks == True:
    
    # Define columns, file/folder names
    cols_benchmarks = ["machine", "size", "realtimePerStep"]
    dir_benchmarks = "benchmarks"
    filedir_benchmarks = "{}/{}.csv".format(dir_benchmarks, filename_data)
    
    # Check that benchmarks file exists
    if os.path.exists(dir_benchmarks) == False:
        os.mkdir(dir_benchmarks)
    if os.path.exists(filedir_benchmarks) == False:
        df_benchmarks_create = pd.DataFrame(columns=cols_benchmarks)
        df_benchmarks_create.to_csv(filedir_benchmarks, index=False)
    
    # Load benchmarks file
    df_benchmarks = pd.read_csv(filedir_benchmarks)
    df_benchmarks.set_index(["machine", "size"], inplace=True)
    
    # Save benchmarks to file
    realtimePerStep = time_duration/(dict_protocol["tau"]/dict_protocol["dt"]/nbr_max_loops)
    index = pd.MultiIndex.from_arrays([[dict_machine["machine"]], [str(dict_protocol["L"] + dict_protocol["R"])]], names=tuple(["machine", "size"]))
    df_benchmarks.loc[index[0], "realtimePerStep"] = np.round(realtimePerStep, 8)
    df_benchmarks.reset_index(inplace=True)
    df_benchmarks.to_csv(filedir_benchmarks, index=False)
    
    
    
    
        
    
    
    
    



