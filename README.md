# Guide and instructions 
# Bill P. Truong 											 

Important: All data may be downloaded through linked Zenodo repository: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18396840.svg)](https://doi.org/10.5281/zenodo.18396840)

This collection consists of the following scripts for producing and managing numerical data:
- pk_functions.py: Contains a suite of functions necessary for the numerical simulation and study of piano key transport. 
- tl_functions.py: Contains a suite of functions necessary for numerical simulations in effective two-level systems; some functions here are used in the scripts for the full piano key simulation/data management.
- pk_simulation.py: Contains a single function that numerically simulates piano key transport. Primarily calculates the time evolution operator and uses it to calculate quantities such as the diabatic error. 
- pk_main.py: Platform for initiating simulations with parameter specifications. Allows for numerical data to be saved and updated. 
- pk_combiner.py: Combines multiple data files together into a single "main" data file or feeds them into existing main data files. Can perform averaging if desired. 
- ax_mingap_datahandler.py: Combines raw minimum gap data into a single file. 
- ax_mingap_averager.py: Takes raw minimum gap data, calculates statistics, and saves/updates to a main file. 
	
The following are scripts which assist in running and automating simulations:
- param_sets/ax_params.sh: Specify parameters for a collection of simulations. Allows for two parameters to be varied. 
- ax_run.sh: Uses the parameters specified in ax_params.sh to run a collection of simulations. 
	
The following are scripts for plotting numerical results:
- plot_diabErr_[].py: Plots diabatic error versus quantity of interest. 
- plot_mingap_[].py: Plots minimum gap statistics versus quantity of interest. 
	
Other scripts:
- ax_linspace.py, ax_logspace.py, ax_geomspace.py: Each of these are intended for use in ax_params.sh to define arrays corresponding to varying parameters. 
- ax_combiner.sh: Uses pk_combiner.py as a backend; provides a quick way to combine data files together without opening pk_combiner.py. 
	
Instructions for use: 
- Running single simulations:
	1. Open pk_main.py
	2. Specify flags in "Operation specifications." For this paper, the most important ones are:
		- flag_fullsim_diabErr: Set to "True" to calculate the accumulated diabatic error by the end of the transport protocol. 
		- flag_fullsim_sep_mingap: Set to "True" to calculate the minimum energy gap of the protocol. 
	3. Specify "Protocol specifications." For this paper, the most relevant ones are:
		- str_tuning_choice: "SMOOTH" (Smooth tuning functions).
		- str_disorder_choice: "NONE" (No disorder), "UCND" (Uncorrelated, normally distributed disorder), "GCND" (Gaussian correlated, normally distributed disorder).
		- str_noise_choice: "NONE" (No noise), "WHTCB" (White noise with cut-off frequencies, noise amplitude specified), "1F1B" (1/f noise with cut-off frequencies, noise amplitude specified), "SINAV" (Single mode with amplitide specified, phase averaged). 
	4. Specify "Protocol parameters" such as chain lengths, values of couplings (all in meV), protocol times, disorder and noise strengths, etc. 
	5. Specify "Directories and file names" where data will be saved. By default, all data will be saved in "data", "data_average" (for disorder/noise), and "data_mingap" (for minimum gaps) folders within the same directory, which are automatically created if they do not already exist. 
	6. Run the script.
	7. Notes on data: 
		- Data contained in files within the "data" folder (as .csv) correspond to the most recent simulation. For example, the results of a simulation with disorder will override any previous result with the same parameter specifications. In essence, the data contained in this folder correspond to single simulations. 
		- Data contained in files within the "data_average" folder (as .csv) are updated as new simulations are performed. For example, the results of a simulation with disorder can either a.) be added as a new data file if it does not already exist or b.) update average quantities within an existing data file. In essence, the data contained in this folder correspond to averages of multiple simulations and are continually updated as more simulations are performed with the same parameter set.
		- Data contained in files within the "data_mingap" folder (as .npz) each contain the data itself (index by "data") and the parameter set (indexed by "params"). Results from additional simulations performed which share the same parameter set are appended to the data already contained within these files.
		
- Running multiple simulations:
	1. Open param_sets/ax_params.sh. Specify all parameters -- the organization of the parameters is similar to that of py_main.py. Pay special attention to "Varying parameters," specifically "STR_VARY_ONE" and "STR_VARY_TWO." Two parameters can be specified to be varied. Arrays containing values to be varied override any single-specifications. For example, if STR_VARY_ONE=TAU, the specified array ARR_TAU will override the parameter TAU that is usually specified. 
	2. Run ax_run.sh by running the command "ax_run.sh ax_params.sh false" in command line. See notes within ax_run.sh for further details.

- Combining data files:
	- Main quantities (e.g. diabatic error)
		1. Open pk_combiner.py. 
		2. Make specifications after control statement (if __name__ == "__main__"). In particular, ensure that desired feature flags (e.g. averaging, data replacement) are turned on and that directories and file names are specified accordingly. 
		3. Run the script. The combined data will be saved/updated as a .csv in the folder specified in dirname_main.
	- Secondary quantities (e.g. minimum gap statistics)
		0. If minimum gap data needs to be combined, use ax_mingap_datahandler.py first and save the data to a "managed" folder (which will be specified through dir_ManagedData later on). 
		1. Open ax_mingap_averager.py.
		2. Make specifications. In particular, ensure that desired feature flags (e.g. data replacement) are turned on and that directories and file names are specified accordingly. Note that the script will search all files in the folder specified in dir_ManagedData and collect files which have the same naming convention conforming to the same protocol type (tuning, disorder and noise).  
		3. Run the script. The averaged data will be saved as a .csv in the folder specified in dir_averagedData.
      
Available data:
- All records of data are contained in the "records" folder as Excel files. These records keep track of all data that is contained in the "data_main" folder for the diabatic error and the "data_mingap" folder for the minimum gap statistics. 
- The data_mingap/managed folder, which contains raw, managed minimum gap data (before averaging) appears as a .rar archive. Accessing this data requires unpacking the contents of this archive. 
- The plotting scripts for the diabatic error directly pull data contained in the "data_main" folder. The plotting scripts for the minimum gap statistics pull data contained in the "data_mingap" folder. 
		
