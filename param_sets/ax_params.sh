#!/bin/bash

# ---------------------- #
# Machine specifications #
# ---------------------- #

# Designations/flag for local machine or cluster
# -> LCL: Personal PC
# -> LAP: Personal Laptop
# -> NVL: Narval
# -> BLG: Beluga
# -> GRM: Graham
# -> CDR: Cedar
STR_MACHINE_CHOICE=LCL
FLAG_MCH_CLUSTER=false # (Important that this is all lowercase)
FLAG_RUNTIME=false # Automatically calculate the simulation run time based on the value of TAU

# Cluster parameters
CLU_JOBNAME=NAME # Job name
CLU_JOBARRAY_MIN=1 # First index for job arrays
CLU_JOBARRAY_MAX=1 # Last index for job arrays
CLU_TIME=05:00:00 # Total allocated time (hh:mm:ss)
CLU_MEM=1G # Memory (megabytes = M, gigabytes = G, etc)

# ------------------------ #
# Operation specifications #
# ------------------------ #

# Main features
FLAG_CLUSTER=false # Cluster mode: all data files will have parameter string appended. By default, takes the same value as FLAG_MCH_CLUSTER
FLAG_AVERAGE=false # Perform averaging of data sets when applicable
FLAG_ONESIM=false # Calculates the diabatic error vs. time in a single simulation (note: this will break the loop after one iteration -- only the first tau value in the array will be used)

# Full simulation features
FLAG_FULLSIM_DIABERR=true # Diabatic error, uses initial and compared states of total Hamiltonian (TT)
FLAG_FULLSIM_DIABERR_CC=false # Diabatic error, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)
FLAG_FULLSIM_DIABERR_CT=false # Diabatic error, uses initial state of clean Hamiltonian and compared stated of instantaneous total Hamiltonian (CT)

FLAG_FULLSIM_TRANSPROB=false # Transition probability, uses initial and compared states of total Hamiltonian (TT)
FLAG_FULLSIM_TRANSPROB_CC=false # Transition probability, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)

FLAG_FULLSIM_MINGAP=false # Calculate the minimum gap of a simulation
FLAG_FULLSIM_VELOCITY=false # Calculate the effective, normalized Landau-Zener velocity

# Full simulation features, separate files
FLAG_FULLSIM_SEP_MINGAP=true # Calculate the minimum gap of a simulation and save to corresponding .npy file. Subsequent simulations performed with the same parameters will *append* minimum gaps to this file as opposed to replacing them

# Single simulation features
FLAG_ONESIM_DIABERR=false # Diabatic error vs. t, uses initial and compared states of total Hamiltonian (TT)
FLAG_ONESIM_DIABERR_CC=false # Diabatic error vs. t, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)
FLAG_ONESIM_DIABERR_CT=false # Diabatic error vs. t, uses initial state of clean Hamiltonian and compared stated of instantaneous total Hamiltonian (CT)

FLAG_ONESIM_EXPTENERGY=false # Expectation value of total Hamiltonian vs. t
FLAG_ONESIM_OCCUPATION=false # Occupation numbers vs. t
FLAG_ONESIM_MATELEMENT=false # Matrix elements vs. t of noise in clean inst. eigenbasis

FLAG_ONESIM_TRANSPROBS=false # Transition probabilities vs. t, uses initial and compared states of total Hamiltonian (TT)
FLAG_ONESIM_TRANSPROBS_CC=false # Transition probabilities vs. t, uses initial state of clean Hamiltonian and compared stated of instantaneous clean Hamiltonian (CC)

# (WIP, these will need to be saved to separate files)
FLAG_ONESIM_INSTENERGY=false # Instantaneous energy vs. t of total Hamiltonian
FLAG_ONESIM_INSTENERGY_C=false # Instantaneous energy vs. t of clean Hamiltonian

# (WIP, these will need to be saved to separate files)
FLAG_ONESIM_INSTEIGVEC=false # Instantaneous eigenvectors vs. t
FLAG_ONESIM_EVOLEIGVEC=false # Time-evolved eigenvectors vs. t

# Other
FLAG_NFIX=false # Fix the number of time steps
FLAG_TIMEOVERRIDE=false # The total times tau_init and tau_final are no longer relative to tau_rel
FLAG_SVD=false # Enable SVD method in calculation of unitary during the calculation
FLAG_SVD_END=true # Enable the SVD method at the end of the calculation

# ----------------------- #
# Protocol specifications #
# ----------------------- #

# Tuning function
# -> LIN: Linear
# -> SMOOTH: Smooth (sin^2)
# -> FQD: Fast-QUAD (adapted from Felix's/Bill C.'s paper)
# -> SHARP: Sharp. At critical point, the energy changes sharply as ~|t-t*|
# -> TRI: Tri-region. Custom tuning function which interpolates (cubic-ly) between regions
# -> TRIB: Tri-region, B-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a cubic function to interpolate between the regions
# -> TRIC: Tri-region, C-variant. Custom tuning function which is determined directly from features of the exact spectrum; Uses a Gaussian kernel to smoothen tuning function
# -> TRID: Tri-region, D-variant. Custom tuning function. Uses a Gaussian kernel to smoothen tuning function
# -> TRIE: Tri-region, E-variant. Same as D-variant, but introduces a scale factor f = slope_mid*s_time_m. s_time_m is eliminated as a parameter in favour of this scale factor
STR_TUNING_FUNCTION=SMOOTH

# Disorder
# -> NONE: No disorder
# -> UCUN: Uncorrelated, uniformly distributed
# -> UCND: Uncorrelated, normally/Gaussian distributed
# -> GCUN: Gaussian correlated, uniformly distributed
# -> GCND: Gaussian correlated, normally/Gaussian distributed
# -> ECND: Exponentially correlated, normally/Gaussian distributed
# -> SCND: Simply correlated, normally/Gaussian distributed
STR_DISORDER_CHOICE=UCND

# Noise
# -> NONE: Default, no noise
# -> WHT: White noise
# -> WHTC: White noise, with cutoff frequencies
# -> WHTCB: White noise, with cutoff frequencies, psd amplitude variant
# -> 1F1: 1/f noise, with cutoff frequencies
# -> 1F1B: 1/f noise, replace the disorder strength with the amplitude of the power spectral density as a parameter
# -> SIN: Sine wave with a specified amplitude, phase, and frequency, meant to represent a single frequency mode in a noise signal
# -> SINAV: Single mode noise; same as SIN but results are phase averaged "on the fly"
# ---> Note: SINAV perform averaging in the same way as for typical noise and disorder
STR_NOISE_CHOICE=NONE

# Varying parameters (only two at a time)
# General
# -> NONE
# -> TAU: Total time  
# -> R: Right side of chain
# -> LP: Piano key size
# Tuning function
# -> STIMES: Duration of start/end regions
# -> STIMEM: Duration of middle region
# -> SLOPE: Slope of middle region
# Disorder
# -> DR: Disorder ratio
# -> LC: Correlation length
# Noise
# -> NR: Noise ratio
# -> NA: Noise PSD amplitude
# -> WCL: Low frequency cutoff 
# -> WCH: High frequency cutoff
# Single mode noise
# -> W: Single mode frequency
# -> PH: Single mode phase
STR_VARY_ONE=TAU
STR_VARY_TWO=DR

# Flags
FLAG_SET_Rlp=false # For fixing R=lp; use if R or lp are varied
FLAG_SET_LR=false # For fixing L=R; use if R, lp, or L are varied

# ------------------ #
# Varying parameters #
# ------------------ #

# General
ARR_TAU=(20.0 40.0 60.0)
#ARR_TAU=($(seq 1.0 1.0 30.0))
ARR_L=(10 20 30)
ARR_R=(10 20 30)
ARR_LP=(10 20 30)

# Tuning function
ARR_STIMES=(0)
ARR_STIMEM=(0)
ARR_SLOPE=(0)

# Disorder
ARR_DISORDER_RATIO=(0.1 0.2 0.3)
ARR_LENGTH_CORR=(1.0 2.0 3.0)

# Noise
ARR_NOISE_RATIO=(0.01 0.05 0.1)
ARR_NOISE_PSD_AMPL=(1e-5 1e-4 1e-3)
ARR_W_CUTOFF_LOW=0
ARR_W_CUTOFF_HIGH=($(seq 1.0 0.5 5.0))
#ARR_W_CUTOFF_HIGH=($(python ax_geomspace.py 0.05 6.3 50 | tr -d '[],'))
#ARR_W_CUTOFF_HIGH=($(python ax_logspace.py -3 2 50 | tr -d '[],'))

# Single mode noise
ARR_FREQ_SIN=0
#ARR_FREQ_SIN=($(seq 0.005 0.005 5.0))
#ARR_FREQ_SIN=($(python ax_logspace.py -3 2 50 | tr -d '[],'))
#ARR_PHASE_SIN=0
ARR_PHASE_SIN=($(seq 0.0 0.06 6.3))

# -> Useful functions
# LINEAR=($(python ax_linspace.py 1.0 10.0 50 | tr -d '[],'))
# GEOMSPACE=($(python ax_geomspace.py 0.05 6.3 50 | tr -d '[],'))
# LOGSPACE=($(python ax_logspace.py -3 2 50 | tr -d '[],'))

# ------------------- #
# Protocol parameters #
# ------------------- #

# System
L=30
R=30
DELTA=0.6
W=3.0
MULEFT=2.8
MURIGHTSTART=3.2
MURIGHTEND=2.8

# Piano keys
LP=$R # Total size of piano key (default = R)
N_STEPS=1 # Number of piano key steps; note that lp/(n_steps) must be an integer

# Time
TAU=1520.0 # Protocol time
DT=0.1 # Maximum time step size
NTHRESH=1500 # Minimum number of time steps
NFIX=1500 # (if FLAG_TIMEOVERRIDE == true) Fixed number of time steps

# Tuning function
STIMES=0.05
STIMEM=0.25
SLOPE=0.0
SCALE_FACTOR=1.0

# Averaging
NBR_REALIZATIONS=5 # Number of realizations (for cluster, in each job)

# Disorder
DISORDER_RATIO=0.3 # Disorder strength (default: in units of the chemical potential difference)
LENGTH_CORR=3.0 # (GCND, SCND) Correlation length, (ECND) decay length

# Noise, universal
NOISE_RATIO=0.01 # Noise strength in units of the chemical potential difference
NOISE_PSD_AMPL=1e-4 # Amplitude of the power spectral density (note: A = 1e-7 corresponds roughly to a variance of 1e-5)
W_CUTOFF_LOW=1e-8 # Low frequency cutoff
W_CUTOFF_HIGH=1.0 # High frequency cutoff
TAU_NOISE=40000 # Total time of signal 
DT_NOISE=$DT # Size of noise time step (default = dt)

# Noise, single modes
FREQ_SIN=0.03 # Frequency
PHASE_SIN=0.0 # Phase

# ----------------------------------- #
# Calculation quantity specifications #
# ----------------------------------- #

LIST_IND_INSTENERG=(0 1) # Instantaneous energies of total Hamiltonian
LIST_IND_INSTENERG_C=(0 1) # Instantaneous energies of clean Hamiltonian

LIST_IND_EXPTENERG=(0 1) # Expectation values of H(t) to calculate
LIST_IND_OCCUPATION=(0 1) # Occupation numbers to calculate
LIST_IND_TRANSPROB=([1] [1,2,3]) # Transition probabilities to calculate, uses initial and compared states of total Hamiltonian (TT)
LIST_IND_TRANSPROB_CC=([1] [1,2]) # Transition probabilities to calculate, uses initial and compared states of clean Hamiltonian (CC)

LIST_IND_INSTEIGVEC=(0 1) # Instantaneous eigenvectors to calculate
LIST_IND_EVOLEIGVEC=(0 1) # Time-evolved eigenvectors to calculate

# ---------------- #
# Other parameters #
# ---------------- # 

# For simulations 
# -> _X is the usual format
NSAMP=500 # (if FLAG_ONESIM == true) Number of points to be sampled and saved to file
STR_DESIG=NONE # Name of any special designator attached to the beginning of data files (i.e. the cluster task ID of the job) 
STR_DESIG2=NONE # Name of any second special designator

# For combiner 
# -> X_ is the usual format for main files
# -> For example: LCL_ (Local), NVL_ (Narval), BLG_ (Beluga), CDR_ (Cedar), GRM_ (Graham)
# -> Cluster mode: _X is the usual format
STR_DESIG_MAIN=NONE
STR_DESIG_SECONDARY=NONE

# -------------------------- #
# Directories and file names #
# -------------------------- #

# For simulations
DIRNAME_DATA=data
DIRNAME_DATA_AVERAGE=data_average
DIRNAME_DATA_ONESIM=data_onesim
DIRNAME_DATA_ONESIM_AVERAGE=data_onesim_average
DIRNAME_DATA_MINGAP=data_mingap

# For combiner
DIRNAME_MAIN=data_main
DIRNAME_SECONDARY=data_average