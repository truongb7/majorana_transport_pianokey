#!/bin/bash

# ----- #
# Notes #
# ----- #

# Running this file requires the following command:
# bash ax_combiner.sh

# ---------- #
# Parameters #
# ---------- #

# Flags
FLAG_CLUSTER=false # Enable when combining individual files on cluster
FLAG_AVERAGE=false # Set to true for noise/disorder
FLAG_TIME=false # Append the date/time on outputted main file
FLAG_DELETE=false # After combining all secondary files, delete them
FLAG_OTHER=false # Combine other data corresponding to full simulations

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
STR_TUNING_CHOICE=SMOOTH

# Disorder
# -> NONE: No disorder
# -> UCUN: Uncorrelated, uniform
# -> (WIP) UCUNA: Variant of UCUN, loads a disorder configuration and applies it to every simulation within the set
# -> (WIP) UCUNB: Variant of UCUN, generates a disorder configuration once and applies it to every simulation within the set 
# -> UCND: Uncorrelated, normal
# -> GCND: Gaussian-correlated, multivariate normal distribution
# -> (WIP) GCNDB: Variant of GCND, generates a disorder configuration once and applies it to every simulation within the set 
# -> ECND: Exponentially-correlated, multivariate normal distribution
# -> (WIP) ECNDB: Variant of ECND, generates a disorder configuration once and applies it to every simulation within the set 
# -> SCND: Simple-correlated, normal distribution 
# -> (WIP) SCNDB: Variant of SCND, generates a disorder configuration once and applies it to every simulation within the set 
STR_DISORDER_CHOICE=NONE

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
STR_NOISE_CHOICE=NONE

# Directories
# -> data, data_average, data_main, data_main/cluster
DIRNAME_MAIN=data_main
DIRNAME_SECONDARY=data

# String designations
# -> Central: CTL_
# -> Narval: NVL_
# -> Beluga: BLG_
# -> Graham: GRM_
# -> Cedar: CDR_
STR_DESIG_MAIN=NONE
STR_DESIG_SECONDARY=NONE

# Secondary file names (include .csv extension)
SECONDARY_FILES=("NULL.csv")

# ----------------------- #
# Load python environment #
# ----------------------- #

# Load python environment if on cluster
if [[ "$FLAG_CLUSTER" = "true" ]]; then
	module load StdEnv/2020 ## Load standard environment version
	module load python/3.8 ## Load Python version
	source ~/python38ENV/bin/activate ## Activate Python environment from directory
fi

# Execute script
python pk_combiner.py --flag_cluster $FLAG_CLUSTER --flag_average $FLAG_AVERAGE --flag_time $FLAG_TIME --flag_delete $FLAG_DELETE --flag_other $FLAG_OTHER --tuning $STR_TUNING_CHOICE --disorder $STR_DISORDER_CHOICE --noise $STR_NOISE_CHOICE --dirmain $DIRNAME_MAIN --dirsec $DIRNAME_SECONDARY --strdesig_main $STR_DESIG_MAIN --strdesig_sec $STR_DESIG_SECONDARY --list_files_sec ${SECONDARY_FILES[@]}

# Deactivate python environment if on cluster
if [[ "$FLAG_CLUSTER" = "true" ]]; then
	deactivate ## Once program completes, deactivate Python environment
fi