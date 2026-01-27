#!/bin/bash

# ----- #
# Notes #
# ----- #

# Running this file requires the following command:
# bash ax_run.sh [PARAMETER_FILE_NAME.sh] [true/false]
#
# The name of the parameter file must be selected and must be contained within the folder "param_sets"
# If the second argument is selected as true (cluster only), a Python environment is loaded

# -------------------- #
# Parameter management #
# -------------------- #

# Activate Python environment if on cluster
FLAG_MANUAL_CLUSTER=$2
if [ "$FLAG_MANUAL_CLUSTER" = "true" ]; then
	module load StdEnv/2020 ## Load standard environment version
	module load python/3.8 ## Load Python version
	source ~/python38ENV/bin/activate ## Activate Python environment from directory
fi

# Load parameters
source ./param_sets/$1

# Declare a hash table (dictionary) of varying parameters
declare -A HSH_VARPARS=([TAU]=$TAU [L]=$L [R]=$R [LP]=$LP [STIMES]=$STIMES [STIMEM]=$STIMEM [SLOPE]=$SLOPE [DR]=$DISORDER_RATIO [LC]=$LENGTH_CORR [NR]=$NOISE_RATIO [NA]=$NOISE_PSD_AMPL [WCL]=$W_CUTOFF_LOW [WCH]=$W_CUTOFF_HIGH [W]=$FREQ_SIN [PH]=$PHASE_SIN)

# Declare a hash table (dictionary) of varying parameters (array version)
declare -A HSH_VARPARS_ARR=([TAU]=${ARR_TAU[@]} [L]=${ARR_L[@]} [R]=${ARR_R[@]} [LP]=${ARR_LP[@]} [STIMES]=${ARR_STIMES[@]} [STIMEM]=${ARR_STIMEM[@]} [SLOPE]=${ARR_SLOPE[@]} [DR]=${ARR_DISORDER_RATIO[@]} [LC]=${ARR_LENGTH_CORR[@]} [NR]=${ARR_NOISE_RATIO[@]} [NA]=${ARR_NOISE_PSD_AMPL[@]} [WCL]=${ARR_W_CUTOFF_LOW[@]} [WCH]=${ARR_W_CUTOFF_HIGH[@]} [W]=${ARR_FREQ_SIN[@]} [PH]=${ARR_PHASE_SIN[@]})

# Define arrays for varying parameters labeled one and two
ARR_VARY_ONE=(${HSH_VARPARS_ARR[$STR_VARY_ONE]})
if [ "$STR_VARY_ONE" = "NONE" ]; then
	ARR_VARY_ONE=(0)
fi
ARR_VARY_TWO=(${HSH_VARPARS_ARR[$STR_VARY_TWO]})
if [ "$STR_VARY_TWO" = "NONE" ]; then
	ARR_VARY_TWO=(0)
fi

# Specify the parameter for the number of job arrays
if [ $((${CLU_JOBARRAY_MAX}-${CLU_JOBARRAY_MIN})) -eq 0 ]; then
	CLU_JOBARRAY=1
else
	CLU_JOBARRAY=${CLU_JOBARRAY_MIN}-${CLU_JOBARRAY_MAX}
fi

# Total number of simulations 
SIZE_ARR_ONE=${#ARR_VARY_ONE[@]}
SIZE_ARR_TWO=${#ARR_VARY_TWO[@]}
SIZE_TOTAL=$((${SIZE_ARR_ONE}*${SIZE_ARR_TWO}))

# Automatically estimate the run time of all TAU's
# -> Do this by establishing an array of TAUs and an array of chain sizes
if [ "$FLAG_RUNTIME" = "true" ]; then

	# Establish TAU arrays
	if [ "$STR_VARY_ONE" = "TAU" ] || [ "$STR_VARY_TWO" = "TAU" ]; then
		ARR_TAU_RT=(${ARR_TAU[@]})
	else
		ARR_TAU_RT=($TAU)
	fi	
	
	# Establish system size arrays	
	# -> For L
	if [ "$STR_VARY_ONE" = "L" ] || [ "$STR_VARY_TWO" = "L" ]; then
		ARR_L_RT=(${ARR_L[@]})
	else
		ARR_L_RT=($L)
	fi	
	# -> For R
	if [ "$STR_VARY_ONE" = "R" ] || [ "$STR_VARY_TWO" = "R" ]; then
		ARR_R_RT=(${ARR_R[@]})
	else
		ARR_R_RT=($R)
	fi	
	# -> For LP
	if [ "$STR_VARY_ONE" = "LP" ] || [ "$STR_VARY_TWO" = "LP" ]; then
		ARR_LP_RT=(${ARR_LP[@]})
	else
		ARR_LP_RT=(${LP})
	fi	
	# -> Note that the ordering of the following matters!
	# -> If we demand that R=LP, set ARR_R_RT = ARR_LP_RT
	if [ "$FLAG_SET_Rlp" = "true" ]; then
		ARR_R_RT=(${ARR_LP_RT[@]})
	fi 
	# -> If we demand that L=R, set ARR_L_RT = ARR_R_RT
	if [ "$FLAG_SET_LR" = "true" ]; then
		ARR_L_RT=(${ARR_R_RT[@]})
	fi
	# -> Determine system sizes
	ARR_N_RT=()
	IND_COUNT=0
	for L_RT in ${ARR_L_RT[@]}
	do
		for R_RT in ${ARR_R_RT[@]}
		do 	
			ARR_N_RT[IND_COUNT]=$(($L_RT+$R_RT))
			IND_COUNT=$(($IND_COUNT+1))
		done
	done
	# -> Remove duplicate system sizes in ARR_N_RT
	ARR_N_RT=($(echo "${ARR_N_RT[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
			
	# Array of estimated cluster times
	# -> Note that the ordering of ARR_CLU_TIME follows the for loop order of ARR_N_RT followed by ARR_TAU_RT
	ARR_CLU_TIME=($(python ax_runtime.py --machine $STR_MACHINE_CHOICE --tuning $STR_TUNING_FUNCTION --disorder $STR_DISORDER_CHOICE --noise $STR_NOISE_CHOICE --Nchain ${ARR_N_RT[@]} --tau ${ARR_TAU_RT[@]} --dt $DT --Nsim $NBR_REALIZATIONS --Nthresh $NTHRESH | tr -d [],\'))

	# Make a hash table corresponding each N/TAU value to the values within ARR_CLU_TIME
	# -> Keys will be in the format of N_TAU
	# -> Important that the for loop order of N, TAU matches that of ax_runtime
	declare -A HSH_CLU_TIME
	IND_COUNT=0
	for N_RT in ${ARR_N_RT[@]}
	do
		for TAU_RT in ${ARR_TAU_RT[@]}
		do 	
			HSH_CLU_TIME[${N_RT}_${TAU_RT}]=${ARR_CLU_TIME[$IND_COUNT]}
			IND_COUNT=$(($IND_COUNT+1))
		done
	done
fi

##for key in "${!HSH_CLU_TIME[@]}"; do echo "$key"; done
##for value in "${HSH_CLU_TIME[@]}"; do echo "$value"; done

# ------------------ #
# Deploy simulations #
# ------------------ #

COUNT_SIM=1

# Loop through parameters labeled one and two
for PAR_VARY_ONE in ${ARR_VARY_ONE[@]}
do
	for PAR_VARY_TWO in ${ARR_VARY_TWO[@]}
	do
		echo "SIM: ${COUNT_SIM}/${SIZE_TOTAL}, ${STR_VARY_ONE}: ${PAR_VARY_ONE}, ${STR_VARY_TWO}: ${PAR_VARY_TWO}"
		# Replace varied parameters in HSH_VARPARS
		if [ "$STR_VARY_ONE" != "NONE" ]; then
			HSH_VARPARS[$STR_VARY_ONE]=$PAR_VARY_ONE
		fi
		
		if [ "$STR_VARY_TWO" != "NONE" ]; then
			HSH_VARPARS[$STR_VARY_TWO]=$PAR_VARY_TWO
		fi
		
		# Reset parameters
		TAU=${HSH_VARPARS[TAU]}
		LP=${HSH_VARPARS[LP]}
		STIMES=${HSH_VARPARS[STIMES]}
		STIMEM=${HSH_VARPARS[STIMEM]}
		SLOPE=${HSH_VARPARS[SLOPE]}
		DISORDER_RATIO=${HSH_VARPARS[DR]}
		LENGTH_CORR=${HSH_VARPARS[LC]}
		NOISE_RATIO=${HSH_VARPARS[NR]}
		NOISE_PSD_AMPL=${HSH_VARPARS[NA]}
		W_CUTOFF_LOW=${HSH_VARPARS[WCL]}
		W_CUTOFF_HIGH=${HSH_VARPARS[WCH]}
		FREQ_SIN=${HSH_VARPARS[W]}
		PHASE_SIN=${HSH_VARPARS[PH]}
		
		# Special cases
		if [ "$FLAG_SET_Rlp" = "true" ]; then
			R=${HSH_VARPARS[LP]}
		else
			R=${HSH_VARPARS[R]}
		fi
		
		if [ "$FLAG_SET_LR" = "true" ]; then
			L=$R
		else
			L=${HSH_VARPARS[L]}
		fi
		
		# Automatically estimate the runtime based on TAU
		# -> This must be activated after all parameters have been redefined
		if [ "$FLAG_RUNTIME" = "true" ]; then
			N=$(($L+$R))
			CLU_TIME=${HSH_CLU_TIME[${N}_${TAU}]}
			##echo $N $TAU $CLU_TIME
		fi
		
		# Run simulations on local machine
		if [ "$FLAG_MCH_CLUSTER" = "false" ]; then
			python -u pk_main.py --mc_mach $STR_MACHINE_CHOICE --op_m_clus $FLAG_CLUSTER --op_m_osim $FLAG_ONESIM --op_m_avg $FLAG_AVERAGE --op_f_diab $FLAG_FULLSIM_DIABERR --op_f_diab_cc $FLAG_FULLSIM_DIABERR_CC --op_f_diab_ct $FLAG_FULLSIM_DIABERR_CT --op_f_trpr $FLAG_FULLSIM_TRANSPROB --op_f_trpr_cc $FLAG_FULLSIM_TRANSPROB_CC --op_f_min $FLAG_FULLSIM_MINGAP --op_f_vel $FLAG_FULLSIM_VELOCITY --op_f_minsep $FLAG_FULLSIM_SEP_MINGAP --op_s_diab $FLAG_ONESIM_DIABERR --op_s_diab_cc $FLAG_ONESIM_DIABERR_CC --op_s_diab_ct $FLAG_ONESIM_DIABERR_CT --op_s_inen $FLAG_ONESIM_INSTENERGY --op_s_inen_c $FLAG_ONESIM_INSTENERGY_C --op_s_exen $FLAG_ONESIM_EXPTENERGY --op_s_occp $FLAG_ONESIM_OCCUPATION --op_s_mate $FLAG_ONESIM_MATELEMENT --op_s_trpr $FLAG_ONESIM_TRANSPROBS --op_s_trpr_cc $FLAG_ONESIM_TRANSPROBS_CC --op_s_inev $FLAG_ONESIM_INSTEIGVEC --op_s_evev $FLAG_ONESIM_EVOLEIGVEC --op_o_nfix $FLAG_NFIX --op_o_tord $FLAG_TIMEOVERRIDE --op_o_svd $FLAG_SVD --op_o_svde $FLAG_SVD_END --pr_tune $STR_TUNING_FUNCTION --pr_disr $STR_DISORDER_CHOICE --pr_nois $STR_NOISE_CHOICE --pa_L $L --pa_R $R --pa_delt $DELTA --pa_w $W --pa_mulf $MULEFT --pa_murs $MURIGHTSTART --pa_mure $MURIGHTEND --pa_lp $LP --pa_nstp $N_STEPS --pa_tau $TAU --pa_dt $DT --pa_sti $STIMES --pa_stm $STIMEM --pa_spm $SLOPE --pa_scl $SCALE_FACTOR --pa_nthr $NTHRESH --pa_nfix $NFIX  --pa_nbrl $NBR_REALIZATIONS --pd_r $DISORDER_RATIO --pd_xi $LENGTH_CORR --pn_r $NOISE_RATIO --pn_A $NOISE_PSD_AMPL --pn_wcl $W_CUTOFF_LOW --pn_wch $W_CUTOFF_HIGH --pn_tau $TAU_NOISE --pn_dt $DT_NOISE --pn_w_sin $FREQ_SIN --pn_p_sin $PHASE_SIN --cq_f_trpr ${LIST_IND_TRANSPROB[@]} --cq_f_trpr_cc ${LIST_IND_TRANSPROB_CC[@]} --cq_s_inen ${LIST_IND_INSTENERG[@]} --cq_s_inen_c ${LIST_IND_INSTENERG_C[@]} --cq_s_exen ${LIST_IND_EXPTENERG[@]} --cq_s_occp ${LIST_IND_OCCUPATION[@]} --cq_s_trpr ${LIST_IND_TRANSPROB[@]}  --cq_s_trpr_cc ${LIST_IND_TRANSPROB_CC[@]} --cq_s_inev ${LIST_IND_INSTEIGVEC[@]} --cq_s_evev ${LIST_IND_EVOLEIGVEC[@]} --po_nsamp $NSAMP --po_desig $STR_DESIG --po_desig2 $STR_DESIG2 --dr_main $DIRNAME_DATA --dr_avg $DIRNAME_DATA_AVERAGE --dr_one $DIRNAME_DATA_ONESIM --dr_oavg $DIRNAME_DATA_ONESIM_AVERAGE --dr_min $DIRNAME_DATA_MINGAP
		
		# Run simulations on cluster
		else
			sbatch --job-name=$CLU_JOBNAME --array=$CLU_JOBARRAY --time=$CLU_TIME --mem=$CLU_MEM --export=FILE_PARAM=$1,STR_VARY_ONE=$STR_VARY_ONE,PAR_VARY_ONE=$PAR_VARY_ONE,STR_VARY_TWO=$STR_VARY_TWO,PAR_VARY_TWO=$PAR_VARY_TWO ax_cluster.sh
		fi
		
		COUNT_SIM=$((${COUNT_SIM}+1))
	done
done

# Deactivate Python environment if on cluster
if [ "$FLAG_MCH_CLUSTER" = "true" ]; then
	deactivate
fi