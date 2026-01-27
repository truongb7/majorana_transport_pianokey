"""
Kitaev chain: Piano key simulations
Simulation

Features:
    - Runs specified time-dependent protocol
    - Central feature is that the full time evolution operator is calculated. This is used to calculated quantities of interest
    - Returns full simulation results (quantities calculated at the end of protocol) and single simulation results (quantities calculated during protocol)
"""

# ------- #
# Modules #
# ------- #

#import matplotlib.pyplot as pl
import numpy as np
import pk_functions as pk
import scipy.fft as sc_fft
import scipy.optimize as sc_optim
#import scipy.linalg as sc_lin

# -------------- #
# Run simulation #
# -------------- #

def run_simulation(dict_operations_main, dict_operations_other, dict_operations_fullsim, dict_operations_onesim, dict_specifications, dict_protocol, dict_disorder, dict_noise, dict_fullsim, dict_onesim, dict_other):
    
    # ----------------------------------- #
    # Unpack parameters from dictionaries #
    # ----------------------------------- #
    
    # Operations, full simulations
    flag_fullsim_diabErr = dict_operations_fullsim["diabErr"]
    flag_fullsim_diabErr_cc = dict_operations_fullsim["diabErr_cc"]
    flag_fullsim_diabErr_ct = dict_operations_fullsim["diabErr_ct"] 
    flag_fullsim_transProb = dict_operations_fullsim["transProb"]
    flag_fullsim_transProb_cc = dict_operations_fullsim["transProb_cc"]
    flag_fullsim_mingap = dict_operations_fullsim["mingap"]
    flag_fullsim_velocity = dict_operations_fullsim["velocity"]
    flag_fullsim_sep_mingap = dict_operations_fullsim["mingap_sep"]
    
    # Operations, other
    flag_timeoverride = dict_operations_other["timeoverride"]
    flag_SVD = dict_operations_other["SVD"]
    flag_SVD_END = dict_operations_other["SVD_END"]
    
    # Specifications
    str_tuning_choice = dict_specifications["tuning"]
    str_disorder_choice = dict_specifications["disorder"]
    str_noise_choice = dict_specifications["noise"]
    
    # Protocol
    L = dict_protocol["L"]
    R = dict_protocol["R"]
    Delta = dict_protocol["Delta"]
    w = dict_protocol["w"]
    muLeft = dict_protocol["muLeft"]
    muRightStart = dict_protocol["muRightStart"]
    muRightEnd = dict_protocol["muRightEnd"]
    lp = dict_protocol["lp"]
    n_steps = dict_protocol["n_steps"]
    tau = dict_protocol["tau"]
    dt = dict_protocol["dt"]
    s_time_init = dict_protocol["s_time_init"]
    s_time_mid = dict_protocol["s_time_mid"]
    slope_mid = dict_protocol["slope_mid"]
    scale_factor = dict_protocol["scale_factor"]
    Nthresh = dict_protocol["Nthresh"]
    
    # Disorder
    disorder_ratio = dict_disorder["disorder_ratio"]
    length_corr = dict_disorder["length_corr"]

    # Noise
    noise_ratio = dict_noise["noise_ratio"]
    noise_psd_ampl = dict_noise["noise_psd_ampl"]
    w_cutoff_low = dict_noise["w_cutoff_low"]
    w_cutoff_high = dict_noise["w_cutoff_high"]
    tau_noise = dict_noise["tau_noise"]
    freq_sin = dict_noise["freq_sin"]
    phase_sin = dict_noise["phase_sin"]
    
    # Other
    NSamp = dict_other["NSamp"]
    
    # ----------------------- #
    # Initial flag management #
    # ----------------------- #
    
    # In the case of minimum gap and velocity calculations, the instantaneous spectrum is required
    if True in [flag_fullsim_mingap, flag_fullsim_velocity, flag_fullsim_sep_mingap]:
        dict_operations_main["onesim"] = True
        dict_operations_onesim["instEnergy"] = True
        
    # --------- #
    # Variables #
    # --------- #
    
    # Disorder 
    muD_max = np.round(disorder_ratio*np.abs(muRightStart - muRightEnd), 8) # Maximum deviation of disordered chemical potential using disorder_ratio; in normal distributions, this is treated as the standard deviation
    
    # Noise
    muN_max = np.round(noise_ratio*np.abs(muRightStart - muRightEnd), 8) # Maximum deviation of noisy chemical potential using noise_ratio
    
    # Characteristic Landau-Zener time
    #tau_LZ = pk.tau_LZ(w, muRightEnd, lp, Delta) # Characteristic time as predicted by LZ

    # Total protocol time in actual energy units
    if flag_timeoverride == True:
        tau_fin = tau
    else:
        tau_fin = tau
        #tau_fin = tau*tau_LZ
        
    # Piano key 
    r = int(lp/n_steps) # Size of each piano key
    tau_n = tau_fin/n_steps # Time for each piano key
        
    # Adjustment of time step size and number of time steps     
    # -> Adjust dt if tau_fin/dt is less than Nthresh 
    if tau_fin/dt < Nthresh:
        dt = tau_fin/Nthresh
    # -> Ensure that dt*N_sim = tau_fin by adjusting dt and N_sim simulataneously
    N_sim_intermediate = tau_fin/dt
    N_sim = int(N_sim_intermediate)
    dt = 1.0*tau_fin/N_sim
    
    # Use the above to calculate N_sim_n, the number of time steps for each piano key
    # -> Calculate N_sim_n 
    N_sim_n = int(N_sim/n_steps)
    # -> Adjust dt so that N_sim_n*dt = tau_n
    dt = 1.0*tau_n/N_sim_n
    # -> Adjust N_sim
    N_sim = n_steps*(N_sim_n+1)
    
    # ------------------------------------------ #
    # Dictionaries for single simulation results #
    # ------------------------------------------ #

    # Initialize dictionary for single simulation results
    dict_results_onesim = {}
    dict_results_onesim_other = {"onesim_spec":{}, "onesim_spec_clean":{}, "onesim_eigvecs":{}, "onesim_evolvecs":{}}
    
    # ----------------- #
    # Generate disorder #
    # ----------------- #
    
    # Note that for random numbers picked from Gaussian distributions (XXND), the standard deviation is taken to be 1/3 of muN_max
    if str_disorder_choice == "UCUN":
        arr_muD = np.random.uniform(-muD_max, muD_max, L+R)
    if str_disorder_choice == "UCND": 
        arr_muD = np.random.normal(loc=0.0, scale=muD_max, size=L+R)
    if str_disorder_choice == "GCUN":
        arr_muD = pk.generate_muD_GCUN(L+R, np.zeros(L+R), 2.0*muD_max, length_corr)
    if str_disorder_choice == "GCND":
        arr_muD = pk.generate_muD_GCND(L+R, np.zeros(L+R), np.full(L+R, muD_max), length_corr)
    if str_disorder_choice == "ECND":
        arr_muD = pk.generate_muD_ECND(L+R, np.zeros(L+R), np.full(L+R, muD_max), length_corr)
    if str_disorder_choice == "SCND":
        arr_muD = pk.generate_muD_SCND(L+R, 0.0, muD_max, length_corr)
    if str_disorder_choice == "NONE":
        arr_muD = np.zeros(L+R)
        
    # -------------- #
    # Generate noise #
    # -------------- #
    
    # White 
    # -> No cutoff frequencies, set noise strength by standard deviation noise_maxampl/3.0
    # -> Note that 3*sigma covers mostly the full width of the distribution
    if str_noise_choice == "WHT":
        arr_muN = np.random.normal(loc=0.0, scale=muN_max/3.0, size=N_sim+1)
        
    # -> With cutoff frequencies, set noise strength by standard deviation noise_maxampl/3.0
    if str_noise_choice == "WHTC":
        
        # Generate number of points, N_noise, from dt_noise and tau_noise
        # -> Adjust so that N_noise is even and dt_noise remains fixed
        N_noise = int(tau_noise/dt)
        if N_noise%2 != 0:
            N_noise += 1
        tau_noise = dt*N_noise
        
        # Determine appropriate PSD amplitude
        arr_w = 2*np.pi*sc_fft.fftfreq(N_noise, d=dt)
        noise_psd_ampl_eff = pk.varToAmp_whitenoise((muN_max/3.0)**2, w_cutoff_low, w_cutoff_high, arr_w, tau_noise)
        
        # Generate and extract
        arr_noise = pk.gen_whitenoise(noise_psd_ampl_eff, w_cutoff_low, w_cutoff_high, dt, N_noise)
        arr_muN = arr_noise[:N_sim+1]
    
    # -> With cutoff frequencies, set noise strength by PSD amplitude (noise_psd_ampl)
    if str_noise_choice == "WHTCB":
        
        # Generate number of points, N_noise, from dt_noise and tau_noise
        # -> Adjust so that N_noise is even and dt_noise remains fixed
        N_noise = int(tau_noise/dt)
        if N_noise%2 != 0:
            N_noise += 1
        tau_noise = dt*N_noise
        
        # Generate and extract
        arr_noise = pk.gen_whitenoise(noise_psd_ampl, w_cutoff_low, w_cutoff_high, dt, N_noise)
        arr_muN = arr_noise[:N_sim+1]
        
    # 1/f
    # -> With cutoff frequencies, set noise strength by standard deviation noise_maxampl/3.0
    if str_noise_choice == "1F1":
        
        # Generate number of points, N_noise, from dt_noise and tau_noise
        # -> Adjust so that N_noise is even and dt_noise remains fixed
        N_noise = int(tau_noise/dt)
        if N_noise%2 != 0:
            N_noise += 1
        tau_noise = dt*N_noise
        
        # Determine appropriate PSD amplitude
        arr_w = 2*np.pi*sc_fft.fftfreq(N_noise, d=dt)
        noise_psd_ampl_eff = pk.varToAmp_1f1noise((muN_max/3.0)**2, w_cutoff_low, w_cutoff_high, arr_w, tau_noise)
        
        # Generate and extract
        arr_noise = pk.gen_1f1noise(noise_psd_ampl_eff, w_cutoff_low, w_cutoff_high, dt, N_noise)
        arr_muN = arr_noise[:N_sim+1]
    
    # -> With cutoff frequencies, set noise strength by PSD amplitude (noise_psd_ampl)
    if str_noise_choice == "1F1B":
        
        # Generate number of points, N_noise, from dt_noise and tau_noise
        # -> Adjust so that N_noise is even and dt_noise remains fixed
        N_noise = int(tau_noise/dt)
        if N_noise%2 != 0:
            N_noise += 1
        tau_noise = dt*N_noise
        
        # Generate and extract
        arr_noise = pk.gen_1f1noise(noise_psd_ampl, w_cutoff_low, w_cutoff_high, dt, N_noise)
        arr_muN = arr_noise[:N_sim+1]
        
    # Single mode
    if str_noise_choice in ["SIN", "SINAV"]:
        arr_t_ns = np.linspace(0, tau_fin, N_sim+1)
        arr_muN = muN_max*np.sin(freq_sin*arr_t_ns + phase_sin)
    
    # None
    if str_noise_choice == "NONE":
        arr_muN = np.zeros(N_sim+1)
        
    # ----------------------------- #
    # Initial and final Hamiltonian #
    # ----------------------------- # 
    
    # Initial values of Delta and w
    arr_Delta_init = np.full(L+R, Delta)
    arr_w_init = np.full(L+R, w)
    
    # Final values of Delta and w
    arr_Delta_end = arr_Delta_init
    arr_w_end = arr_w_init

    # Initial values of chemical potential
    arr_muLeft_init = np.full(L, muLeft)
    arr_muRight_init = np.full(R, muRightStart)
    arr_mu_init = np.concatenate((arr_muLeft_init, arr_muRight_init))

    # Final values of chemical potential
    arr_muLeft_end = arr_muLeft_init
    arr_muRight_end = np.full(R, muRightStart)
    arr_muRight_end[:lp] = muRightEnd
    arr_mu_end = np.concatenate((arr_muLeft_end, arr_muRight_end))
    
    # ------------------ #
    # Clean Hamiltonians #
    # ------------------ #
    
    # Initial and final Hamiltonians in Majorana basis
    matH_init_clean = pk.ham_kit(L+R, arr_mu_init, arr_w_init, arr_Delta_init)
    matH_end_clean = pk.ham_kit(L+R, arr_mu_end, arr_w_end, arr_Delta_end)
    
    # Extract the real, anti-symmetric matrix A defined through H = iA
    matA_init_clean = (-1j*matH_init_clean).real
    matA_end_clean = (-1j*matH_end_clean).real
    
    # Create covariance matrices for the many-body ground states (even and odd)
    # -> Orthogonal transformations
    matO_init_clean = pk.calc_matO(matA_init_clean)
    matO_end_clean = pk.calc_matO(matA_end_clean)
    # -> Covariance matrice in the eigenbases
    matMO_clean = pk.calc_matMO(L, R, [])
    # -> Covariance matrices in the original basis
    matM_init_even_clean, matM_init_odd_clean = pk.calc_matM(matMO_clean, matO_init_clean)
    matM_end_even_clean, matM_end_odd_clean = pk.calc_matM(matMO_clean, matO_end_clean)    
    
    # ------------------ #
    # Total Hamiltonians #
    # ------------------ #
    
    # With noise, total Hamiltonian = clean H + disordered/noisy H
    if str_disorder_choice != "NONE" or str_noise_choice != "NONE":
        
        # Disorder Hamiltonian
        if str_disorder_choice != "NONE":
            arr_mu_disorder = arr_muD
            matH_disorder = pk.ham_kit(L+R, arr_mu_disorder, np.zeros(L+R), np.zeros(L+R))
        else:
            matH_disorder = pk.ham_kit(L+R, np.zeros(L+R), np.zeros(L+R), np.zeros(L+R))
    
        # Noise Hamiltonian
        if str_noise_choice != "NONE":
            arr_noise_initvals = [arr_muN[ind_key*(N_sim_n+1)] for ind_key in range(0, n_steps)]
            arr_noise_endvals = [arr_muN[(ind_key+1)*(N_sim_n+1)-1] for ind_key in range(0, n_steps)]
            arr_mu_lp_noise_init = np.concatenate([np.full(r, x) for x in arr_noise_initvals])
            arr_mu_lp_noise_end = np.concatenate([np.full(r, x) for x in arr_noise_endvals])
            arr_mu_noise_init = np.concatenate([np.zeros(L), arr_mu_lp_noise_init, np.zeros(R-lp)])
            arr_mu_noise_end = np.concatenate([np.zeros(L), arr_mu_lp_noise_end, np.zeros(R-lp)])
            matH_noise_init = pk.ham_kit(L+R, arr_mu_noise_init, np.zeros(L+R), np.zeros(L+R))
            matH_noise_end = pk.ham_kit(L+R, arr_mu_noise_end, np.zeros(L+R), np.zeros(L+R))
        else:
            matH_noise_init = pk.ham_kit(L+R, np.zeros(L+R), np.zeros(L+R), np.zeros(L+R))
            matH_noise_end = pk.ham_kit(L+R, np.zeros(L+R), np.zeros(L+R), np.zeros(L+R))
        
        # Total Hamiltonians
        matH_init = matH_init_clean + matH_disorder + matH_noise_init
        matH_end = matH_end_clean + matH_disorder + matH_noise_end
        
        # Extract the real, anti-symmetric matrix A defined through H = iA
        matA_init = (-1j*matH_init).real
        matA_end = (-1j*matH_end).real
        
        # Create covariance matrices for the many-body ground states (even and odd)
        # -> Orthogonal transformations
        matO_init = pk.calc_matO(matA_init)
        matO_end = pk.calc_matO(matA_end)
        # -> Covariance matrice in the eigenbases
        matMO = pk.calc_matMO(L, R, [])
        # -> Covariance matrices in the original basis
        matM_init_even, matM_init_odd = pk.calc_matM(matMO, matO_init)
        matM_end_even, matM_end_odd = pk.calc_matM(matMO, matO_end)  
        
    # Without noise, the total and clean Hamiltonians are equal
    else:
        # Orthogonal transformations
        matO_init = matO_init_clean
        matO_end = matO_end_clean
        # Covariance matrice in the eigenbases
        matMO = matMO_clean
        # Covariance matrices
        matM_init_even, matM_init_odd = matM_init_even_clean, matM_init_odd_clean
        matM_end_even, matM_end_odd = matM_end_even_clean, matM_end_odd_clean
    
    # ----------------------- #
    # Tuning protocol (clean) #
    # ----------------------- #
    
    # Array of times for a single piano key
    arr_s = np.round(np.linspace(0, tau_n, N_sim_n+1)/tau_n, 8)
    
    # Establish the tuning of the chemical potential 
    if str_tuning_choice == "LIN":
        arr_muRight_s_clean = pk.mu_stime_lin(arr_s, muRightStart, muRightEnd)
    elif str_tuning_choice == "SMOOTH":
        arr_muRight_s_clean = pk.mu_stime(arr_s, muRightStart, muRightEnd)
    elif str_tuning_choice == "FQD":
        DeltaLZ = np.pi*Delta/(2*r)
        arr_muRight_s_clean = pk.mu_stime_fqd(arr_s, tau_n, DeltaLZ, w, muRightStart, muRightEnd)
    elif str_tuning_choice == "SHARP":
        DeltaLZ = np.pi*Delta/(2*r)
        arr_muRight_s_clean = pk.tuning_sharp(muRightStart, muRightEnd, w, DeltaLZ, arr_s*tau_n)
    elif str_tuning_choice == "TRI":
        s2_st = 0.5 - s_time_mid/2.0
        tuning_tri = pk.tuning_custom_triRegion(arr_s, s_time_init, s2_st, slope_mid)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "TRIB":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        s2_st = 0.5 - s_time_mid/2.0
        tuning_tri = pk.tuning_custom_triRegion_B(arr_s, s_time_init, s2_st, delta_mu, DeltaLZ)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "TRIC":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        tuning_tri = pk.tuning_custom_triRegion_C(arr_s, s_time_init, s_time_mid, delta_mu, DeltaLZ)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "TRID":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        tuning_tri = pk.tuning_custom_triRegion_D(arr_s, s_time_init, s_time_mid, slope_mid, delta_mu, DeltaLZ)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "TRID2":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        tuning_tri_first = pk.tuning_custom_triRegion_D(arr_s, s_time_init, s_time_mid, slope_mid, delta_mu, DeltaLZ)
        tuning_tri = pk.smoother_gaussian(arr_s, tuning_tri_first)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "TRIE":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        slope_mid_act = scale_factor/s_time_mid
        tuning_tri = pk.tuning_custom_triRegion_D(arr_s, s_time_init, s_time_mid, slope_mid_act, delta_mu, DeltaLZ)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "SNP":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        tuning_tri = pk.tuning_custom_SNP(arr_s, s_time_init, s_time_mid, delta_mu, DeltaLZ)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    elif str_tuning_choice == "SNPB":
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)        
        DeltaLZ = np.pi*Delta/(2*r)
        tuning_tri_first = pk.tuning_custom_SNP(arr_s, s_time_init, s_time_mid, delta_mu, DeltaLZ)
        tuning_tri = pk.smoother_gaussian(arr_s, tuning_tri_first)
        arr_muRight_s_clean = (1 - tuning_tri)*muRightStart + (tuning_tri)*muRightEnd
    else:
        arr_muRight_s_clean = pk.mu_stime(arr_s, muRightStart, muRightEnd)
    
    # -------------- #
    # Run simulation #
    # -------------- #
    
    # Initialize the full time evolution operator as an identity matrix
    matU = np.eye(2*(L+R), dtype='complex') # Time evolution operator
    #matU_back = np.eye(2*(L+R), dtype='complex') # Time evolution operator for a backwards protocol
    
    # Establish the initial chemical potential
    arr_mu_upd = arr_mu_init
    arr_mu_clean = np.copy(arr_mu_init)
    
    if str_disorder_choice != "NONE":
        arr_mu_upd = arr_mu_upd + arr_muD
    if str_noise_choice != "NONE":
        arr_mu_upd[L:L+lp] = arr_mu_upd[L:L+lp] + arr_mu_noise_init[L:L+lp]
    
    # Loop through piano keys sequentially
    for ind_pk in range(0, n_steps):
        
        # Establish site indices of piano key
        site_pk_start = L + ind_pk*r
        site_pk_end = L + (ind_pk+1)*r
        
        # Establish the noise on the chemical potential with time for each piano key
        arr_mu_pk_s_noise = arr_muN[ind_pk*(N_sim_n+1):(ind_pk+1)*(N_sim_n+1)]
                
        # Press the piano key
        if ind_pk == 0:
            flag_firstkey = True
        else:
            flag_firstkey = False      
            
        results_pk = pk.simulate_pianokey(L, R, arr_w_init, arr_Delta_init, arr_mu_upd, arr_mu_clean, site_pk_start, site_pk_end, ind_pk, arr_muRight_s_clean, arr_muD, arr_mu_pk_s_noise, dt, N_sim_n, NSamp, matU, matM_init_even, matM_init_even_clean, dict_operations_main, dict_operations_onesim, dict_onesim, flag_firstkey=flag_firstkey, flag_SVD=flag_SVD)
        
        # Update arr_mu to reflect that a piano key has been pressed
        arr_mu_upd[site_pk_start:site_pk_end] = arr_mu_end[site_pk_start:site_pk_end]
        arr_mu_clean[site_pk_start:site_pk_end] = arr_mu_end[site_pk_start:site_pk_end]
        if str_disorder_choice != "NONE":
            arr_mu_upd[site_pk_start:site_pk_end] = arr_mu_upd[site_pk_start:site_pk_end] + arr_muD[site_pk_start:site_pk_end]
        if str_noise_choice != "NONE":
            arr_mu_upd[site_pk_start:site_pk_end] = arr_mu_upd[site_pk_start:site_pk_end] + arr_mu_noise_end[site_pk_start:site_pk_end]
            
        # Update the time evolution operator
        matU = results_pk["matU"]
        
        # Collect single simulation results and append to dictionaries
        for key in results_pk["onesim"].keys():
            if ind_pk == 0:
                dict_results_onesim[key] = results_pk["onesim"][key]
            else:
                dict_results_onesim[key] = np.append(dict_results_onesim[key], results_pk["onesim"][key])
        
        # Collect all other single simulation results and append to dictionaries
        for key_results in ["onesim_spec", "onesim_spec_clean", "onesim_eigvecs", "onesim_evolvecs"]:
            for key in results_pk[key_results].keys():
                if ind_pk == 0:
                    dict_results_onesim_other[key_results][key] = results_pk[key_results][key]
                else:
                    dict_results_onesim_other[key_results][key] = np.append(dict_results_onesim_other[key_results][key], results_pk[key_results][key])
                    
    # ------------------------------- #
    # Full simulation data management #
    # ------------------------------- #
    
    dict_results_fullsim = {}
    
    # Use SVD method to remove numerical errors (noise) from the time evolution operator
    if flag_SVD_END == True:
        try:
            SVD_m = np.linalg.svd(matU)
            matU = SVD_m[0] @ SVD_m[2]
            #SVD_f = np.linalg.svd(U_f)
            #U_f = SVD_f[0] @ SVD_f[2]
        except:
            pass
    
    # Diabatic error (TT : init total - compare total, "default")
    if flag_fullsim_diabErr == True:
        
        # Evolve covariance matrix of initial, instantaneous ground state of total Hamiltonian
        matM_evol = matU @ matM_init_even @ matU.T.conj()
        # Square of the overlap between the final, instantaneous MB ground state and time-evolved MB ground state
        sqrmod = pk.sqrmod_overlap_covar(matM_evol, matM_end_even)
        # Calculate the diabatic error
        diabErr = np.abs(1 - np.abs(sqrmod))
        dict_results_fullsim["diabErr"] = diabErr
    
    # Diabatic error (CC : init clean - compare clean)
    if flag_fullsim_diabErr_cc == True:
        
        # Evolve covariance matrix of initial, instantaneous ground state of clean Hamiltonian
        matM_evol = matU @ matM_init_even_clean @ matU.T.conj()
        # Square of the overlap between the final, instantaneous MB clean ground state and time-evolved MB clean ground state
        sqrmod = pk.sqrmod_overlap_covar(matM_evol, matM_end_even_clean)
        # Calculate the diabatic error
        diabErr_cc = np.abs(1 - np.abs(sqrmod))
        dict_results_fullsim["diabErr_cc"] = diabErr_cc
    
    # Diabatic error (CT : init clean - compare total)
    if flag_fullsim_diabErr_ct == True:
        
        # Evolve covariance matrix of initial, instantaneous ground state of clean Hamiltonian
        matM_evol = matU @ matM_init_even_clean @ matU.T.conj()
        # Square of the overlap between the final, instantaneous MB ground state and time-evolved MB clean ground state
        sqrmod = pk.sqrmod_overlap_covar(matM_evol, matM_end_even)
        # Calculate the diabatic error
        diabErr_ct = np.abs(1 - np.abs(sqrmod))
        dict_results_fullsim["diabErr_ct"] = diabErr_ct
        
    # Transition probabilities (TT : init total - compare total, "default")
    if flag_fullsim_transProb == True:
        
        # Time evolve covariance matrix of the initial ground state of total Hamiltonian
        matM_evol = matU @ matM_init_even @ matU.T.conj()
        
        # Loop over chosen many-body excited states
        for ind in dict_fullsim["transProbs"]: 
            
            # Covariance matrix of excited state in the eigenbasis
            matMO_exc = pk.calc_matMO(L, R, ind)
            # Above covariance matrices in the original basis
            matM_exc_even, matM_exc_odd = pk.calc_matM(matMO_exc, matO_end)
            # Take the even result
            matM_exc = matM_exc_even
            # Transition probability
            dict_transProbs = pk.sqrmod_overlap_covar(matM_evol, matM_exc)
            
            # Dictionary key for each transition probability
            ind.sort()
            str_ind = list(map(str, ind))
            key_transProb = "transProb_" + "_".join(str_ind)
            
            # Storage
            dict_results_fullsim[key_transProb] = dict_transProbs
        
    # Transition probabilities (CC : init clean - compare clean)
    if flag_fullsim_transProb_cc == True:
        
        # Time evolve covariance matrix of the initial ground state of total Hamiltonian
        matM_evol = matU @ matM_init_even_clean @ matU.T.conj()
        
        # Loop over chosen many-body excited states
        for ind in dict_fullsim["transProbs_cc"]: 
            
            # Covariance matrix of excited state in the eigenbasis
            matMO_exc = pk.calc_matMO(L, R, ind)
            # Above covariance matrices in the original basis
            matM_exc_even, matM_exc_odd = pk.calc_matM(matMO_exc, matO_end_clean)
            # Take the even result
            matM_exc = matM_exc_even
            # Transition probability
            dict_transProbs_cc = pk.sqrmod_overlap_covar(matM_evol, matM_exc)
            
            # Dictionary key for each transition probability
            ind.sort()
            str_ind = list(map(str, ind))
            key_transProb = "transProb_cc_" + "_".join(str_ind)
            
            # Storage
            dict_results_fullsim[key_transProb] = dict_transProbs_cc
            
    # ------------------------------------- #
    # Other full simulation data management #
    # ------------------------------------- #
    
    # These results are kept separate from dict_results_fullsim
    dict_results_fullsim_other = {}
    
    # Minimum gap and location (single piano key only)
    # -> Recall that this requires the calculation of the spectrum (dict_operations_onesim["instEnergy"] == True)
    if flag_fullsim_mingap == True or flag_fullsim_sep_mingap == True:
        
        # Determine minimum gap and location
        energy_zero = results_pk["onesim_spec"]["instEnergy"][L+R,:]
        energy_exc = results_pk["onesim_spec"]["instEnergy"][L+R+1,:]
        time_rel = results_pk["onesim_spec"]["time"]/tau_fin
        ind_min = np.argmin(energy_exc - energy_zero)
        energy_mingap = (energy_exc - energy_zero)[ind_min]
        energy_mingap_loc = time_rel[ind_min]
        
        # Save to dictionary
        dict_results_fullsim_other["mingap"] = energy_mingap
        dict_results_fullsim_other["mingap_loc"] = energy_mingap_loc
        
    # Effective, normalized Landau-Zener velocity (single piano key only)
    # -> This velocity is defined as A, where the energy should follow sqrt((At/tau)**2 + B**2)
    # -> Recall that this requires the calculation of the spectrum (dict_operations_onesim["instEnergy"] == True)
    if flag_fullsim_velocity == True:
        
        # Determine minimum gap and location
        energy_zero = results_pk["onesim_spec"]["instEnergy"][L+R,:]
        energy_exc = results_pk["onesim_spec"]["instEnergy"][L+R+1,:]
        time_rel = results_pk["onesim_spec"]["time"]/tau_fin
        ind_min = np.argmin(energy_exc - energy_zero)
        energy_mingap = (energy_exc - energy_zero)[ind_min]
        energy_mingap_loc = time_rel[ind_min]
        
        # Use curve fitting to determine the Landau-Zener velocity
        # -> Relevant system parameters
        delta_mu = 0.5*np.abs(muRightStart - muRightEnd)
        # -> Relevant curve fitting parameters
        x_data_range = 15
        guess_init = [delta_mu*np.pi/2]
        # -> Data to fit
        x_data_fit = time_rel[ind_min-x_data_range:ind_min+x_data_range]
        y_data_fit = energy_exc[ind_min-x_data_range: ind_min+x_data_range]
        # -> Curve fit to square root function
        fit_bulkenerg_delta = lambda x, A, B=energy_mingap + energy_zero[ind_min] : np.sqrt((A*x)**2 + B**2)
        try:
            popt, pcov = sc_optim.curve_fit(fit_bulkenerg_delta, x_data_fit - energy_mingap_loc, y_data_fit, p0=guess_init)
            # -> Calculate effective velocity as A
            velocity_eff = popt[0]
        except: 
            velocity_eff = "FAIL"
        
        # Save to dictionary
        dict_results_fullsim_other["velocity"] = velocity_eff
        
        """
        pl.plot(time_rel, energy_exc - energy_zero)
        pl.axhline(energy_mingap, linestyle="--", color="Black")
        pl.axvline(energy_mingap_loc, linestyle="--", color="Black")
        pl.plot(time_rel, fit_bulkenerg_delta(time_rel - energy_mingap_loc, velocity_eff), color="Red")
        """
        
    # Return all results
    dict_results = {"fullsim":dict_results_fullsim, "onesim":dict_results_onesim}
    dict_results["fullsim_other"] = dict_results_fullsim_other
    dict_results.update(dict_results_onesim_other)
    
    """
    # Testing purposes:
    #import matplotlib.pyplot as pl
    pl.plot(arr_muRight_s_clean)
    pl.plot(arr_muN)
    """
    
    # --------------------- #
    # Final flag management #
    # --------------------- #
    
    # In the case of minimum gap and velocity calculations, the instantaneous spectrum is required -- turn flags off after calculations are performed
    if True in [flag_fullsim_mingap, flag_fullsim_velocity, flag_fullsim_sep_mingap]:
        dict_operations_main["onesim"] = False
        dict_operations_onesim["instEnergy"] = False
    
    return dict_results
    
