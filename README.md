# Human L2/3 Cortical Circuit Model for Testing New Pharmacology in Depression --- Guet-McCreight-et-al.-2023
===================================================================================================================================================================
Author: Alexandre Guet-McCreight

This is the readme for the model associated with the paper:

Guet-McCreight A, Chameh HM, Mazza F, Prevot TD, Valiante TA, Sibille E, Hay E (2023) In-silico testing of new pharmacology for restoring inhibition and human cortical function in depression.

This code is part of a provisional US patent. Name: EEG biomarkers for Alpha5-PAM therapy (US provisional patent no. 63/382,577).

Single Cell Gtonic Estimations (figure 1):
All code associated with single cell Gtonic Estimation is found in the /GTonicEstimation_Fig1/ directory.


Network Simulations:    
Note that high performance computing is necessary to run these simulations.

Simulation code associated with the L2/3 circuit used throughout the manuscript is in the /L23Net_Drug/ directory. Note that this circuit model is adapted from 10.5281/zenodo.5771000.

To run simulations, install all of the necessary python modules (see lfpy_env.yml), compile the mod files within the mod folder, and submit the simulations in parallel (e.g., see job.sh).

In job.sh, the number in the mpiexec command (see below - 1234) controls the random seed used for both the circuit variance (i.e., connection matrix, synapse placement, etc.) and the stimulus variance (i.e. Ornstein Uhlenbeck noise and stimulus presynaptic spike train timing). Simulations using 400 processors will take less than 10 mins when running 4.5s long simulations.

mpiexec -n 400 python circuit.py 1234

Here are some of the different configurations of parameters in circuit.py that we change to look at different conditions and levels of analysis.

Short Simulations Parameters (spiking activity + stimulus simulations):    
stimulate = True    
tstop = 4500.    

Long Simulations Parameters (EEG simulations):    
stimulate = False    
tstop = 25000.    

Healthy Parameters:    
MDD = False    
DRUG_a5PAM = 0    
DRUG_benzo = 0    

MDD Parameters:    
MDD = True    
DRUG_a5PAM = 0    
DRUG_benzo = 0    

MDD+a5PAM Parameters:    
MDD = True    
DRUG_a5PAM = 1.0 # or 0.25 to 1.50, depending on desired dose factor    
DRUG_benzo = 0    

MDD+nsPAM Parameters:    
MDD = True    
DRUG_a5PAM = 0    
DRUG_benzo = 1    


Analysis Code:    
All code used for analyzing the circuit simulation results is found in the /L23Net_Analyses/ directory. In all cases this code should run on personal machines - see below for specs of the system on which this code was tested.

MacBook Pro, 13-inch, 2019    
1.4 GHz Quad-Core    
Intel Core i5    
Intel Iris Plus Graphics     
645 1536 MB    
8 GB 2133 MHz    
LPDDR3    
Sonoma 14.0    

Simulated data is required to run these files - to obtain this data, run the network simulations (as above) using HPC resources and then process the simulated data to generate PSD outputs saved to npy files (see L23Net_Analyses/Build_PSD_NPY_files.py) or generate oscillatory event analysis outputs saved to csv files (L23Net_Analyses/Build_OEvents_csv_files.py), which can then more readily be analyzed by the code files in L23Net_Analyses/. The analysis codes perform analyses of multiple simulations across random seeds per condition.

For any code requiring the OEvents toolbox (i.e., Build_OEvents_csv_files.py or Plot_FigS4_PSD_OEvents_Analysis_thresh4.py) download and placement within the same folder of the OEvents toolbox is necessary ( available here: https://github.com/NathanKlineInstitute/OEvent ). Also note that creation of subfolders for plots and analysis results is necessary to run this code.
