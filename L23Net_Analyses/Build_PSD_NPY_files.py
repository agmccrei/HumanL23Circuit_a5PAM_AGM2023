################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import (LogLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as mticker
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import LFPy
import pandas as pd
from fooof import FOOOF

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

N_seeds = 50
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
startslice = 1000 # ms
endslice = 25000 # ms
t1 = int(startslice*(1/dt))
t2 = int(endslice*(1/dt))
tvec = np.arange(endslice/dt+1)*dt
nperseg = 140000 # len(tvec[t1:t2])/2

radii = [79000., 80000., 85000., 90000.]
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L23_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

conds = ['healthy_long','MDD_long','MDD_a5PAM_0.25_long','MDD_a5PAM_0.50_long','MDD_a5PAM_0.75_long','MDD_a5PAM_long','MDD_a5PAM_1.25_long','MDD_benzo_long']
paths = [i + '/HL23net_Drug/Circuit_output/' for i in conds]

for cind, path in enumerate(paths):
	eeg = []
	spikes = []
	spikes_PN = []
	for seed in N_seedsList:
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		# Load outputs
		temp_s = np.load(path + 'SPIKES_Seed' + str(seed) + '.npy',allow_pickle=True)
		temp_e = np.load(path + 'DIPOLEMOMENT_Seed' + str(seed) + '.npy')
		
		# Spikes PSD
		SPIKES1 = [x for _,x in sorted(zip(temp_s.item()['gids'][0],temp_s.item()['times'][0]))]
		SPIKES2 = [x for _,x in sorted(zip(temp_s.item()['gids'][1],temp_s.item()['times'][1]))]
		SPIKES3 = [x for _,x in sorted(zip(temp_s.item()['gids'][2],temp_s.item()['times'][2]))]
		SPIKES4 = [x for _,x in sorted(zip(temp_s.item()['gids'][3],temp_s.item()['times'][3]))]
		SPIKES_all = SPIKES1+SPIKES2+SPIKES3+SPIKES4
		
		popspikes_All = np.concatenate(SPIKES_all).ravel()
		popspikes_PN = np.concatenate(SPIKES1).ravel()
		popspikes_MN = np.concatenate(SPIKES2).ravel()
		popspikes_BN = np.concatenate(SPIKES3).ravel()
		popspikes_VN = np.concatenate(SPIKES4).ravel()
		spikebinvec = np.histogram(popspikes_All,bins=np.arange(startslice,endslice+dt,dt))[0]
		spikebinvec_PN = np.histogram(popspikes_PN,bins=np.arange(startslice,endslice+dt,dt))[0]
		spikebinvec_MN = np.histogram(popspikes_MN,bins=np.arange(startslice,endslice+dt,dt))[0]
		spikebinvec_BN = np.histogram(popspikes_BN,bins=np.arange(startslice,endslice+dt,dt))[0]
		spikebinvec_VN = np.histogram(popspikes_VN,bins=np.arange(startslice,endslice+dt,dt))[0]
		
		sampling_rate = (1/dt)*1000
		f_All, Pxx_den_All = ss.welch(spikebinvec, fs=sampling_rate, nperseg=nperseg)
		f_PN, Pxx_den_PN = ss.welch(spikebinvec_PN, fs=sampling_rate, nperseg=nperseg)
		f_MN, Pxx_den_MN = ss.welch(spikebinvec_MN, fs=sampling_rate, nperseg=nperseg)
		f_BN, Pxx_den_BN = ss.welch(spikebinvec_BN, fs=sampling_rate, nperseg=nperseg)
		f_VN, Pxx_den_VN = ss.welch(spikebinvec_VN, fs=sampling_rate, nperseg=nperseg)
		
		fmaxval = 101
		fmaxind = np.where(f_All>=fmaxval)[0][0]
		
		spikes.append(Pxx_den_All[:fmaxind])
		spikes_PN.append(Pxx_den_PN[:fmaxind])
		
		# EEG
		temp_e2 = temp_e['HL23PYR']
		temp_e2 = np.add(temp_e2,temp_e['HL23SST'])
		temp_e2 = np.add(temp_e2,temp_e['HL23PV'])
		temp_e2 = np.add(temp_e2,temp_e['HL23VIP'])
		
		potential = EEG_args.calc_potential(temp_e2, L23_pos)
		EEG = potential[0][t1:t2]
		
		freqrawEEG, psrawEEG = ss.welch(EEG, sampling_rate, nperseg=nperseg)
		fmaxind = np.where(freqrawEEG>=fmaxval)[0][0]
		eeg.append(psrawEEG[:fmaxind])
	
	np.save('Saved_PSDs_AllSeeds/'+conds[cind]+'_spikes',spikes)
	np.save('Saved_PSDs_AllSeeds/'+conds[cind]+'_spikes_PN',spikes_PN)
	np.save('Saved_PSDs_AllSeeds/'+conds[cind]+'_EEG',eeg)
