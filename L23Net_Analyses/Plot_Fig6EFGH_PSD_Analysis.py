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

conds = ['healthy_long','MDD_long','MDD_a5PAM_long','MDD_benzo_long']
paths = ['Saved_PSDs_AllSeeds/' + i for i in conds]

x_labels = ['Healthy','MDD','MDD\n'+r'$\alpha$'+'5-PAM','MDD\nBenzo.']
x_labels_types = ['Pyr','SST','PV','VIP']

colors_neurs = ['dimgrey', 'red', 'green', 'orange']
colors_conds = ['dimgrey', 'tab:purple', 'dodgerblue','chocolate']

thetaband = (4,8)
alphaband = (8,12)
betaband = (12,21)#30)
broadband = (3,30)#30)

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

eeg = [np.load(path + '_EEG.npy') for path in paths]
spikes = [np.load(path + '_spikes.npy') for path in paths]
spikes_PN = [np.load(path + '_spikes_PN.npy') for path in paths]

offsets = [[] for _ in conds]
exponents = [[] for _ in conds]
knees = [[] for _ in conds]
errors = [[] for _ in conds]
AUC = [[] for _ in conds]

Ls = [[] for _ in conds]
Gns = [[] for _ in conds]

center_freqs_t = [[] for _ in conds]
power_amps_t = [[] for _ in conds]
bandwidths_t = [[] for _ in conds]
num_peaks_t = [[] for _ in conds]
AUC_t = [[] for _ in conds]

center_freqs_a = [[] for _ in conds]
power_amps_a = [[] for _ in conds]
bandwidths_a = [[] for _ in conds]
num_peaks_a = [[] for _ in conds]
AUC_a = [[] for _ in conds]

center_freqs_b = [[] for _ in conds]
power_amps_b = [[] for _ in conds]
bandwidths_b = [[] for _ in conds]
num_peaks_b = [[] for _ in conds]
AUC_b = [[] for _ in conds]

def scalebar(axis,xy,lw=3):
	# xy = [left,right,bottom,top]
	xscalebar = np.array([xy[0],xy[1],xy[1]])
	yscalebar = np.array([xy[2],xy[2],xy[3]])
	axis.plot(xscalebar,yscalebar,'k',linewidth=lw)

for seed in N_seedsList:
	for cind, path in enumerate(paths):
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		
		f_res = 1/(nperseg*dt/1000)
		freqrawEEG = np.arange(0,101,f_res)
		
		frange_init = 3
		frange = [frange_init,30]
		fm = FOOOF(peak_width_limits=(2, 6.),
					min_peak_height=0,
					aperiodic_mode='fixed',
					max_n_peaks=3,
					peak_threshold=2.)
		
		# Adjusted code to dynamically change the range in the event of failed fits.
		fm.fit(freqrawEEG, eeg[cind][seed-1], frange)
		# ce = 0.25
		# while fm.error_ > 0.13:
		# 	frange[0] = frange[0]-ce
		# 	fm.fit(freqrawEEG, eeg[cind][seed-1], frange)
		# 	ce+=0.25
		
		fm.plot(save_fig=True,file_name='figs_FOOOFfits_V4/fits_'+conds[cind]+'_'+str(seed)+'.png')
		plt.close()
		offsets[cind].append(fm.aperiodic_params_[0])
		exponents[cind].append(fm.aperiodic_params_[-1])
		errors[cind].append(fm.error_)
		
		L = 10**fm._ap_fit[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
		Gn = fm._peak_fit[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
		F = fm.freqs[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
		
		Ls[cind].append(L)
		Gns[cind].append(Gn)
		
		inds_ = [iids[0] for iids in np.argwhere((F>=broadband[0]) & (F<=broadband[1]))]
		inds_t = [iids[0] for iids in np.argwhere((F>=thetaband[0]) & (F<=thetaband[1]))]
		inds_a = [iids[0] for iids in np.argwhere((F>=alphaband[0]) & (F<=alphaband[1]))]
		inds_b = [iids[0] for iids in np.argwhere((F>=betaband[0]) & (F<=betaband[1]))]
		
		AUC[cind].append(np.trapz(L[inds_],x=F[inds_]))
		AUC_t[cind].append(np.trapz(Gn[inds_t],x=F[inds_t]))
		AUC_a[cind].append(np.trapz(Gn[inds_a],x=F[inds_a]))
		AUC_b[cind].append(np.trapz(Gn[inds_b],x=F[inds_b]))
		
		tcount = 0
		acount = 0
		bcount = 0
		for ii in fm.peak_params_:
			cf = ii[0]
			pw = ii[1]
			bw = ii[2]
			if thetaband[0] <= cf <= thetaband[1]:
				center_freqs_t[cind].append(cf)
				power_amps_t[cind].append(pw)
				bandwidths_t[cind].append(bw)
				tcount =+ 1
			if alphaband[0] <= cf <= alphaband[1]:
				center_freqs_a[cind].append(cf)
				power_amps_a[cind].append(pw)
				bandwidths_a[cind].append(bw)
				acount =+ 1
			if betaband[0] <= cf <= betaband[1]:
				center_freqs_b[cind].append(cf)
				power_amps_b[cind].append(pw)
				bandwidths_b[cind].append(bw)
				bcount =+ 1
		num_peaks_t[cind].append(tcount)
		num_peaks_a[cind].append(acount)
		num_peaks_b[cind].append(bcount)

# Area under frequency bands - bar plots
offsets_m = [np.mean(allvals) for allvals in offsets]
exponents_m = [np.mean(allvals) for allvals in exponents]
errors_m = [np.mean(allvals) for allvals in errors]
center_freqs_t_m = [np.mean(allvals) for allvals in center_freqs_t]
power_amps_t_m = [np.mean(allvals) for allvals in power_amps_t]
bandwidths_t_m = [np.mean(allvals) for allvals in bandwidths_t]
num_peaks_t_m = [np.mean(allvals) for allvals in num_peaks_t]
center_freqs_a_m = [np.mean(allvals) for allvals in center_freqs_a]
power_amps_a_m = [np.mean(allvals) for allvals in power_amps_a]
bandwidths_a_m = [np.mean(allvals) for allvals in bandwidths_a]
num_peaks_a_m = [np.mean(allvals) for allvals in num_peaks_a]
center_freqs_b_m = [np.mean(allvals) for allvals in center_freqs_b]
power_amps_b_m = [np.mean(allvals) for allvals in power_amps_b]
bandwidths_b_m = [np.mean(allvals) for allvals in bandwidths_b]
num_peaks_b_m = [np.mean(allvals) for allvals in num_peaks_b]
AUC_m = [np.mean(allvals) for allvals in AUC]
AUC_t_m = [np.mean(allvals) for allvals in AUC_t]
AUC_a_m = [np.mean(allvals) for allvals in AUC_a]
AUC_b_m = [np.mean(allvals) for allvals in AUC_b]

offsets_sd = [np.std(allvals) for allvals in offsets]
exponents_sd = [np.std(allvals) for allvals in exponents]
errors_sd = [np.std(allvals) for allvals in errors]
center_freqs_t_sd = [np.std(allvals) for allvals in center_freqs_t]
power_amps_t_sd = [np.std(allvals) for allvals in power_amps_t]
bandwidths_t_sd = [np.std(allvals) for allvals in bandwidths_t]
num_peaks_t_sd = [np.std(allvals) for allvals in num_peaks_t]
center_freqs_a_sd = [np.std(allvals) for allvals in center_freqs_a]
power_amps_a_sd = [np.std(allvals) for allvals in power_amps_a]
bandwidths_a_sd = [np.std(allvals) for allvals in bandwidths_a]
num_peaks_a_sd = [np.std(allvals) for allvals in num_peaks_a]
center_freqs_b_sd = [np.std(allvals) for allvals in center_freqs_b]
power_amps_b_sd = [np.std(allvals) for allvals in power_amps_b]
bandwidths_b_sd = [np.std(allvals) for allvals in bandwidths_b]
num_peaks_b_sd = [np.std(allvals) for allvals in num_peaks_b]
AUC_sd = [np.std(allvals) for allvals in AUC]
AUC_t_sd = [np.std(allvals) for allvals in AUC_t]
AUC_a_sd = [np.std(allvals) for allvals in AUC_a]
AUC_b_sd = [np.std(allvals) for allvals in AUC_b]

# vs healthy
offsets_tstat0 = [st.ttest_rel(offsets[0],offsets[ind])[0] for ind in range(0,len(offsets))]
exponents_tstat0 = [st.ttest_rel(exponents[0],exponents[ind])[0] for ind in range(0,len(exponents))]
errors_tstat0 = [st.ttest_rel(errors[0],errors[ind])[0] for ind in range(0,len(errors))]
center_freqs_t_tstat0 = [st.ttest_ind(center_freqs_t[0],center_freqs_t[ind])[0] for ind in range(0,len(center_freqs_t))]
power_amps_t_tstat0 = [st.ttest_ind(power_amps_t[0],power_amps_t[ind])[0] for ind in range(0,len(power_amps_t))]
bandwidths_t_tstat0 = [st.ttest_ind(bandwidths_t[0],bandwidths_t[ind])[0] for ind in range(0,len(bandwidths_t))]
num_peaks_t_tstat0 = [st.ttest_ind(num_peaks_t[0],num_peaks_t[ind])[0] for ind in range(0,len(num_peaks_t))]
center_freqs_a_tstat0 = [st.ttest_ind(center_freqs_a[0],center_freqs_a[ind])[0] for ind in range(0,len(center_freqs_a))]
power_amps_a_tstat0 = [st.ttest_ind(power_amps_a[0],power_amps_a[ind])[0] for ind in range(0,len(power_amps_a))]
bandwidths_a_tstat0 = [st.ttest_ind(bandwidths_a[0],bandwidths_a[ind])[0] for ind in range(0,len(bandwidths_a))]
num_peaks_a_tstat0 = [st.ttest_ind(num_peaks_a[0],num_peaks_a[ind])[0] for ind in range(0,len(num_peaks_a))]
center_freqs_b_tstat0 = [st.ttest_ind(center_freqs_b[0],center_freqs_b[ind])[0] for ind in range(0,len(center_freqs_b))]
power_amps_b_tstat0 = [st.ttest_ind(power_amps_b[0],power_amps_b[ind])[0] for ind in range(0,len(power_amps_b))]
bandwidths_b_tstat0 = [st.ttest_ind(bandwidths_b[0],bandwidths_b[ind])[0] for ind in range(0,len(bandwidths_b))]
num_peaks_b_tstat0 = [st.ttest_ind(num_peaks_b[0],num_peaks_b[ind])[0] for ind in range(0,len(num_peaks_b))]
AUC_tstat0 = [st.ttest_rel(AUC[0],AUC[ind])[0] for ind in range(0,len(AUC))]
AUC_t_tstat0 = [st.ttest_rel(AUC_t[0],AUC_t[ind])[0] for ind in range(0,len(AUC_t))]
AUC_a_tstat0 = [st.ttest_rel(AUC_a[0],AUC_a[ind])[0] for ind in range(0,len(AUC_a))]
AUC_b_tstat0 = [st.ttest_rel(AUC_b[0],AUC_b[ind])[0] for ind in range(0,len(AUC_b))]

offsets_p0 = [st.ttest_rel(offsets[0],offsets[ind])[1] for ind in range(0,len(offsets))]
exponents_p0 = [st.ttest_rel(exponents[0],exponents[ind])[1] for ind in range(0,len(exponents))]
errors_p0 = [st.ttest_rel(errors[0],errors[ind])[1] for ind in range(0,len(errors))]
center_freqs_t_p0 = [st.ttest_ind(center_freqs_t[0],center_freqs_t[ind])[1] for ind in range(0,len(center_freqs_t))]
power_amps_t_p0 = [st.ttest_ind(power_amps_t[0],power_amps_t[ind])[1] for ind in range(0,len(power_amps_t))]
bandwidths_t_p0 = [st.ttest_ind(bandwidths_t[0],bandwidths_t[ind])[1] for ind in range(0,len(bandwidths_t))]
num_peaks_t_p0 = [st.ttest_ind(num_peaks_t[0],num_peaks_t[ind])[1] for ind in range(0,len(num_peaks_t))]
center_freqs_a_p0 = [st.ttest_ind(center_freqs_a[0],center_freqs_a[ind])[1] for ind in range(0,len(center_freqs_a))]
power_amps_a_p0 = [st.ttest_ind(power_amps_a[0],power_amps_a[ind])[1] for ind in range(0,len(power_amps_a))]
bandwidths_a_p0 = [st.ttest_ind(bandwidths_a[0],bandwidths_a[ind])[1] for ind in range(0,len(bandwidths_a))]
num_peaks_a_p0 = [st.ttest_ind(num_peaks_a[0],num_peaks_a[ind])[1] for ind in range(0,len(num_peaks_a))]
center_freqs_b_p0 = [st.ttest_ind(center_freqs_b[0],center_freqs_b[ind])[1] for ind in range(0,len(center_freqs_b))]
power_amps_b_p0 = [st.ttest_ind(power_amps_b[0],power_amps_b[ind])[1] for ind in range(0,len(power_amps_b))]
bandwidths_b_p0 = [st.ttest_ind(bandwidths_b[0],bandwidths_b[ind])[1] for ind in range(0,len(bandwidths_b))]
num_peaks_b_p0 = [st.ttest_ind(num_peaks_b[0],num_peaks_b[ind])[1] for ind in range(0,len(num_peaks_b))]
AUC_p0 = [st.ttest_rel(AUC[0],AUC[ind])[1] for ind in range(0,len(AUC))]
AUC_t_p0 = [st.ttest_rel(AUC_t[0],AUC_t[ind])[1] for ind in range(0,len(AUC_t))]
AUC_a_p0 = [st.ttest_rel(AUC_a[0],AUC_a[ind])[1] for ind in range(0,len(AUC_a))]
AUC_b_p0 = [st.ttest_rel(AUC_b[0],AUC_b[ind])[1] for ind in range(0,len(AUC_b))]

offsets_cd0 = [cohen_d(offsets[0],offsets[ind]) for ind in range(0,len(offsets))]
exponents_cd0 = [cohen_d(exponents[0],exponents[ind]) for ind in range(0,len(exponents))]
errors_cd0 = [cohen_d(errors[0],errors[ind]) for ind in range(0,len(errors))]
center_freqs_t_cd0 = [cohen_d(center_freqs_t[0],center_freqs_t[ind]) for ind in range(0,len(center_freqs_t))]
power_amps_t_cd0 = [cohen_d(power_amps_t[0],power_amps_t[ind]) for ind in range(0,len(power_amps_t))]
bandwidths_t_cd0 = [cohen_d(bandwidths_t[0],bandwidths_t[ind]) for ind in range(0,len(bandwidths_t))]
num_peaks_t_cd0 = [cohen_d(num_peaks_t[0],num_peaks_t[ind]) for ind in range(0,len(num_peaks_t))]
center_freqs_a_cd0 = [cohen_d(center_freqs_a[0],center_freqs_a[ind]) for ind in range(0,len(center_freqs_a))]
power_amps_a_cd0 = [cohen_d(power_amps_a[0],power_amps_a[ind]) for ind in range(0,len(power_amps_a))]
bandwidths_a_cd0 = [cohen_d(bandwidths_a[0],bandwidths_a[ind]) for ind in range(0,len(bandwidths_a))]
num_peaks_a_cd0 = [cohen_d(num_peaks_a[0],num_peaks_a[ind]) for ind in range(0,len(num_peaks_a))]
center_freqs_b_cd0 = [cohen_d(center_freqs_b[0],center_freqs_b[ind]) for ind in range(0,len(center_freqs_b))]
power_amps_b_cd0 = [cohen_d(power_amps_b[0],power_amps_b[ind]) for ind in range(0,len(power_amps_b))]
bandwidths_b_cd0 = [cohen_d(bandwidths_b[0],bandwidths_b[ind]) for ind in range(0,len(bandwidths_b))]
num_peaks_b_cd0 = [cohen_d(num_peaks_b[0],num_peaks_b[ind]) for ind in range(0,len(num_peaks_b))]
AUC_cd0 = [cohen_d(AUC[0],AUC[ind]) for ind in range(0,len(AUC))]
AUC_t_cd0 = [cohen_d(AUC_t[0],AUC_t[ind]) for ind in range(0,len(AUC_t))]
AUC_a_cd0 = [cohen_d(AUC_a[0],AUC_a[ind]) for ind in range(0,len(AUC_a))]
AUC_b_cd0 = [cohen_d(AUC_b[0],AUC_b[ind]) for ind in range(0,len(AUC_b))]

# vs MDD
offsets_tstat = [st.ttest_rel(offsets[1],offsets[ind])[0] for ind in range(0,len(offsets))]
exponents_tstat = [st.ttest_rel(exponents[1],exponents[ind])[0] for ind in range(0,len(exponents))]
errors_tstat = [st.ttest_rel(errors[1],errors[ind])[0] for ind in range(0,len(errors))]
center_freqs_t_tstat = [st.ttest_ind(center_freqs_t[1],center_freqs_t[ind])[0] for ind in range(0,len(center_freqs_t))]
power_amps_t_tstat = [st.ttest_ind(power_amps_t[1],power_amps_t[ind])[0] for ind in range(0,len(power_amps_t))]
bandwidths_t_tstat = [st.ttest_ind(bandwidths_t[1],bandwidths_t[ind])[0] for ind in range(0,len(bandwidths_t))]
num_peaks_t_tstat = [st.ttest_ind(num_peaks_t[1],num_peaks_t[ind])[0] for ind in range(0,len(num_peaks_t))]
center_freqs_a_tstat = [st.ttest_ind(center_freqs_a[1],center_freqs_a[ind])[0] for ind in range(0,len(center_freqs_a))]
power_amps_a_tstat = [st.ttest_ind(power_amps_a[1],power_amps_a[ind])[0] for ind in range(0,len(power_amps_a))]
bandwidths_a_tstat = [st.ttest_ind(bandwidths_a[1],bandwidths_a[ind])[0] for ind in range(0,len(bandwidths_a))]
num_peaks_a_tstat = [st.ttest_ind(num_peaks_a[1],num_peaks_a[ind])[0] for ind in range(0,len(num_peaks_a))]
center_freqs_b_tstat = [st.ttest_ind(center_freqs_b[1],center_freqs_b[ind])[0] for ind in range(0,len(center_freqs_b))]
power_amps_b_tstat = [st.ttest_ind(power_amps_b[1],power_amps_b[ind])[0] for ind in range(0,len(power_amps_b))]
bandwidths_b_tstat = [st.ttest_ind(bandwidths_b[1],bandwidths_b[ind])[0] for ind in range(0,len(bandwidths_b))]
num_peaks_b_tstat = [st.ttest_ind(num_peaks_b[1],num_peaks_b[ind])[0] for ind in range(0,len(num_peaks_b))]
AUC_tstat = [st.ttest_rel(AUC[1],AUC[ind])[0] for ind in range(0,len(AUC))]
AUC_t_tstat = [st.ttest_rel(AUC_t[1],AUC_t[ind])[0] for ind in range(0,len(AUC_t))]
AUC_a_tstat = [st.ttest_rel(AUC_a[1],AUC_a[ind])[0] for ind in range(0,len(AUC_a))]
AUC_b_tstat = [st.ttest_rel(AUC_b[1],AUC_b[ind])[0] for ind in range(0,len(AUC_b))]

offsets_p = [st.ttest_rel(offsets[1],offsets[ind])[1] for ind in range(0,len(offsets))]
exponents_p = [st.ttest_rel(exponents[1],exponents[ind])[1] for ind in range(0,len(exponents))]
errors_p = [st.ttest_rel(errors[1],errors[ind])[1] for ind in range(0,len(errors))]
center_freqs_t_p = [st.ttest_ind(center_freqs_t[1],center_freqs_t[ind])[1] for ind in range(0,len(center_freqs_t))]
power_amps_t_p = [st.ttest_ind(power_amps_t[1],power_amps_t[ind])[1] for ind in range(0,len(power_amps_t))]
bandwidths_t_p = [st.ttest_ind(bandwidths_t[1],bandwidths_t[ind])[1] for ind in range(0,len(bandwidths_t))]
num_peaks_t_p = [st.ttest_ind(num_peaks_t[1],num_peaks_t[ind])[1] for ind in range(0,len(num_peaks_t))]
center_freqs_a_p = [st.ttest_ind(center_freqs_a[1],center_freqs_a[ind])[1] for ind in range(0,len(center_freqs_a))]
power_amps_a_p = [st.ttest_ind(power_amps_a[1],power_amps_a[ind])[1] for ind in range(0,len(power_amps_a))]
bandwidths_a_p = [st.ttest_ind(bandwidths_a[1],bandwidths_a[ind])[1] for ind in range(0,len(bandwidths_a))]
num_peaks_a_p = [st.ttest_ind(num_peaks_a[1],num_peaks_a[ind])[1] for ind in range(0,len(num_peaks_a))]
center_freqs_b_p = [st.ttest_ind(center_freqs_b[1],center_freqs_b[ind])[1] for ind in range(0,len(center_freqs_b))]
power_amps_b_p = [st.ttest_ind(power_amps_b[1],power_amps_b[ind])[1] for ind in range(0,len(power_amps_b))]
bandwidths_b_p = [st.ttest_ind(bandwidths_b[1],bandwidths_b[ind])[1] for ind in range(0,len(bandwidths_b))]
num_peaks_b_p = [st.ttest_ind(num_peaks_b[1],num_peaks_b[ind])[1] for ind in range(0,len(num_peaks_b))]
AUC_p = [st.ttest_rel(AUC[1],AUC[ind])[1] for ind in range(0,len(AUC))]
AUC_t_p = [st.ttest_rel(AUC_t[1],AUC_t[ind])[1] for ind in range(0,len(AUC_t))]
AUC_a_p = [st.ttest_rel(AUC_a[1],AUC_a[ind])[1] for ind in range(0,len(AUC_a))]
AUC_b_p = [st.ttest_rel(AUC_b[1],AUC_b[ind])[1] for ind in range(0,len(AUC_b))]

offsets_cd = [cohen_d(offsets[1],offsets[ind]) for ind in range(0,len(offsets))]
exponents_cd = [cohen_d(exponents[1],exponents[ind]) for ind in range(0,len(exponents))]
errors_cd = [cohen_d(errors[1],errors[ind]) for ind in range(0,len(errors))]
center_freqs_t_cd = [cohen_d(center_freqs_t[1],center_freqs_t[ind]) for ind in range(0,len(center_freqs_t))]
power_amps_t_cd = [cohen_d(power_amps_t[1],power_amps_t[ind]) for ind in range(0,len(power_amps_t))]
bandwidths_t_cd = [cohen_d(bandwidths_t[1],bandwidths_t[ind]) for ind in range(0,len(bandwidths_t))]
num_peaks_t_cd = [cohen_d(num_peaks_t[1],num_peaks_t[ind]) for ind in range(0,len(num_peaks_t))]
center_freqs_a_cd = [cohen_d(center_freqs_a[1],center_freqs_a[ind]) for ind in range(0,len(center_freqs_a))]
power_amps_a_cd = [cohen_d(power_amps_a[1],power_amps_a[ind]) for ind in range(0,len(power_amps_a))]
bandwidths_a_cd = [cohen_d(bandwidths_a[1],bandwidths_a[ind]) for ind in range(0,len(bandwidths_a))]
num_peaks_a_cd = [cohen_d(num_peaks_a[1],num_peaks_a[ind]) for ind in range(0,len(num_peaks_a))]
center_freqs_b_cd = [cohen_d(center_freqs_b[1],center_freqs_b[ind]) for ind in range(0,len(center_freqs_b))]
power_amps_b_cd = [cohen_d(power_amps_b[1],power_amps_b[ind]) for ind in range(0,len(power_amps_b))]
bandwidths_b_cd = [cohen_d(bandwidths_b[1],bandwidths_b[ind]) for ind in range(0,len(bandwidths_b))]
num_peaks_b_cd = [cohen_d(num_peaks_b[1],num_peaks_b[ind]) for ind in range(0,len(num_peaks_b))]
AUC_cd = [cohen_d(AUC[1],AUC[ind]) for ind in range(0,len(AUC))]
AUC_t_cd = [cohen_d(AUC_t[1],AUC_t[ind]) for ind in range(0,len(AUC_t))]
AUC_a_cd = [cohen_d(AUC_a[1],AUC_a[ind]) for ind in range(0,len(AUC_a))]
AUC_b_cd = [cohen_d(AUC_b[1],AUC_b[ind]) for ind in range(0,len(AUC_b))]

df = pd.DataFrame(columns=["Metric",
			"Comparison",
			"G1 Mean",
			"G1 SD",
			"G2 Mean",
			"G2 SD",
			"t-stat",
			"p-value",
			"Cohen's d"])


metric_names = ["Offset",
				"Exponent",
				"Error",
				"Aperiodic AUC",
				"Theta AUC",
				"Alpha AUC",
				"Beta AUC",
				"Center Frequency (Theta)",
				"Relative Power (Theta)",
				"Bandwidth (Theta)",
				"Number of Peaks (Theta)",
				"Center Frequency (alpha)",
				"Relative Power (alpha)",
				"Bandwidth (alpha)",
				"Number of Peaks (alpha)",
				"Center Frequency (beta)",
				"Relative Power (beta)",
				"Bandwidth (beta)",
				"Number of Peaks (beta)"]

x_labels2 = ['Healthy','MDD','MDD + a5-PAM','MDD + Benzo.']

m = [offsets_m,
	exponents_m,
	errors_m,
	AUC_m,
	AUC_t_m,
	AUC_a_m,
	AUC_b_m,
	center_freqs_t_m,
	power_amps_t_m,
	bandwidths_t_m,
	num_peaks_t_m,
	center_freqs_a_m,
	power_amps_a_m,
	bandwidths_a_m,
	num_peaks_a_m,
	center_freqs_b_m,
	power_amps_b_m,
	bandwidths_b_m,
	num_peaks_b_m]
sd = [offsets_sd,
	exponents_sd,
	errors_sd,
	AUC_sd,
	AUC_t_sd,
	AUC_a_sd,
	AUC_b_sd,
	center_freqs_t_sd,
	power_amps_t_sd,
	bandwidths_t_sd,
	num_peaks_t_sd,
	center_freqs_a_sd,
	power_amps_a_sd,
	bandwidths_a_sd,
	num_peaks_a_sd,
	center_freqs_b_sd,
	power_amps_b_sd,
	bandwidths_b_sd,
	num_peaks_b_sd]

tstat0 = [offsets_tstat0,
	exponents_tstat0,
	errors_tstat0,
	AUC_tstat0,
	AUC_t_tstat0,
	AUC_a_tstat0,
	AUC_b_tstat0,
	center_freqs_t_tstat0,
	power_amps_t_tstat0,
	bandwidths_t_tstat0,
	num_peaks_t_tstat0,
	center_freqs_a_tstat0,
	power_amps_a_tstat0,
	bandwidths_a_tstat0,
	num_peaks_a_tstat0,
	center_freqs_b_tstat0,
	power_amps_b_tstat0,
	bandwidths_b_tstat0,
	num_peaks_b_tstat0]
p0 = [offsets_p0,
	exponents_p0,
	errors_p0,
	AUC_p0,
	AUC_t_p0,
	AUC_a_p0,
	AUC_b_p0,
	center_freqs_t_p0,
	power_amps_t_p0,
	bandwidths_t_p0,
	num_peaks_t_p0,
	center_freqs_a_p0,
	power_amps_a_p0,
	bandwidths_a_p0,
	num_peaks_a_p0,
	center_freqs_b_p0,
	power_amps_b_p0,
	bandwidths_b_p0,
	num_peaks_b_p0]
cd0 = [offsets_cd0,
	exponents_cd0,
	errors_cd0,
	AUC_cd0,
	AUC_t_cd0,
	AUC_a_cd0,
	AUC_b_cd0,
	center_freqs_t_cd0,
	power_amps_t_cd0,
	bandwidths_t_cd0,
	num_peaks_t_cd0,
	center_freqs_a_cd0,
	power_amps_a_cd0,
	bandwidths_a_cd0,
	num_peaks_a_cd0,
	center_freqs_b_cd0,
	power_amps_b_cd0,
	bandwidths_b_cd0,
	num_peaks_b_cd0]

tstat = [offsets_tstat,
	exponents_tstat,
	errors_tstat,
	AUC_tstat,
	AUC_t_tstat,
	AUC_a_tstat,
	AUC_b_tstat,
	center_freqs_t_tstat,
	power_amps_t_tstat,
	bandwidths_t_tstat,
	num_peaks_t_tstat,
	center_freqs_a_tstat,
	power_amps_a_tstat,
	bandwidths_a_tstat,
	num_peaks_a_tstat,
	center_freqs_b_tstat,
	power_amps_b_tstat,
	bandwidths_b_tstat,
	num_peaks_b_tstat]
p = [offsets_p,
	exponents_p,errors_p,
	AUC_p,
	AUC_t_p,
	AUC_a_p,
	AUC_b_p,
	center_freqs_t_p,
	power_amps_t_p,
	bandwidths_t_p,
	num_peaks_t_p,
	center_freqs_a_p,
	power_amps_a_p,
	bandwidths_a_p,
	num_peaks_a_p,
	center_freqs_b_p,
	power_amps_b_p,
	bandwidths_b_p,
	num_peaks_b_p]
cd = [offsets_cd,
	exponents_cd,
	errors_cd,
	AUC_cd,
	AUC_t_cd,
	AUC_a_cd,
	AUC_b_cd,
	center_freqs_t_cd,
	power_amps_t_cd,
	bandwidths_t_cd,
	num_peaks_t_cd,
	center_freqs_a_cd,
	power_amps_a_cd,
	bandwidths_a_cd,
	num_peaks_a_cd,
	center_freqs_b_cd,
	power_amps_b_cd,
	bandwidths_b_cd,
	num_peaks_b_cd]

# vs Healthy
for lind,labeli in enumerate(x_labels2):
	for ind1,metric in enumerate(metric_names):
		df = df.append({"Metric":metric,
					"Comparison":x_labels2[0] + ' vs ' + labeli,
					"G1 Mean":m[ind1][0],
					"G1 SD":sd[ind1][0],
					"G2 Mean":m[ind1][lind],
					"G2 SD":sd[ind1][lind],
					"t-stat":tstat0[ind1][lind],
					"p-value":p0[ind1][lind],
					"Cohen's d":cd0[ind1][lind]},
					ignore_index = True)

# vs MDD
for lind,labeli in enumerate(x_labels2):
	for ind1,metric in enumerate(metric_names):
		df = df.append({"Metric":metric,
					"Comparison":x_labels2[1] + ' vs ' + labeli,
					"G1 Mean":m[ind1][1],
					"G1 SD":sd[ind1][1],
					"G2 Mean":m[ind1][lind],
					"G2 SD":sd[ind1][lind],
					"t-stat":tstat[ind1][lind],
					"p-value":p[ind1][lind],
					"Cohen's d":cd[ind1][lind]},
					ignore_index = True)

df.to_csv('figs_PSD_V6/stats_PSD.csv')


figsize1 = (7,5)
figsize2 = (14,5)
dh1 = 0.03

p_thresh = 0.05
c_thresh = 1.

xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(offsets[cind]))*0.4-0.2),offsets[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[offsets_m[cind],offsets_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if offsets_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, offsets_m, yerr=[sd*2 for sd in offsets_sd], dh=dh1)
	if ((offsets_p0[cind] < p_thresh) & (abs(offsets_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],offsets_m[cind]+offsets_sd[cind]+offsets_sd[cind]/2,'*',c='k',ha='right' if ((offsets_p[cind] < p_thresh) & (abs(offsets_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((offsets_p[cind] < p_thresh) & (abs(offsets_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],offsets_m[cind]+offsets_sd[cind]+offsets_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((offsets_p0[cind] < p_thresh) & (abs(offsets_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel(r'Offset (mV$^2$)')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_offsets.png',dpi=300,transparent=True)

xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(AUC[cind]))*0.4-0.2),AUC[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_m[cind],AUC_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, AUC_m, yerr=[sd*2 for sd in AUC_sd], dh=dh1)
	if ((AUC_p0[cind] < p_thresh) & (abs(AUC_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_m[cind]+AUC_sd[cind]+AUC_sd[cind]/2,'*',c='k',ha='right' if ((AUC_p[cind] < p_thresh) & (abs(AUC_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_p[cind] < p_thresh) & (abs(AUC_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_m[cind]+AUC_sd[cind]+AUC_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_p0[cind] < p_thresh) & (abs(AUC_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel('AUC')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_aperiodicAUC.png',dpi=300,transparent=True)

# Plot FOOOF outputs
def dotplot_multigroup(metric,metric_m,metric_sd,metric_p0,metric_p,metric_cd0,metric_cd,ylabelname,filename):
	xinds = np.arange(0,len(conds))
	fig_bands, ax_bands = plt.subplots(figsize=figsize1)
	for cind in range(len(conds)):
		ax_bands.scatter(xinds[cind]+(np.random.random(len(metric[cind]))*0.4-0.2),metric[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
		ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[metric_m[cind],metric_m[cind]],'k',alpha=1,linewidth=3)
		ln1.set_solid_capstyle('round')
		
		if ((metric_p0[cind] < p_thresh) & (abs(metric_cd0[cind]) > c_thresh) & (cind>0)):
			ax_bands.text(xinds[cind],metric_m[cind]+metric_sd[cind]+metric_sd[cind],'*',c='k',ha='right' if ((metric_p[cind] < p_thresh) & (abs(metric_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
		if ((metric_p[cind] < p_thresh) & (abs(metric_cd[cind]) > c_thresh) & (cind>0)):
			ax_bands.text(xinds[cind],metric_m[cind]+metric_sd[cind]+metric_sd[cind],'*',c=colors_conds[1],ha='left' if ((metric_p0[cind] < p_thresh) & (abs(metric_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

	ax_bands.set_ylabel(ylabelname)
	ax_bands.set_xticks(xinds)
	ax_bands.set_xticklabels(x_labels, rotation = 45, ha="center")
	ax_bands.grid(False)
	ax_bands.spines['right'].set_visible(False)
	ax_bands.spines['top'].set_visible(False)
	fig_bands.tight_layout()
	fig_bands.savefig('figs_PSD_V6/'+filename+'.png',dpi=300,transparent=True)
	plt.close()

dotplot_multigroup(center_freqs_t,center_freqs_t_m,center_freqs_t_sd,center_freqs_t_p0,center_freqs_t_p,center_freqs_t_cd0,center_freqs_t_cd,r'$\theta$'+' Centre (Hz)','center_freqs_theta')
dotplot_multigroup(power_amps_t,power_amps_t_m,power_amps_t_sd,power_amps_t_p0,power_amps_t_p,power_amps_t_cd0,power_amps_t_cd,r'$\theta$'+' Peak','power_amps_theta')
dotplot_multigroup(bandwidths_t,bandwidths_t_m,bandwidths_t_sd,bandwidths_t_p0,bandwidths_t_p,bandwidths_t_cd0,bandwidths_t_cd,r'$\theta$'+' Bandwidth (Hz)','bandwidths_theta')
dotplot_multigroup(num_peaks_t,num_peaks_t_m,num_peaks_t_sd,num_peaks_t_p0,num_peaks_t_p,num_peaks_t_cd0,num_peaks_t_cd,r'$\theta$'+' Number of Peaks','num_peaks_theta')

dotplot_multigroup(center_freqs_a,center_freqs_a_m,center_freqs_a_sd,center_freqs_a_p0,center_freqs_a_p,center_freqs_a_cd0,center_freqs_a_cd,r'$\alpha$'+' Centre (Hz)','center_freqs_alpha')
dotplot_multigroup(power_amps_a,power_amps_a_m,power_amps_a_sd,power_amps_a_p0,power_amps_a_p,power_amps_a_cd0,power_amps_a_cd,r'$\alpha$'+' Peak','power_amps_alpha')
dotplot_multigroup(bandwidths_a,bandwidths_a_m,bandwidths_a_sd,bandwidths_a_p0,bandwidths_a_p,bandwidths_a_cd0,bandwidths_a_cd,r'$\alpha$'+' Bandwidth (Hz)','bandwidths_alpha')
dotplot_multigroup(num_peaks_a,num_peaks_a_m,num_peaks_a_sd,num_peaks_a_p0,num_peaks_a_p,num_peaks_a_cd0,num_peaks_a_cd,r'$\alpha$'+' Number of Peaks','num_peaks_alpha')

dotplot_multigroup(center_freqs_b,center_freqs_b_m,center_freqs_b_sd,center_freqs_b_p0,center_freqs_b_p,center_freqs_b_cd0,center_freqs_b_cd,r'$\beta$'+' Centre (Hz)','center_freqs_beta')
dotplot_multigroup(power_amps_b,power_amps_b_m,power_amps_b_sd,power_amps_b_p0,power_amps_b_p,power_amps_b_cd0,power_amps_b_cd,r'$\beta$'+' Peak','power_amps_beta')
dotplot_multigroup(bandwidths_b,bandwidths_b_m,bandwidths_b_sd,bandwidths_b_p0,bandwidths_b_p,bandwidths_b_cd0,bandwidths_b_cd,r'$\beta$'+' Bandwidth (Hz)','bandwidths_beta')
dotplot_multigroup(num_peaks_b,num_peaks_b_m,num_peaks_b_sd,num_peaks_b_p0,num_peaks_b_p,num_peaks_b_cd0,num_peaks_b_cd,r'$\beta$'+' Number of Peaks','num_peaks_beta')



xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(AUC_t[cind]))*0.4-0.2),AUC_t[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_t_m[cind],AUC_t_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_t_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, AUC_t_m, yerr=[sd*2 for sd in AUC_t_sd], dh=dh1)
	if ((AUC_t_p0[cind] < p_thresh) & (abs(AUC_t_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_t_m[cind]+AUC_t_sd[cind]+AUC_t_sd[cind]/2,'*',c='k',ha='right' if ((AUC_t_p[cind] < p_thresh) & (abs(AUC_t_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_t_p[cind] < p_thresh) & (abs(AUC_t_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_t_m[cind]+AUC_t_sd[cind]+AUC_t_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_t_p0[cind] < p_thresh) & (abs(AUC_t_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel('Theta AUC')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_thetaAUC.png',dpi=300,transparent=True)

xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(AUC_a[cind]))*0.4-0.2),AUC_a[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_a_m[cind],AUC_a_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_a_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, AUC_a_m, yerr=[sd*2 for sd in AUC_a_sd], dh=dh1)
	if ((AUC_a_p0[cind] < p_thresh) & (abs(AUC_a_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_a_m[cind]+AUC_a_sd[cind]+AUC_a_sd[cind]/2,'*',c='k',ha='right' if ((AUC_a_p[cind] < p_thresh) & (abs(AUC_a_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_a_p[cind] < p_thresh) & (abs(AUC_a_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_a_m[cind]+AUC_a_sd[cind]+AUC_a_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_a_p0[cind] < p_thresh) & (abs(AUC_a_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel('Alpha AUC')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_alphaAUC.png',dpi=300,transparent=True)

xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(AUC_b[cind]))*0.4-0.2),AUC_b[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_b_m[cind],AUC_b_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_b_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, AUC_b_m, yerr=[sd*2 for sd in AUC_b_sd], dh=dh1)
	if ((AUC_b_p0[cind] < p_thresh) & (abs(AUC_b_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_b_m[cind]+AUC_b_sd[cind]+AUC_b_sd[cind]/2,'*',c='k',ha='right' if ((AUC_b_p[cind] < p_thresh) & (abs(AUC_b_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_b_p[cind] < p_thresh) & (abs(AUC_b_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],AUC_b_m[cind]+AUC_b_sd[cind]+AUC_b_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_b_p0[cind] < p_thresh) & (abs(AUC_b_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel('Beta AUC')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_betaAUC.png',dpi=300,transparent=True)


xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(exponents[cind]))*0.4-0.2),exponents[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[exponents_m[cind],exponents_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if exponents_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, exponents_m, yerr=[sd*2 for sd in exponents_sd], dh=dh1)
	if ((exponents_p0[cind] < p_thresh) & (abs(exponents_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],exponents_m[cind]+exponents_sd[cind]+exponents_sd[cind]/2,'*',c='k',ha='right' if ((exponents_p[cind] < p_thresh) & (abs(exponents_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((exponents_p[cind] < p_thresh) & (abs(exponents_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],exponents_m[cind]+exponents_sd[cind]+exponents_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((exponents_p0[cind] < p_thresh) & (abs(exponents_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel(r'Exponent (mV$^2$/Hz)')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_exponents.png',dpi=300,transparent=True)

xinds = np.arange(0,len(conds))
fig_bands, ax_bands = plt.subplots(figsize=figsize1)
for cind in range(len(conds)):
	ax_bands.scatter(xinds[cind]+(np.random.random(len(errors[cind]))*0.4-0.2),errors[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = ax_bands.plot([xinds[cind]-0.25,xinds[cind]+0.25],[errors_m[cind],errors_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if errors_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, ax_bands, cind-1, cind, '*', xinds, errors_m, yerr=[sd*2 for sd in errors_sd], dh=dh1)
	if ((errors_p0[cind] < p_thresh) & (abs(errors_cd0[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],errors_m[cind]+errors_sd[cind]+errors_sd[cind]/2,'*',c='k',ha='right' if ((errors_p[cind] < p_thresh) & (abs(errors_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((errors_p[cind] < p_thresh) & (abs(errors_cd[cind]) > c_thresh) & (cind>0)):
		ax_bands.text(xinds[cind],errors_m[cind]+errors_sd[cind]+errors_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((errors_p0[cind] < p_thresh) & (abs(errors_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

ax_bands.set_ylabel('Error')
ax_bands.set_xticks(xinds)
ax_bands.set_xticklabels(x_labels)
ax_bands.grid(False)
ax_bands.spines['right'].set_visible(False)
ax_bands.spines['top'].set_visible(False)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_errors.png',dpi=300,transparent=True)



fig_bands, ax_bands = plt.subplots(nrows=1,ncols=len(conds),figsize=figsize2,sharex=True,sharey=True)
for cind in range(len(conds)):
	for cf,pw,bw in zip(center_freqs_t[cind],power_amps_t[cind],bandwidths_t[cind]):
		ax_bands[cind].plot([cf-bw/2,cf+bw/2],[pw,pw],colors_conds[cind],alpha=0.5,linewidth=1)
		ax_bands[cind].scatter(cf,pw,s=2,facecolor=colors_conds[cind],edgecolors='face',alpha=0.5)
	
	ln1, = ax_bands[cind].plot([center_freqs_t_m[cind]-bandwidths_t_m[cind]/2,center_freqs_t_m[cind]+bandwidths_t_m[cind]/2],[power_amps_t_m[cind],power_amps_t_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	ax_bands[cind].scatter(center_freqs_t_m[cind],power_amps_t_m[cind],s=13,facecolor='k',edgecolors='face',alpha=1)
	ax_bands[cind].set_xlabel(x_labels[cind])
	ax_bands[cind].grid(False)
	ax_bands[cind].spines['right'].set_visible(False)
	ax_bands[cind].spines['top'].set_visible(False)

ax_bands[0].set_ylabel(r'Power (mV$^2$)')
ax_bands[0].set_xlim(thetaband[0]-2,thetaband[1]+2)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_Theta.png',dpi=300,transparent=True)


fig_bands, ax_bands = plt.subplots(nrows=1,ncols=len(conds),figsize=figsize2,sharex=True,sharey=True)
for cind in range(len(conds)):
	for cf,pw,bw in zip(center_freqs_a[cind],power_amps_a[cind],bandwidths_a[cind]):
		ax_bands[cind].plot([cf-bw/2,cf+bw/2],[pw,pw],colors_conds[cind],alpha=0.5,linewidth=1)
		ax_bands[cind].scatter(cf,pw,s=2,facecolor=colors_conds[cind],edgecolors='face',alpha=0.5)
	
	ln1, = ax_bands[cind].plot([center_freqs_a_m[cind]-bandwidths_a_m[cind]/2,center_freqs_a_m[cind]+bandwidths_a_m[cind]/2],[power_amps_a_m[cind],power_amps_a_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	ax_bands[cind].scatter(center_freqs_a_m[cind],power_amps_a_m[cind],s=13,facecolor='k',edgecolors='face',alpha=1)
	ax_bands[cind].set_xlabel(x_labels[cind])
	ax_bands[cind].grid(False)
	ax_bands[cind].spines['right'].set_visible(False)
	ax_bands[cind].spines['top'].set_visible(False)

ax_bands[0].set_ylabel(r'Power (mV$^2$)')
ax_bands[0].set_xlim(alphaband[0]-3,alphaband[1]+3)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_Alpha.png',dpi=300,transparent=True)

fig_bands, ax_bands = plt.subplots(nrows=1,ncols=len(conds),figsize=figsize2,sharex=True,sharey=True)
for cind in range(len(conds)):
	for cf,pw,bw in zip(center_freqs_b[cind],power_amps_b[cind],bandwidths_b[cind]):
		ax_bands[cind].plot([cf-bw/2,cf+bw/2],[pw,pw],colors_conds[cind],alpha=0.5,linewidth=1)
		ax_bands[cind].scatter(cf,pw,s=2,facecolor=colors_conds[cind],edgecolors='face',alpha=0.5)
	
	ln1, = ax_bands[cind].plot([center_freqs_b_m[cind]-bandwidths_b_m[cind]/2,center_freqs_b_m[cind]+bandwidths_b_m[cind]/2],[power_amps_b_m[cind],power_amps_b_m[cind]],'k',alpha=1,linewidth=2)
	ln1.set_solid_capstyle('round')
	ax_bands[cind].scatter(center_freqs_b_m[cind],power_amps_b_m[cind],s=13,facecolor='k',edgecolors='face',alpha=1)
	ax_bands[cind].set_xlabel(x_labels[cind])
	ax_bands[cind].grid(False)
	ax_bands[cind].spines['right'].set_visible(False)
	ax_bands[cind].spines['top'].set_visible(False)

ax_bands[0].set_ylabel(r'Power (mV$^2$)')
ax_bands[0].set_xlim(betaband[0]-3,betaband[1]+3)
fig_bands.tight_layout()
fig_bands.savefig('figs_PSD_V6/EEG_Beta.png',dpi=300,transparent=True)


# Plot EEG & LFP
bootCI = True
fig_eeg, ax_eeg = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(7, 5))
for cind, cond in enumerate(conds):
	
	if bootCI:
		CI_means_EEG = []
		CI_lower_EEG = []
		CI_upper_EEG = []
		for l in range(0,len(eeg[cind][0])):
			x = bs.bootstrap(np.transpose(eeg[cind])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=100)
			CI_means_EEG.append(x.value)
			CI_lower_EEG.append(x.lower_bound)
			CI_upper_EEG.append(x.upper_bound)
	else:
		CI_means_EEG = np.mean(eeg[cind],0)
		CI_lower_EEG = np.mean(eeg[cind],0)-np.std(eeg[cind],0)
		CI_upper_EEG = np.mean(eeg[cind],0)+np.std(eeg[cind],0)
	
	freq0 = 3
	freq1 = 30
	freq2 = 100
	f1 = np.where(freqrawEEG>=freq0)
	f1 = f1[0][0]-1
	f2 = np.where(freqrawEEG>=freq1)
	f2 = f2[0][0]+1
	f3 = np.where(freqrawEEG>=freq2)
	f3 = f3[0][0]
	
	ax_eeg.plot(freqrawEEG[f1:f2], CI_means_EEG[f1:f2], colors_conds[cind])
	ax_eeg.fill_between(freqrawEEG[f1:f2], CI_lower_EEG[f1:f2], CI_upper_EEG[f1:f2], color=colors_conds[cind], alpha=0.6)
	ax_eeg.tick_params(axis='x', which='major', bottom=True)
	ax_eeg.tick_params(axis='y', which='major', left=True)
	if (cind == 0): inset = ax_eeg.inset_axes([.64,.55,.3,.4])
	inset.plot(freqrawEEG[:f3], CI_means_EEG[:f3], colors_conds[cind])
	inset.set_xscale('log')
	inset.set_yscale('log')
	inset.tick_params(axis='x', which='major', bottom=True)
	inset.tick_params(axis='y', which='major', left=True)
	inset.tick_params(axis='x', which='minor', bottom=True)
	inset.tick_params(axis='y', which='minor', left=True)
	inset.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
	inset.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
	inset.fill_between(freqrawEEG[:f3], CI_lower_EEG[:f3], CI_upper_EEG[:f3],color=colors_conds[cind],alpha=0.4)
	inset.set_xticks([1,10,freq2])
	inset.xaxis.set_major_formatter(mticker.ScalarFormatter())
	inset.set_xlim(0.3,freq2)
	ax_eeg.set_xticks([freq0,5,10,15,20,25,freq1])
	ax_eeg.set_xticklabels(['','5','10','15','20','25',str(freq1)])
	ax_eeg.set_xlim(freq0,freq1)
	ax_eeg.set_xlabel('Frequency (Hz)')
	ax_eeg.set_ylabel('EEG PSD (mV'+r'$^{2}$'+'/Hz)')

ax_eeg.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0,r'$\theta$',fontsize=14)
ax_eeg.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0,r'$\alpha$',fontsize=14)
ax_eeg.text(13.25,0,r'$\beta$',fontsize=14)
ylims_2 = ax_eeg.get_ylim()
ax_eeg.plot([thetaband[0],thetaband[0]],ylims_2,c='dimgrey',ls=':')
ax_eeg.plot([alphaband[0],alphaband[0]],ylims_2,c='dimgrey',ls=':')
ax_eeg.plot([alphaband[1],alphaband[1]],ylims_2,c='dimgrey',ls=':')
ax_eeg.set_ylim(ylims_2)
ax_eeg.set_xlim(freq0,freq1)

fig_eeg.savefig('figs_PSD_V6/EEG_PSD.png',bbox_inches='tight',dpi=300,transparent=True)
plt.close()



# Plot aperiodic components
fig_bands, ax_bands = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(7, 5))
for cind in range(len(conds)):
	CI_means = []
	CI_lower = []
	CI_upper = []
	for l in range(0,len(Ls[cind][0])):
		x = bs.bootstrap(np.transpose(Ls[cind])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=100)
		CI_means.append(x.value)
		CI_lower.append(x.lower_bound)
		CI_upper.append(x.upper_bound)
	ax_bands.fill_between(F,CI_lower,CI_upper, color=colors_conds[cind], alpha=0.6)
	ax_bands.plot(F,CI_means,colors_conds[cind],alpha=1,linewidth=2)

inset1 = ax_bands.inset_axes([.57,.7,.4,.22])
xinds = np.arange(0,len(conds))
for cind in range(len(conds)):
	inset1.scatter(xinds[cind]+(np.random.random(len(AUC[cind]))*0.4-0.2),AUC[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = inset1.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_m[cind],AUC_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, inset1, cind-1, cind, '*', xinds, AUC_m, yerr=[sd*2 for sd in AUC_sd], dh=dh1)
	if ((AUC_p0[cind] < p_thresh) & (abs(AUC_cd0[cind]) > c_thresh) & (cind>0)):
		inset1.text(xinds[cind],AUC_m[cind]+AUC_sd[cind]+AUC_sd[cind]/2,'*',c='k',ha='right' if ((AUC_p[cind] < p_thresh) & (abs(AUC_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_p[cind] < p_thresh) & (abs(AUC_cd[cind]) > c_thresh) & (cind>0)):
		inset1.text(xinds[cind],AUC_m[cind]+AUC_sd[cind]+AUC_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_p0[cind] < p_thresh) & (abs(AUC_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

inset1.set_ylabel('AUC')
inset1.set_xticks(xinds)
inset1.set_xticklabels([''])
inset1.grid(False)
inset1.spines['right'].set_visible(False)
inset1.spines['top'].set_visible(False)

inset2 = ax_bands.inset_axes([.57,.4,.4,.22])
xinds = np.arange(0,len(conds))
for cind in range(len(conds)):
	inset2.scatter(xinds[cind]+(np.random.random(len(exponents[cind]))*0.4-0.2),exponents[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = inset2.plot([xinds[cind]-0.25,xinds[cind]+0.25],[exponents_m[cind],exponents_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if exponents_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, inset2, cind-1, cind, '*', xinds, exponents_m, yerr=[sd*2 for sd in exponents_sd], dh=dh1)
	if ((exponents_p0[cind] < p_thresh) & (abs(exponents_cd0[cind]) > c_thresh) & (cind>0)):
		inset2.text(xinds[cind],exponents_m[cind]+exponents_sd[cind]+exponents_sd[cind]/2,'*',c='k',ha='right' if ((exponents_p[cind] < p_thresh) & (abs(exponents_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((exponents_p[cind] < p_thresh) & (abs(exponents_cd[cind]) > c_thresh) & (cind>0)):
		inset2.text(xinds[cind],exponents_m[cind]+exponents_sd[cind]+exponents_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((exponents_p0[cind] < p_thresh) & (abs(exponents_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

inset2.set_ylabel(r'$\chi$')
inset2.set_xticks(xinds)
inset2.set_xticklabels([''])
inset2.grid(False)
inset2.spines['right'].set_visible(False)
inset2.spines['top'].set_visible(False)

ax_bands.tick_params(axis='x', which='major', bottom=True)
ax_bands.tick_params(axis='y', which='major', left=True)
ax_bands.set_xticks([freq0,5,10,15,20,25,freq1])
ax_bands.set_xticklabels(['','5','10','15','20','25',str(freq1)])
ax_bands.set_xlabel('Frequency (Hz)')
ax_bands.set_ylabel('Power (mV'+r'$^{2}$'+'/Hz)')

# ax_bands.set_ylim(ylims_2) # to set to same scale as the original PSD
ylims = ax_bands.get_ylim()
ax_bands.set_ylim(ylims[0],2*10**-14)
ax_bands.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0,r'$\theta$',fontsize=14)
ax_bands.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0,r'$\alpha$',fontsize=14)
ax_bands.text(13.25,0,r'$\beta$',fontsize=14)
ylims = ax_bands.get_ylim()
ax_bands.plot([thetaband[0],thetaband[0]],ylims,c='dimgrey',ls=':')
ax_bands.plot([alphaband[0],alphaband[0]],ylims,c='dimgrey',ls=':')
ax_bands.plot([alphaband[1],alphaband[1]],ylims,c='dimgrey',ls=':')
ax_bands.set_ylim(ylims)
ax_bands.set_xlim(freq0,freq1)

fig_bands.savefig('figs_PSD_V6/aperiodic.png',bbox_inches='tight',dpi=300,transparent=True)
plt.close()

# Plot periodic components
fig_bands, ax_bands = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(7, 5))
for cind in range(len(conds)):
	CI_means = []
	CI_lower = []
	CI_upper = []
	for l in range(0,len(Gns[cind][0])):
		x = bs.bootstrap(np.transpose(Gns[cind])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=100)
		CI_means.append(x.value)
		CI_lower.append(x.lower_bound)
		CI_upper.append(x.upper_bound)
	ax_bands.fill_between(F,CI_lower,CI_upper, color=colors_conds[cind], alpha=0.6)
	ax_bands.plot(F,CI_means,colors_conds[cind],alpha=1,linewidth=2)

inset3 = ax_bands.inset_axes([.57,.7,.4,.22])
xinds = np.arange(0,len(conds))
for cind in range(len(conds)):
	inset3.scatter(xinds[cind]+(np.random.random(len(AUC_t[cind]))*0.4-0.2),AUC_t[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = inset3.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_t_m[cind],AUC_t_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_t_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, inset3, cind-1, cind, '*', xinds, AUC_t_m, yerr=[sd*2 for sd in AUC_t_sd], dh=dh1)
	if ((AUC_t_p0[cind] < p_thresh) & (abs(AUC_t_cd0[cind]) > c_thresh) & (cind>0)):
		inset3.text(xinds[cind],AUC_t_m[cind]+AUC_t_sd[cind]+AUC_t_sd[cind]/2,'*',c='k',ha='right' if ((AUC_t_p[cind] < p_thresh) & (abs(AUC_t_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_t_p[cind] < p_thresh) & (abs(AUC_t_cd[cind]) > c_thresh) & (cind>0)):
		inset3.text(xinds[cind],AUC_t_m[cind]+AUC_t_sd[cind]+AUC_t_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_t_p0[cind] < p_thresh) & (abs(AUC_t_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

inset3.set_ylabel(r'$\theta$')
inset3.set_xticks(xinds)
inset3.set_xticklabels([''])
inset3.grid(False)
inset3.spines['right'].set_visible(False)
inset3.spines['top'].set_visible(False)

inset4 = ax_bands.inset_axes([.57,.4,.4,.22])
xinds = np.arange(0,len(conds))
for cind in range(len(conds)):
	inset4.scatter(xinds[cind]+(np.random.random(len(AUC_b[cind]))*0.4-0.2),AUC_b[cind],s=23,facecolor=colors_conds[cind],edgecolors='face',alpha=0.4)
	ln1, = inset4.plot([xinds[cind]-0.25,xinds[cind]+0.25],[AUC_b_m[cind],AUC_b_m[cind]],'k',alpha=1,linewidth=3)
	ln1.set_solid_capstyle('round')
	
	# if cind>0:
	# 	if AUC_b_p[cind-1] < 0.05:
	# 		barplot_annotate_brackets(fig_bands, inset4, cind-1, cind, '*', xinds, AUC_b_m, yerr=[sd*2 for sd in AUC_b_sd], dh=dh1)
	if ((AUC_b_p0[cind] < p_thresh) & (abs(AUC_b_cd0[cind]) > c_thresh) & (cind>0)):
		inset4.text(xinds[cind],AUC_b_m[cind]+AUC_b_sd[cind]+AUC_b_sd[cind]/2,'*',c='k',ha='right' if ((AUC_b_p[cind] < p_thresh) & (abs(AUC_b_cd[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')
	if ((AUC_b_p[cind] < p_thresh) & (abs(AUC_b_cd[cind]) > c_thresh) & (cind>0)):
		inset4.text(xinds[cind],AUC_b_m[cind]+AUC_b_sd[cind]+AUC_b_sd[cind]/2,'*',c=colors_conds[1],ha='left' if ((AUC_b_p0[cind] < p_thresh) & (abs(AUC_b_cd0[cind]) > c_thresh)) else 'center', va='bottom',fontweight='bold')

inset4.set_ylabel(r'$\beta$')
inset4.set_xticks(xinds)
inset4.set_xticklabels([''])
inset4.grid(False)
inset4.spines['right'].set_visible(False)
inset4.spines['top'].set_visible(False)


ax_bands.tick_params(axis='x', which='major', bottom=True)
ax_bands.tick_params(axis='y', which='major', left=True)
ax_bands.set_xticks([freq0,5,10,15,20,25,freq1])
ax_bands.set_xticklabels(['','5','10','15','20','25',str(freq1)])
ax_bands.set_xlabel('Frequency (Hz)')
ax_bands.set_ylabel(r'log(Power) - log(Power$_{AP}$)')

ylims = ax_bands.get_ylim()
ax_bands.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0.02,r'$\theta$',fontsize=14)
ax_bands.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0.02,r'$\alpha$',fontsize=14)
ax_bands.text(13.25,0.02,r'$\beta$',fontsize=14)
ylims = ax_bands.get_ylim()
ax_bands.plot([thetaband[0],thetaband[0]],ylims,c='dimgrey',ls=':')
ax_bands.plot([alphaband[0],alphaband[0]],ylims,c='dimgrey',ls=':')
ax_bands.plot([alphaband[1],alphaband[1]],ylims,c='dimgrey',ls=':')
ax_bands.set_ylim(ylims)
ax_bands.set_xlim(freq0,freq1)

fig_bands.savefig('figs_PSD_V6/periodic.png',bbox_inches='tight',dpi=300,transparent=True)
plt.close()



# Plot spike PSDs
bootCI = True
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(7, 5),sharex=True,sharey=True) # figsize=(17,11)
inset = ax.inset_axes([.64,.55,.3,.4])
for cind, cond in enumerate(conds):
	
	if bootCI:
		CI_means_PN = []
		CI_lower_PN = []
		CI_upper_PN = []
		for l in range(0,len(spikes_PN[0][0])):
			x = bs.bootstrap(np.transpose(spikes_PN[cind])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=100)
			CI_means_PN.append(x.value)
			CI_lower_PN.append(x.lower_bound)
			CI_upper_PN.append(x.upper_bound)
	else:
		CI_means_PN = np.mean(spikes_PN[cind],0)
		CI_lower_PN = np.mean(spikes_PN[cind],0)-np.std(spikes_PN[cind],0)
		CI_upper_PN = np.mean(spikes_PN[cind],0)+np.std(spikes_PN[cind],0)
	
	f_res = 1/(nperseg*dt/1000)
	f_All = np.arange(0,101,f_res)
	f_PN = np.arange(0,101,f_res)
	
	freq0 = 3
	freq1 = 30
	freq2 = 100
	f1 = np.where(f_All>=freq0)
	f1 = f1[0][0]-1
	f2 = np.where(f_All>=freq1)
	f2 = f2[0][0]+1
	f3 = np.where(f_All>=freq2)
	f3 = f3[0][0]
	
	ax.plot(f_PN[f1:f2], CI_means_PN[f1:f2], colors_conds[cind])
	ax.fill_between(f_PN[f1:f2], CI_lower_PN[f1:f2], CI_upper_PN[f1:f2],color=colors_conds[cind],alpha=0.4)
	ax.tick_params(axis='x', which='major', bottom=True)
	ax.tick_params(axis='y', which='major', left=True)
	inset.plot(f_PN[:f3], CI_means_PN[:f3], colors_conds[cind])
	inset.set_xscale('log')
	inset.set_yscale('log')
	inset.tick_params(axis='x', which='major', bottom=True)
	inset.tick_params(axis='y', which='major', left=True)
	inset.tick_params(axis='x', which='minor', bottom=True)
	inset.tick_params(axis='y', which='minor', left=True)
	inset.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
	inset.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
	inset.fill_between(f_PN[:f3], CI_lower_PN[:f3], CI_upper_PN[:f3],color=colors_conds[cind],alpha=0.4)
	inset.set_xticks([1,10,freq2])
	inset.xaxis.set_major_formatter(mticker.ScalarFormatter())
	inset.set_xlim(0.3,freq2)
	ax.set_xticks([freq0,5,10,15,20,25,freq1])
	ax.set_xticklabels(['','5','10','15','20','25',str(freq1)])
	ax.set_xlim(freq0,freq1)
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Pyr Network PSD (Spikes'+r'$^{2}$'+'/Hz)')

ax.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0,r'$\theta$',fontsize=14)
ax.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0,r'$\alpha$',fontsize=14)
ax.text(13.25,0,r'$\beta$',fontsize=14)
ylims = ax.get_ylim()
ax.plot([thetaband[0],thetaband[0]],ylims,c='dimgrey',ls=':')
ax.plot([alphaband[0],alphaband[0]],ylims,c='dimgrey',ls=':')
ax.plot([alphaband[1],alphaband[1]],ylims,c='dimgrey',ls=':')
ax.set_ylim(ylims)
ax.set_xlim(freq0,freq1)

fig.savefig('figs_PSD_V6/Spikes_PSD_Boot95CI_PN.png',bbox_inches='tight',dpi=300, transparent=True)
plt.close()
