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

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})
fsize = 30

N_seeds = 50
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
tstop = 4500 # ms
response_window = 50 # ms
startsclice = 1000 # ms
endslice = 4000 # ms
stimtime = 4000 # ms
t1 = int(startsclice*(1/dt))
t2 = int(endslice*(1/dt))
tstim = int(stimtime*(1/dt))
t3 = int((stimtime-response_window*4)*(1/dt))
t4 = int((stimtime+response_window*4)*(1/dt))+1
tvec = np.arange(endslice/dt+1)*dt
tvec_stim = np.arange(response_window*8/dt+1)*dt-response_window*4
nperseg = len(tvec)/2

radii = [79000., 80000., 85000., 90000.]
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L23_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

conds = ['healthy','MDD','MDD_a5PAM',
		'healthy_5Hz','MDD_5Hz','MDD_a5PAM_5Hz',
		'healthy_10Hz','MDD_10Hz','MDD_a5PAM_10Hz']
paths = ['Saved_SpikesOnly/' + i + '/' for i in conds]

x_labels = ['Healthy (X1Gou)','MDD (X1Gou)','MDD+a5PAM (X1Gou)',
			'Healthy (X3Gou)','MDD (X3Gou)','MDD+a5PAM (X3Gou)',
			'Healthy (X6Gou)','MDD (X6Gou)','MDD+a5PAM (X6Gou)']
x_labels_types = ['Pyr','SST','PV','VIP']

colors_neurs = ['dimgrey', 'red', 'green', 'orange']
colors_conds = ['tab:gray', 'tab:purple', 'dodgerblue']

broadband = (4,30)
thetaband = (4,8)
alphaband = (8,12)
betaband = (12,30)

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

rates = [[] for _ in conds]
stimrates = [[] for _ in conds]
percentsilent = [[] for _ in conds]
snrs = [[] for _ in conds]

spikes = [[] for _ in conds]
spikes_PN = [[] for _ in conds]
spikes_MN = [[] for _ in conds]
spikes_BN = [[] for _ in conds]
spikes_VN = [[] for _ in conds]

for seed in N_seedsList:
	for cind, path in enumerate(paths):
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		# Load outputs
		temp_s = np.load(path + 'SPIKES_Seed' + str(seed) + '.npy',allow_pickle=True)
		
		# Spikes
		SPIKES = temp_s.item()
		temp_rn = []
		temp_sn = []
		temp_snr = []
		temp_rnS = []
		SPIKE_list = [np.zeros(len(SPIKES['times'][i])) for i in range(len(SPIKES['times']))]
		SPIKE_list_stim = [np.zeros(len(SPIKES['times'][i])) for i in range(len(SPIKES['times']))]
		SPIKE_list_snr = [np.zeros(len(SPIKES['times'][i])) for i in range(len(SPIKES['times']))]
		SILENT_list = np.zeros(len(SPIKE_list))
		for i in range(0,len(SPIKES['times'])):
			for j in range(0,len(SPIKES['times'][i])):
				
				# baseline
				scount = SPIKES['times'][i][j][(SPIKES['times'][i][j]>startsclice) & (SPIKES['times'][i][j]<=stimtime)]
				Hz = (scount.size)/((int(stimtime)-startsclice)/1000)
				SPIKE_list[i][j] = Hz
				
				# stimulation
				scount_stim = SPIKES['times'][i][j][(SPIKES['times'][i][j]>stimtime+5) & (SPIKES['times'][i][j]<=stimtime+response_window+5)]
				Hz_stim = (scount_stim.size)/(response_window/1000)
				SPIKE_list_stim[i][j] = Hz_stim
				
				# snr
				SPIKE_list_snr[i][j] = Hz_stim/Hz if Hz>0.2 else np.nan
				
				# percent silent
				if Hz <= rate_threshold:
					SILENT_list[i] += 1
			
			temp_rn.append(np.mean(SPIKE_list[i]))#[SPIKE_list[i]>rate_threshold]))
			temp_sn.append(np.mean(SPIKE_list_stim[i]))#[SPIKE_list[i]>rate_threshold]))
			temp_snr.append(np.nanmean(SPIKE_list_snr[i]))#[SPIKE_list[i]>rate_threshold]))
			temp_rnS.append((SILENT_list[i]/len(SPIKES['times'][i]))*100)
		
		rates[cind].append(temp_rn)
		stimrates[cind].append(temp_sn)
		snrs[cind].append(temp_snr)
		percentsilent[cind].append(temp_rnS)

# Plot Rates
x = [[] for _ in x_labels_types]
ntypes = len(x_labels_types)
nconds = len(conds)
for i in range(0,ntypes):
	x[i] = np.linspace(0,nconds-1,nconds) + (nconds+1)*i

x = np.transpose(x)
x_PN = np.linspace(0,nconds-1,nconds)
x_types = np.linspace(0,ntypes-1,ntypes)

df = pd.DataFrame(columns=["Comparison",
			"Metric",
			"Group1 Mean",
			"Group1 SD",
			"Group2 Mean",
			"Group2 SD",
			"t-stat",
			"p-value",
			"Cohen's d"])
df_SST = pd.DataFrame(columns=["Comparison",
			"Metric",
			"Group1 Mean",
			"Group1 SD",
			"Group2 Mean",
			"Group2 SD",
			"t-stat",
			"p-value",
			"Cohen's d"])
df_PV = pd.DataFrame(columns=["Comparison",
			"Metric",
			"Group1 Mean",
			"Group1 SD",
			"Group2 Mean",
			"Group2 SD",
			"t-stat",
			"p-value",
			"Cohen's d"])
df_VIP = pd.DataFrame(columns=["Comparison",
			"Metric",
			"Group1 Mean",
			"Group1 SD",
			"Group2 Mean",
			"Group2 SD",
			"t-stat",
			"p-value",
			"Cohen's d"])

# Stats

p_thresh = 0.05
c_thresh = 1.

tstats_base_h = [[[] for _ in x_labels_types] for _ in conds]
tstats_stim_h = [[[] for _ in x_labels_types] for _ in conds]
tstats_snr_h = [[[] for _ in x_labels_types] for _ in conds]
tstats_silent_h = [[[] for _ in x_labels_types] for _ in conds]

pvals_base_h = [[[] for _ in x_labels_types] for _ in conds]
pvals_stim_h = [[[] for _ in x_labels_types] for _ in conds]
pvals_snr_h = [[[] for _ in x_labels_types] for _ in conds]
pvals_silent_h = [[[] for _ in x_labels_types] for _ in conds]

cd_base_h = [[[] for _ in x_labels_types] for _ in conds]
cd_stim_h = [[[] for _ in x_labels_types] for _ in conds]
cd_snr_h = [[[] for _ in x_labels_types] for _ in conds]
cd_silent_h = [[[] for _ in x_labels_types] for _ in conds]

tstats_base_d = [[[] for _ in x_labels_types] for _ in conds]
tstats_stim_d = [[[] for _ in x_labels_types] for _ in conds]
tstats_snr_d = [[[] for _ in x_labels_types] for _ in conds]
tstats_silent_d = [[[] for _ in x_labels_types] for _ in conds]

pvals_base_d = [[[] for _ in x_labels_types] for _ in conds]
pvals_stim_d = [[[] for _ in x_labels_types] for _ in conds]
pvals_snr_d = [[[] for _ in x_labels_types] for _ in conds]
pvals_silent_d = [[[] for _ in x_labels_types] for _ in conds]

cd_base_d = [[[] for _ in x_labels_types] for _ in conds]
cd_stim_d = [[[] for _ in x_labels_types] for _ in conds]
cd_snr_d = [[[] for _ in x_labels_types] for _ in conds]
cd_silent_d = [[[] for _ in x_labels_types] for _ in conds]

m0 = [[[] for _ in x_labels_types] for _ in conds]
sd0 = [[[] for _ in x_labels_types] for _ in conds]
m1 = [[[] for _ in x_labels_types] for _ in conds]
sd1 = [[[] for _ in x_labels_types] for _ in conds]
m2 = [[[] for _ in x_labels_types] for _ in conds]
sd2 = [[[] for _ in x_labels_types] for _ in conds]

tstat_h = [[[] for _ in x_labels_types] for _ in conds]
p_h = [[[] for _ in x_labels_types] for _ in conds]
cd_h = [[[] for _ in x_labels_types] for _ in conds]

tstat_d = [[[] for _ in x_labels_types] for _ in conds]
p_d = [[[] for _ in x_labels_types] for _ in conds]
cd_d = [[[] for _ in x_labels_types] for _ in conds]

metrics = ["Baseline Rate","Response Rate", "SNR", "% Silent"]
for cind, cond in enumerate(conds):
	if cind <= 2:
		cind_healthy = 0 # healthy
		cind_mdd = 1
	elif ((cind > 2) & (cind <= 5)):
		cind_healthy = 3
		cind_mdd = 4
	elif cind > 5:
		cind_healthy = 6
		cind_mdd = 7
	
	# vs healthy
	for tind,_ in enumerate(x_labels_types):
		tstat_base_temp, pval_base_temp = st.ttest_rel(np.transpose(rates[cind_healthy])[tind],np.transpose(rates[cind])[tind])
		tstat_stim_temp, pval_stim_temp = st.ttest_rel(np.transpose(stimrates[cind_healthy])[tind],np.transpose(stimrates[cind])[tind])
		tstat_snr_temp, pval_snr_temp = st.ttest_rel(np.transpose(snrs[cind_healthy])[tind],np.transpose(snrs[cind])[tind])
		tstat_silent_temp, pval_silent_temp = st.ttest_rel(np.transpose(percentsilent[cind_healthy])[tind],np.transpose(percentsilent[cind])[tind])
		cd_base_temp = cohen_d(np.transpose(rates[cind_healthy])[tind],np.transpose(rates[cind])[tind])
		cd_stim_temp = cohen_d(np.transpose(stimrates[cind_healthy])[tind],np.transpose(stimrates[cind])[tind])
		cd_snr_temp = cohen_d(np.transpose(snrs[cind_healthy])[tind],np.transpose(snrs[cind])[tind])
		cd_silent_temp = cohen_d(np.transpose(percentsilent[cind_healthy])[tind],np.transpose(percentsilent[cind])[tind])
		
		tstats_base_h[cind][tind].append(tstat_base_temp)
		tstats_stim_h[cind][tind].append(tstat_stim_temp)
		tstats_snr_h[cind][tind].append(tstat_snr_temp)
		tstats_silent_h[cind][tind].append(tstat_silent_temp)

		pvals_base_h[cind][tind].append(pval_base_temp)
		pvals_stim_h[cind][tind].append(pval_stim_temp)
		pvals_snr_h[cind][tind].append(pval_snr_temp)
		pvals_silent_h[cind][tind].append(pval_silent_temp)

		cd_base_h[cind][tind].append(cd_base_temp)
		cd_stim_h[cind][tind].append(cd_stim_temp)
		cd_snr_h[cind][tind].append(cd_snr_temp)
		cd_silent_h[cind][tind].append(cd_silent_temp)
		
		tstat_base_temp, pval_base_temp = st.ttest_rel(np.transpose(rates[cind_mdd])[tind],np.transpose(rates[cind])[tind])
		tstat_stim_temp, pval_stim_temp = st.ttest_rel(np.transpose(stimrates[cind_mdd])[tind],np.transpose(stimrates[cind])[tind])
		tstat_snr_temp, pval_snr_temp = st.ttest_rel(np.transpose(snrs[cind_mdd])[tind],np.transpose(snrs[cind])[tind])
		tstat_silent_temp, pval_silent_temp = st.ttest_rel(np.transpose(percentsilent[cind_mdd])[tind],np.transpose(percentsilent[cind])[tind])
		cd_base_temp = cohen_d(np.transpose(rates[cind_mdd])[tind],np.transpose(rates[cind])[tind])
		cd_stim_temp = cohen_d(np.transpose(stimrates[cind_mdd])[tind],np.transpose(stimrates[cind])[tind])
		cd_snr_temp = cohen_d(np.transpose(snrs[cind_mdd])[tind],np.transpose(snrs[cind])[tind])
		cd_silent_temp = cohen_d(np.transpose(percentsilent[cind_mdd])[tind],np.transpose(percentsilent[cind])[tind])
		
		tstats_base_d[cind][tind].append(tstat_base_temp)
		tstats_stim_d[cind][tind].append(tstat_stim_temp)
		tstats_snr_d[cind][tind].append(tstat_snr_temp)
		tstats_silent_d[cind][tind].append(tstat_silent_temp)

		pvals_base_d[cind][tind].append(pval_base_temp)
		pvals_stim_d[cind][tind].append(pval_stim_temp)
		pvals_snr_d[cind][tind].append(pval_snr_temp)
		pvals_silent_d[cind][tind].append(pval_silent_temp)

		cd_base_d[cind][tind].append(cd_base_temp)
		cd_stim_d[cind][tind].append(cd_stim_temp)
		cd_snr_d[cind][tind].append(cd_snr_temp)
		cd_silent_d[cind][tind].append(cd_silent_temp)
		
		m0[cind][tind].extend([np.mean(rates[cind_healthy],0)[tind],np.mean(stimrates[cind_healthy],0)[tind],np.mean(snrs[cind_healthy],0)[tind],np.mean(percentsilent[cind_healthy],0)[tind]])
		sd0[cind][tind].extend([np.std(rates[cind_healthy],0)[tind],np.std(stimrates[cind_healthy],0)[tind],np.std(snrs[cind_healthy],0)[tind],np.std(percentsilent[cind_healthy],0)[tind]])
		m1[cind][tind].extend([np.mean(rates[cind_mdd],0)[tind],np.mean(stimrates[cind_mdd],0)[tind],np.mean(snrs[cind_mdd],0)[tind],np.mean(percentsilent[cind_mdd],0)[tind]])
		sd1[cind][tind].extend([np.std(rates[cind_mdd],0)[tind],np.std(stimrates[cind_mdd],0)[tind],np.std(snrs[cind_mdd],0)[tind],np.std(percentsilent[cind_mdd],0)[tind]])
		m2[cind][tind].extend([np.mean(rates[cind],0)[tind],np.mean(stimrates[cind],0)[tind],np.mean(snrs[cind],0)[tind],np.mean(percentsilent[cind],0)[tind]])
		sd2[cind][tind].extend([np.std(rates[cind],0)[tind],np.std(stimrates[cind],0)[tind],np.std(snrs[cind],0)[tind],np.std(percentsilent[cind],0)[tind]])
		
		tstat_h[cind][tind].extend([tstats_base_h[cind][tind],tstats_stim_h[cind][tind],tstats_snr_h[cind][tind],tstats_silent_h[cind][tind]])
		p_h[cind][tind].extend([pvals_base_h[cind][tind],pvals_stim_h[cind][tind],pvals_snr_h[cind][tind],pvals_silent_h[cind][tind]])
		cd_h[cind][tind].extend([cd_base_h[cind][tind],cd_stim_h[cind][tind],cd_snr_h[cind][tind],cd_silent_h[cind][tind]])
		
		tstat_d[cind][tind].extend([tstats_base_d[cind][tind],tstats_stim_d[cind][tind],tstats_snr_d[cind][tind],tstats_silent_d[cind][tind]])
		p_d[cind][tind].extend([pvals_base_d[cind][tind],pvals_stim_d[cind][tind],pvals_snr_d[cind][tind],pvals_silent_d[cind][tind]])
		cd_d[cind][tind].extend([cd_base_d[cind][tind],cd_stim_d[cind][tind],cd_snr_d[cind][tind],cd_silent_d[cind][tind]])
	
	# vs Healthy
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels[cind_healthy] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0[cind][0][dfi],
					"Group1 SD" : sd0[cind][0][dfi],
					"Group2 Mean" : m2[cind][0][dfi],
					"Group2 SD" : sd2[cind][0][dfi],
					"t-stat" : tstat_h[cind][0][dfi],
					"p-value" : p_h[cind][0][dfi],
					"Cohen's d" : cd_h[cind][0][dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_SST = df_SST.append({"Comparison" : x_labels[cind_healthy] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0[cind][1][dfi],
					"Group1 SD" : sd0[cind][1][dfi],
					"Group2 Mean" : m2[cind][1][dfi],
					"Group2 SD" : sd2[cind][1][dfi],
					"t-stat" : tstat_h[cind][1][dfi],
					"p-value" : p_h[cind][1][dfi],
					"Cohen's d" : cd_h[cind][1][dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_PV = df_PV.append({"Comparison" : x_labels[cind_healthy] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0[cind][2][dfi],
					"Group1 SD" : sd0[cind][2][dfi],
					"Group2 Mean" : m2[cind][2][dfi],
					"Group2 SD" : sd2[cind][2][dfi],
					"t-stat" : tstat_h[cind][2][dfi],
					"p-value" : p_h[cind][2][dfi],
					"Cohen's d" : cd_h[cind][2][dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_VIP = df_VIP.append({"Comparison" : x_labels[cind_healthy] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0[cind][3][dfi],
					"Group1 SD" : sd0[cind][3][dfi],
					"Group2 Mean" : m2[cind][3][dfi],
					"Group2 SD" : sd2[cind][3][dfi],
					"t-stat" : tstat_h[cind][3][dfi],
					"p-value" : p_h[cind][3][dfi],
					"Cohen's d" : cd_h[cind][3][dfi]},
					ignore_index = True)

	# vs MDD
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels[cind_mdd] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1[cind][0][dfi],
					"Group1 SD" : sd1[cind][0][dfi],
					"Group2 Mean" : m2[cind][0][dfi],
					"Group2 SD" : sd2[cind][0][dfi],
					"t-stat" : tstat_d[cind][0][dfi],
					"p-value" : p_d[cind][0][dfi],
					"Cohen's d" : cd_d[cind][0][dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_SST = df_SST.append({"Comparison" : x_labels[cind_mdd] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1[cind][1][dfi],
					"Group1 SD" : sd1[cind][1][dfi],
					"Group2 Mean" : m2[cind][1][dfi],
					"Group2 SD" : sd2[cind][1][dfi],
					"t-stat" : tstat_d[cind][1][dfi],
					"p-value" : p_d[cind][1][dfi],
					"Cohen's d" : cd_d[cind][1][dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_PV = df_PV.append({"Comparison" : x_labels[cind_mdd] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1[cind][2][dfi],
					"Group1 SD" : sd1[cind][2][dfi],
					"Group2 Mean" : m2[cind][2][dfi],
					"Group2 SD" : sd2[cind][2][dfi],
					"t-stat" : tstat_d[cind][2][dfi],
					"p-value" : p_d[cind][2][dfi],
					"Cohen's d" : cd_d[cind][2][dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_VIP = df_VIP.append({"Comparison" : x_labels[cind_mdd] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1[cind][3][dfi],
					"Group1 SD" : sd1[cind][3][dfi],
					"Group2 Mean" : m2[cind][3][dfi],
					"Group2 SD" : sd2[cind][3][dfi],
					"t-stat" : tstat_d[cind][3][dfi],
					"p-value" : p_d[cind][3][dfi],
					"Cohen's d" : cd_d[cind][3][dfi]},
					ignore_index = True)

df.to_csv('figsV1_backgroundinput/stats_Rates.csv')
df_SST.to_csv('figsV1_backgroundinput/stats_Rates_SST.csv')
df_PV.to_csv('figsV1_backgroundinput/stats_Rates_PV.csv')
df_VIP.to_csv('figsV1_backgroundinput/stats_Rates_VIP.csv')

def plot_metric_across_conditions(data,ylabel,filename,sig_ind=0,multiplier=4):
	x = [0,3,6]
	shift=[-0.8,0,0.8]
	xtick_labels = ['1X','3X','6X']
	
	fig_Pyr, ax_Pyr = plt.subplots(figsize=(6, 9))
	fig_SST, ax_SST = plt.subplots(figsize=(6, 9))
	fig_PV, ax_PV = plt.subplots(figsize=(6, 9))
	fig_VIP, ax_VIP = plt.subplots(figsize=(6, 9))
	
	m_Pyr = np.transpose(np.reshape([np.mean(data[cind],0)[0] for cind,_ in enumerate(conds)], (3,3)))
	m_SST = np.transpose(np.reshape([np.mean(data[cind],0)[1] for cind,_ in enumerate(conds)], (3,3)))
	m_PV = np.transpose(np.reshape([np.mean(data[cind],0)[2] for cind,_ in enumerate(conds)], (3,3)))
	m_VIP = np.transpose(np.reshape([np.mean(data[cind],0)[3] for cind,_ in enumerate(conds)], (3,3)))
	sd_Pyr = np.transpose(np.reshape([np.std(data[cind],0)[0] for cind,_ in enumerate(conds)], (3,3)))
	sd_SST = np.transpose(np.reshape([np.std(data[cind],0)[1] for cind,_ in enumerate(conds)], (3,3)))
	sd_PV = np.transpose(np.reshape([np.std(data[cind],0)[2] for cind,_ in enumerate(conds)], (3,3)))
	sd_VIP = np.transpose(np.reshape([np.std(data[cind],0)[3] for cind,_ in enumerate(conds)], (3,3)))
	
	for cind, (y,err) in enumerate(zip(m_Pyr,sd_Pyr)):
		ax_Pyr.bar([x0+shift[cind] for x0 in x],y,yerr=err,capsize=8,width=0.8,color=colors_conds[cind],edgecolor='k',ecolor='black',linewidth=1,error_kw={'elinewidth':3,'markeredgewidth':3})
	for cind, (y,err) in enumerate(zip(m_SST,sd_SST)):
		ax_SST.bar([x0+shift[cind] for x0 in x],y,yerr=err,capsize=8,width=0.8,color=colors_conds[cind],edgecolor='k',ecolor='black',linewidth=1,error_kw={'elinewidth':3,'markeredgewidth':3})
	for cind, (y,err) in enumerate(zip(m_PV,sd_PV)):
		ax_PV.bar([x0+shift[cind] for x0 in x],y,yerr=err,capsize=8,width=0.8,color=colors_conds[cind],edgecolor='k',ecolor='black',linewidth=1,error_kw={'elinewidth':3,'markeredgewidth':3})
	for cind, (y,err) in enumerate(zip(m_VIP,sd_VIP)):
		ax_VIP.bar([x0+shift[cind] for x0 in x],y,yerr=err,capsize=8,width=0.8,color=colors_conds[cind],edgecolor='k',ecolor='black',linewidth=1,error_kw={'elinewidth':3,'markeredgewidth':3})
	
	for cind,_ in enumerate(conds):
		if cind <= 2:
			x_pos = x[0]+shift[cind]
			y_pos_pyr = np.mean(data[cind],0)[0]+np.max([np.std(d,0)[0] for d in data])*multiplier
			y_pos_sst = np.mean(data[cind],0)[1]+np.max([np.std(d,0)[1] for d in data])*multiplier
			y_pos_pv = np.mean(data[cind],0)[2]+np.max([np.std(d,0)[2] for d in data])*multiplier
			y_pos_vip = np.mean(data[cind],0)[3]+np.max([np.std(d,0)[3] for d in data])*multiplier
		elif ((cind > 2) & (cind <= 5)):
			if cind == 3: cind_shift = 0
			if cind == 4: cind_shift = 1
			if cind == 5: cind_shift = 2
			x_pos = x[1]+shift[cind_shift]
			y_pos_pyr = np.mean(data[cind],0)[0]+np.max([np.std(d,0)[0] for d in data])*multiplier
			y_pos_sst = np.mean(data[cind],0)[1]+np.max([np.std(d,0)[1] for d in data])*multiplier
			y_pos_pv = np.mean(data[cind],0)[2]+np.max([np.std(d,0)[2] for d in data])*multiplier
			y_pos_vip = np.mean(data[cind],0)[3]+np.max([np.std(d,0)[3] for d in data])*multiplier
		elif cind > 5:
			if cind == 6: cind_shift = 0
			if cind == 7: cind_shift = 1
			if cind == 8: cind_shift = 2
			x_pos = x[2]+shift[cind_shift]
			y_pos_pyr = np.mean(data[cind],0)[0]+np.max([np.std(d,0)[0] for d in data])*multiplier
			y_pos_sst = np.mean(data[cind],0)[1]+np.max([np.std(d,0)[1] for d in data])*multiplier
			y_pos_pv = np.mean(data[cind],0)[2]+np.max([np.std(d,0)[2] for d in data])*multiplier
			y_pos_vip = np.mean(data[cind],0)[3]+np.max([np.std(d,0)[3] for d in data])*multiplier
		
		ph = p_h[cind][0][sig_ind][0]
		ch = cd_h[cind][0][sig_ind][0]
		pd = p_d[cind][0][sig_ind][0]
		cd = cd_d[cind][0][sig_ind][0]
		if ((ph < p_thresh) & (abs(ch) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_Pyr.text(x_pos,y_pos_pyr,'*',c='k',va='bottom' if ((pd < p_thresh) & (abs(cd) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pd < p_thresh) & (abs(cd) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_Pyr.text(x_pos,y_pos_pyr,'*',c=colors_conds[1],va='top' if ((ph < p_thresh) & (abs(ch) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ph = p_h[cind][1][sig_ind][0]
		ch = cd_h[cind][1][sig_ind][0]
		pd = p_d[cind][1][sig_ind][0]
		cd = cd_d[cind][1][sig_ind][0]
		if ((ph < p_thresh) & (abs(ch) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_SST.text(x_pos,y_pos_sst,'*',c='k',va='bottom' if ((pd < p_thresh) & (abs(cd) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pd < p_thresh) & (abs(cd) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_SST.text(x_pos,y_pos_sst,'*',c=colors_conds[1],va='top' if ((ph < p_thresh) & (abs(ch) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ph = p_h[cind][2][sig_ind][0]
		ch = cd_h[cind][2][sig_ind][0]
		pd = p_d[cind][2][sig_ind][0]
		cd = cd_d[cind][2][sig_ind][0]
		if ((ph < p_thresh) & (abs(ch) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_PV.text(x_pos,y_pos_pv,'*',c='k',va='bottom' if ((pd < p_thresh) & (abs(cd) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pd < p_thresh) & (abs(cd) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_PV.text(x_pos,y_pos_pv,'*',c=colors_conds[1],va='top' if ((ph < p_thresh) & (abs(ch) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ph = p_h[cind][3][sig_ind][0]
		ch = cd_h[cind][3][sig_ind][0]
		pd = p_d[cind][3][sig_ind][0]
		cd = cd_d[cind][3][sig_ind][0]
		if ((ph < p_thresh) & (abs(ch) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_VIP.text(x_pos,y_pos_vip,'*',c='k',va='bottom' if ((pd < p_thresh) & (abs(cd) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pd < p_thresh) & (abs(cd) > c_thresh) & (cind!=0) & (cind!=3) & (cind!=6)):
			ax_VIP.text(x_pos,y_pos_vip,'*',c=colors_conds[1],va='top' if ((ph < p_thresh) & (abs(ch) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	
	ax_Pyr.set_ylabel('Pyr '+ylabel)
	ax_SST.set_ylabel('SST '+ylabel)
	ax_PV.set_ylabel('PV '+ylabel)
	ax_VIP.set_ylabel('VIP '+ylabel)
	ax_Pyr.set_xlabel(r'$G_{OU}$')
	ax_SST.set_xlabel(r'$G_{OU}$')
	ax_PV.set_xlabel(r'$G_{OU}$')
	ax_VIP.set_xlabel(r'$G_{OU}$')
	ax_Pyr.set_xticks(x)
	ax_SST.set_xticks(x)
	ax_PV.set_xticks(x)
	ax_VIP.set_xticks(x)
	ax_Pyr.set_xlim(x[0]-1.5,x[-1]+1.5)
	ax_SST.set_xlim(x[0]-1.5,x[-1]+1.5)
	ax_PV.set_xlim(x[0]-1.5,x[-1]+1.5)
	ax_VIP.set_xlim(x[0]-1.5,x[-1]+1.5)
	ax_Pyr.set_xticklabels(xtick_labels)
	ax_SST.set_xticklabels(xtick_labels)
	ax_PV.set_xticklabels(xtick_labels)
	ax_VIP.set_xticklabels(xtick_labels)
	ax_Pyr.grid(False)
	ax_SST.grid(False)
	ax_PV.grid(False)
	ax_VIP.grid(False)
	ax_Pyr.spines['right'].set_visible(False)
	ax_SST.spines['right'].set_visible(False)
	ax_PV.spines['right'].set_visible(False)
	ax_VIP.spines['right'].set_visible(False)
	ax_Pyr.spines['top'].set_visible(False)
	ax_SST.spines['top'].set_visible(False)
	ax_PV.spines['top'].set_visible(False)
	ax_VIP.spines['top'].set_visible(False)
	fig_Pyr.tight_layout()
	fig_SST.tight_layout()
	fig_PV.tight_layout()
	fig_VIP.tight_layout()
	fig_Pyr.savefig('figsV1_backgroundinput/Pyr_'+filename+'.png',dpi=300,transparent=True)
	fig_SST.savefig('figsV1_backgroundinput/SST_'+filename+'.png',dpi=300,transparent=True)
	fig_PV.savefig('figsV1_backgroundinput/PV_'+filename+'.png',dpi=300,transparent=True)
	fig_VIP.savefig('figsV1_backgroundinput/VIP_'+filename+'.png',dpi=300,transparent=True)

plot_metric_across_conditions(rates,'Baseline Rate (Hz)','base_rate', sig_ind=0, multiplier=6)
plot_metric_across_conditions(stimrates,'Response Rate (Hz)','stim_rate', sig_ind=1, multiplier=4)
plot_metric_across_conditions(snrs,'SNR','snr', sig_ind=2, multiplier=2)
plot_metric_across_conditions(percentsilent,'% Silent','silent', sig_ind=3, multiplier=3)
