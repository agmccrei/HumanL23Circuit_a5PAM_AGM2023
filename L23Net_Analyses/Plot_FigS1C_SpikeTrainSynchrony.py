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

N_seeds = 20 # 200 # 200 seeds will take forever to run...
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

lag_window = 50
lag_window_inds = int((lag_window/2)/dt) # ms converted to index

conds = ['healthy','MDD','MDD_a5PAM']
paths = ['Saved_SpikesOnly/' + i + '/' for i in conds]

x_labels = ['Healthy','MDD','MDD\n'+r'$\alpha$'+'5-PAM']
x_labels_types = ['Pyr','SST','PV','VIP']

colors_neurs = ['dimgrey', 'red', 'green', 'orange']
colors_conds = ['tab:gray', 'tab:purple', 'dodgerblue']

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

xcorr_0lag = [[] for _ in conds]

for seed in N_seedsList:
	for cind, path in enumerate(paths):
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		# Load outputs
		temp_s = np.load(path + 'SPIKES_Seed' + str(seed) + '.npy',allow_pickle=True)
		
		# Spikes
		spike_times_Pyr = [x[(x>startsclice) & (x<stimtime+dt)] for _,x in sorted(zip(temp_s.item()['gids'][0],temp_s.item()['times'][0]))]
		spike_vec_Pyr = [np.histogram(x,bins=np.arange(startsclice,stimtime+dt,dt))[0] for _,x in sorted(zip(temp_s.item()['gids'][0],temp_s.item()['times'][0]))]
		
		# Check equivalency (sanity check)
		num_spikes_times = [len(x) for x in spike_times_Pyr]
		num_spikes_vec = [np.sum(x) for x in spike_vec_Pyr]
		if num_spikes_times == num_spikes_vec: print('lists are equivalent')
		else: print('check code - something is not right')
		
		temp_xcorr_metric = []
		for i1, s1 in enumerate(spike_vec_Pyr):
			if i1 % 100 == 0: print('Analyzing Spike Train '+str(i1))
			for i2, s2 in enumerate(spike_vec_Pyr):
				# mode='valid' should only give the value for overlapping signals
				xcorr = np.correlate(s1,s2,mode='valid')
				
				# Find sum of xcorr between +/- 50 ms
				temp_xcorr_metric.extend(xcorr.tolist())
		
		xcorr_0lag[cind].append(np.mean(temp_xcorr_metric))

np.save('figs_spiketrainsynchrony_V1/xcorrs_seeds'+str(N_seedsList[0])+'to'+str(N_seedsList[-1]),xcorr_0lag)

# Plot Rates
nconds = len(conds)

x_PN = np.linspace(0,nconds-1,nconds)

df = pd.DataFrame(columns=["Comparison",
			"Metric",
			"Group1 Mean",
			"Group1 SD",
			"Group2 Mean",
			"Group2 SD",
			"t-stat",
			"p-value",
			"Cohen's d"])

p_thresh = 0.05
c_thresh = 1.

fig_xcorr_PN, ax_xcorr_PN = plt.subplots(figsize=(8, 3.5))
for cind, cond in enumerate(conds):
	meanXcorrs = np.mean(xcorr_0lag[cind])
	stdevXcorrs = np.std(xcorr_0lag[cind])
	
	# vs healthy
	tstat_PN_0, pval_PN_0 = st.ttest_rel(np.transpose(xcorr_0lag[0]),np.transpose(xcorr_0lag[cind]))
	cd_PN_0 = cohen_d(np.transpose(xcorr_0lag[0]),np.transpose(xcorr_0lag[cind]))
	
	# vs MDD
	tstat_PN, pval_PN = st.ttest_rel(np.transpose(xcorr_0lag[1]),np.transpose(xcorr_0lag[cind]))
	cd_PN = cohen_d(np.transpose(xcorr_0lag[1]),np.transpose(xcorr_0lag[cind]))
	
	metrics = ["Cross-Correlation"]
	m0 = [np.mean(xcorr_0lag[0])]
	sd0 = [np.std(xcorr_0lag[0])]
	m1 = [np.mean(xcorr_0lag[1])]
	sd1 = [np.std(xcorr_0lag[1])]
	m2 = [np.mean(xcorr_0lag[cind])]
	sd2 = [np.std(xcorr_0lag[cind])]
	
	tstat = [tstat_PN]
	p = [pval_PN]
	cd = [cd_PN]
	
	tstat_0 = [tstat_PN_0]
	p_0 = [pval_PN_0]
	cd_0 = [cd_PN_0]
	
	# vs Healthy
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels[0] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0[dfi],
					"Group1 SD" : sd0[dfi],
					"Group2 Mean" : m2[dfi],
					"Group2 SD" : sd2[dfi],
					"t-stat" : tstat_0[dfi],
					"p-value" : p_0[dfi],
					"Cohen's d" : cd_0[dfi]},
					ignore_index = True)
	
	# vs MDD
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels[1] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1[dfi],
					"Group1 SD" : sd1[dfi],
					"Group2 Mean" : m2[dfi],
					"Group2 SD" : sd2[dfi],
					"t-stat" : tstat[dfi],
					"p-value" : p[dfi],
					"Cohen's d" : cd[dfi]},
					ignore_index = True)
	
	# plot
	ax_xcorr_PN.bar(x_PN[cind],height=meanXcorrs,
		   yerr=stdevXcorrs,    # error bars
		   capsize=12, # error bar cap width in points
		   width=0.7,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		   )
	if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh) & (cind>0)):
		ax_xcorr_PN.text(x_PN[cind],meanXcorrs+stdevXcorrs+0.001,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh) & (cind>0)):
		ax_xcorr_PN.text(x_PN[cind],meanXcorrs+stdevXcorrs+0.001,'*',c=colors_conds[1],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	

df.to_csv('figs_spiketrainsynchrony_V1/stats_synchrony.csv')

ax_xcorr_PN.set_ylabel('Spike Train\nCross-Correlation')
ax_xcorr_PN.set_xticks(x_PN)
ax_xcorr_PN.set_xlim(-0.6,2.6)
ax_xcorr_PN.set_xticklabels(x_labels)
ax_xcorr_PN.grid(False)
ax_xcorr_PN.spines['right'].set_visible(False)
ax_xcorr_PN.spines['top'].set_visible(False)
fig_xcorr_PN.tight_layout()
fig_xcorr_PN.savefig('figs_spiketrainsynchrony_V1/Synchrony_PN.png',dpi=300,transparent=True)
