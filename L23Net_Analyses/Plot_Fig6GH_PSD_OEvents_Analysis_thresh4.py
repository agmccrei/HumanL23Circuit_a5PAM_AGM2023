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
sys.path.append("OEvent-master")
from oevent import *


font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}
fsize = 30

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

colors_conds = ['tab:gray', 'tab:purple', 'dodgerblue']

# Sim params
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
sampling_rate = (1/dt)*1000
#nperseg = 140000 # len(tvec[t1:t2])/2

# EEG params
radii = [79000., 80000., 85000., 90000.]
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L23_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

# folder names
conds = ['healthy_long','MDD_long','MDD_a5PAM_long']
paths = [i + '/HL23net_Drug/Circuit_output/' for i in conds]

# parameters for OEvent
medthresh = 4.0 # median threshold
sampr = sampling_rate #11000 # sampling rate
winsz = int((endslice-startslice)/1000) # (seconds) window size
freqmin = 1 #0.25 # minimum frequency (Hz)
freqmax = 100 #250.0  # maximum frequency (Hz)
freqstep = 0.5 #0.25 # frequency step (Hz)
overlapth = 0.5 # overlapping bounding box threshold (threshold for merging event bounding boxes)
chan = 0 # which channel to use for event analysis
lchan = [chan] # list of channels to use for event analysis
MUA = None # multiunit activity; not required

bands = ['delta','theta','alpha','beta','lgamma','gamma','hgamma']
band_labels = [r'$\delta$',r'$\theta$',r'$\alpha$',r'$\beta$','l'+r'$\gamma$',r'$\gamma$','h'+r'$\gamma$']

OEvent_counts = [[[] for _ in conds] for _ in bands]
OEvent_dur = [[[] for _ in conds] for _ in bands]
OEvent_ncycles = [[[] for _ in conds] for _ in bands]
OEvent_avgpow = [[[] for _ in conds] for _ in bands]
OEvent_peakF = [[[] for _ in conds] for _ in bands]
OEvent_Fspan = [[[] for _ in conds] for _ in bands]
OEvent_WavePeakVal = [[[] for _ in conds] for _ in bands]
OEvent_WaveletPeakVal = [[[] for _ in conds] for _ in bands]
OEvent_WaveH = [[[] for _ in conds] for _ in bands]

OEvent_codelta = [[[] for _ in conds] for _ in bands]
OEvent_cotheta = [[[] for _ in conds] for _ in bands]
OEvent_coalpha = [[[] for _ in conds] for _ in bands]
OEvent_cobeta = [[[] for _ in conds] for _ in bands]
OEvent_colgamma = [[[] for _ in conds] for _ in bands]
OEvent_cogamma = [[[] for _ in conds] for _ in bands]
OEvent_cohgamma = [[[] for _ in conds] for _ in bands]

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

p_thresh = 0.05
c_thresh = 1.

for cind, path in enumerate(paths):
	for seed in N_seedsList:
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		df = pd.read_csv('Saved_OEvents_thresh4/'+conds[cind]+'_seed'+str(seed)+'_OEvents.csv')
		for bind,b in enumerate(bands):
			df_band = df[(df.band == b)]
			
			OEvent_counts[bind][cind].append(df_band.shape[0])
			OEvent_dur[bind][cind].append(np.mean(df_band.dur))
			OEvent_ncycles[bind][cind].append(np.mean(df_band.ncycle))
			OEvent_avgpow[bind][cind].append(np.mean(df_band.avgpow))
			OEvent_peakF[bind][cind].append(np.mean(df_band.peakF))
			OEvent_Fspan[bind][cind].append(np.mean(df_band.Fspan))
			OEvent_WavePeakVal[bind][cind].append(np.mean(df_band.WavePeakVal))
			OEvent_WaveletPeakVal[bind][cind].append(np.mean(df_band.WaveletPeakVal))
			OEvent_WaveH[bind][cind].append(np.mean(df_band.WaveH))

			OEvent_codelta[bind][cind].append(np.mean(df_band.codelta))
			OEvent_cotheta[bind][cind].append(np.mean(df_band.cotheta))
			OEvent_coalpha[bind][cind].append(np.mean(df_band.coalpha))
			OEvent_cobeta[bind][cind].append(np.mean(df_band.cobeta))
			OEvent_colgamma[bind][cind].append(np.mean(df_band.colgamma))
			OEvent_cogamma[bind][cind].append(np.mean(df_band.cogamma))
			OEvent_cohgamma[bind][cind].append(np.mean(df_band.cohgamma))

def plot_bars(data,ylabel,filename):
	df_stats = pd.DataFrame(columns=["Band",
									"Group 1",
									"Group 2",
									"Metric",
									"Group 1 Mean",
									"Group 1 SD",
									"t-stat (rel)",
									"p-value (rel)",
									"Cohen's d",
									"Normal test stat",
									"Normal test p-value",
									"Levene test stat",
									"Levene test pvalue",
									"MWU stat",
									"MWU p-value"])
	
	fig_bands, ax_bands = plt.subplots(nrows=1,ncols=1,figsize=(8,4))
	xvals = [x for x in range(len(bands))]
	width = 0.3
	offset = [-width,0,width] # of same length conds
	
	all_m = [[np.mean(d) for d in d1] for d1 in data]
	all_sd = [[np.std(d) for d in d1] for d1 in data]
	max_m = np.max(all_m)
	min_m = np.min(all_m)
	max_sd = np.max(all_sd)
	for bind,b in enumerate(bands):
		for cind,c in enumerate(conds):
			m_ = np.mean(data[bind][cind])
			sd_ = np.std(data[bind][cind])
			max_ = np.max(data[bind][cind])
			min_ = np.min(data[bind][cind])
			for cind2,c2 in enumerate(conds):
				tstat,pval = st.ttest_rel(data[bind][cind],data[bind][cind2])
				cd = cohen_d(data[bind][cind],data[bind][cind2])
				nstat,npval = st.normaltest(data[bind][cind])
				lstat,lpval = st.levene(data[bind][cind],data[bind][cind2],center='mean')
				mwu_stat,mwu_pval = st.mannwhitneyu(data[bind][cind],data[bind][cind2])
				df_stats = df_stats.append({"Band":b,
									"Group 1" : c,
									"Group 2": c2,
									"Metric": ylabel,
									"Group 1 Mean": m_,
									"Group 1 SD": sd_,
									"t-stat (rel)": tstat,
									"p-value (rel)": pval,
									"Cohen's d": cd,
									"Normal test stat": nstat,
									"Normal test p-value": npval,
									"Levene test stat": lstat,
									"Levene test pvalue": lpval,
									"MWU stat": mwu_stat,
									"MWU p-value": mwu_pval},
									ignore_index = True)
				if ((cind>0) & (cind2==0)): # if the index is MDD or MDD+a5PAM + if the comparison index is healthy
					if ((pval < p_thresh) & (abs(cd) > c_thresh)):
						ax_bands.text(xvals[bind]+offset[cind],m_+3*max_sd if np.sign(m_) == 1 else m_-3*max_sd,'*',c='k',va='top', ha='center',fontweight='bold',fontsize=fsize)
				if ((cind==2) & (cind2==1)):  # if the index is MDD+a5PAM + if the comparison index is MDD
					if ((pval < p_thresh) & (abs(cd) > c_thresh)):
						ax_bands.text(xvals[bind]+offset[cind],m_+3*max_sd if np.sign(m_) == 1 else m_-3*max_sd,'*',c=colors_conds[1],va='center', ha='center',fontweight='bold',fontsize=fsize)
			
			ax_bands.bar(xvals[bind]+offset[cind],
						height=m_,
						yerr=sd_,
						capsize=8,
						width=width,
						color=colors_conds[cind],
						edgecolor='k',
						linewidth=1,
						error_kw={'elinewidth':3,'markeredgewidth':3}
					   )
	
	ax_bands.set_ylim((0,max_m+4.5*max_sd) if np.sign(max_m) == 1 else (min_m-4.5*max_sd,0))
	ax_bands.set_xticks(xvals)
	ax_bands.set_xticklabels(band_labels)
	ax_bands.set_ylabel(ylabel)
	ax_bands.grid(False)
	ax_bands.spines['right'].set_visible(False)
	ax_bands.spines['top'].set_visible(False)
	fig_bands.tight_layout()
	fig_bands.savefig('figs_OEvents_V1_thresh4/OEvent_'+filename+'.png',dpi=300,transparent=True,bbox_inches='tight')
	plt.close()
	
	df_stats.to_csv('figs_OEvents_V1_thresh4/stats_OEvents_'+filename+'.csv')

plot_bars(OEvent_counts,'Event Count','counts')
plot_bars(OEvent_dur,'Event Duration (ms)','durations')
plot_bars(OEvent_ncycles,'Cycle Count','cycles')
plot_bars(OEvent_avgpow,'Normalized Power','powers')
plot_bars(OEvent_peakF,'Peak Frequency (Hz)','peakfreq')
plot_bars(OEvent_Fspan,'Frequency Span (Hz)','freqspan')
plot_bars(OEvent_WavePeakVal,'Wave Peak','WavePeakVal')
plot_bars(OEvent_WaveletPeakVal,'Wavelet Peak','WaveletPeakVal')
plot_bars(OEvent_WaveH,'Wave Height (mV)','WaveH')

def plot_boxplots(data,ylabel,filename):
	df_stats = pd.DataFrame(columns=["Band",
									"Group 1",
									"Group 2",
									"Metric",
									"Group 1 Mean",
									"Group 1 SD",
									"t-stat (rel)",
									"p-value (rel)",
									"Cohen's d",
									"Normal test stat",
									"Normal test p-value",
									"Levene test stat",
									"Levene test pvalue",
									"MWU stat",
									"MWU p-value"])
	
	fig_bands, ax_bands = plt.subplots(nrows=1,ncols=1,figsize=(13,5))
	xvals = [x for x in range(len(bands))]
	width = 0.3
	offset = [-width,0,width] # of same length conds
	
	all_m = [[np.mean(d) for d in d1] for d1 in data]
	all_sd = [[np.std(d) for d in d1] for d1 in data]
	max_m = np.max(all_m)
	min_m = np.min(all_m)
	max_sd = np.max(all_sd)
	for bind,b in enumerate(bands[:-1]): # Skip high gamma
		for cind,c in enumerate(conds):
			m_ = np.mean(data[bind][cind])
			sd_ = np.std(data[bind][cind])
			max_ = np.max(data[bind][cind])
			min_ = np.min(data[bind][cind])
			for cind2,c2 in enumerate(conds):
				tstat,pval = st.ttest_rel(data[bind][cind],data[bind][cind2])
				cd = cohen_d(data[bind][cind],data[bind][cind2])
				nstat,npval = st.normaltest(data[bind][cind])
				lstat,lpval = st.levene(data[bind][cind],data[bind][cind2],center='mean')
				mwu_stat,mwu_pval = st.mannwhitneyu(data[bind][cind],data[bind][cind2])
				df_stats = df_stats.append({"Band":b,
									"Group 1" : c,
									"Group 2": c2,
									"Metric": ylabel,
									"Group 1 Mean": m_,
									"Group 1 SD": sd_,
									"t-stat (rel)": tstat,
									"p-value (rel)": pval,
									"Cohen's d": cd,
									"Normal test stat": nstat,
									"Normal test p-value": npval,
									"Levene test stat": lstat,
									"Levene test pvalue": lpval,
									"MWU stat": mwu_stat,
									"MWU p-value": mwu_pval},
									ignore_index = True)
				if ((cind>0) & (cind2==0)): # if the index is MDD or MDD+a5PAM + if the comparison index is healthy
					if ((pval < p_thresh) & (abs(cd) > c_thresh)):
						ax_bands.text(xvals[bind]+offset[cind],max_+2*max_sd if np.sign(m_) == 1 else m_-3*max_sd,'*',c='k',va='top', ha='center',fontweight='bold',fontsize=fsize)
				if ((cind==2) & (cind2==1)):  # if the index is MDD+a5PAM + if the comparison index is MDD
					if ((pval < p_thresh) & (abs(cd) > c_thresh)):
						ax_bands.text(xvals[bind]+offset[cind],max_+2*max_sd if np.sign(m_) == 1 else m_-3*max_sd,'*',c=colors_conds[1],va='center', ha='center',fontweight='bold',fontsize=fsize)
			
			ax_bands.boxplot(data[bind][cind],
							positions=[xvals[bind]+offset[cind]],
							boxprops=dict(color=colors_conds[cind],linewidth=3),
							capprops=dict(color=colors_conds[cind],linewidth=3),
							whiskerprops=dict(color=colors_conds[cind],linewidth=3),
							flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
							medianprops=dict(color=colors_conds[cind],linewidth=3),
							widths=width-0.05
							)
	
	if np.sign(max_m) == 1:
		ax_bands.set_ylim(top=max_m+4.5*max_sd)
	else:
		ax_bands.set_ylim(bottom=min_m-4.5*max_sd)
	ax_bands.set_xticks(xvals[:-1])
	ax_bands.set_xticklabels(band_labels[:-1])
	ax_bands.set_ylabel(ylabel)
	ax_bands.grid(False)
	ax_bands.spines['right'].set_visible(False)
	ax_bands.spines['top'].set_visible(False)
	fig_bands.tight_layout()
	fig_bands.savefig('figs_OEvents_V1_thresh4/OEvent_'+filename+'_bp.png',dpi=300,transparent=True,bbox_inches='tight')
	plt.close()
	
	df_stats.to_csv('figs_OEvents_V1_thresh4/stats_OEvents_'+filename+'_bp.csv')

plot_boxplots(OEvent_counts,'Event Count','counts')
plot_boxplots(OEvent_dur,'Event Duration (ms)','durations')
plot_boxplots(OEvent_ncycles,'Cycle Count','cycles')
plot_boxplots(OEvent_avgpow,'Normalized Power','powers')
plot_boxplots(OEvent_peakF,'Peak Frequency (Hz)','peakfreq')
plot_boxplots(OEvent_Fspan,'Frequency Span (Hz)','freqspan')
plot_boxplots(OEvent_WavePeakVal,'Wave Peak','WavePeakVal')
plot_boxplots(OEvent_WaveletPeakVal,'Wavelet Peak','WaveletPeakVal')
plot_boxplots(OEvent_WaveH,'Wave Height (mV)','WaveH')


def plot_cooccurences(data):
	# convert raw data to means per condition
	new_data = [[[[] for _ in conds] for _ in bands] for _ in data]
	for d1,_ in enumerate(data):
		for bind,b in enumerate(bands):
			for cind,c in enumerate(conds):
				new_data[d1][bind][cind] = np.mean(data[d1][bind][cind])*100
	
	rearranged_data = np.transpose(new_data, axes=[2, 0, 1])
	for cind,c in enumerate(conds):
		fig, ax = plt.subplots()
		im = ax.imshow(rearranged_data[cind],cmap="viridis")
		cbar = ax.figure.colorbar(im, ax=ax)
		cbar.ax.set_ylabel('Mean % Co-occurence', rotation=-90, va="bottom")
		ax.set_title('X Co-occurence with Y')
		ax.set_xticks(np.arange(len(band_labels)))
		ax.set_xticklabels(band_labels)
		ax.set_yticks(np.arange(len(band_labels)))
		ax.set_yticklabels(band_labels)
		fig.tight_layout()
		fig.savefig('figs_OEvents_V1_thresh4/OEvent_cooccurences_'+c+'.png',dpi=300,transparent=True)
		plt.close()
all_data = [OEvent_codelta,OEvent_cotheta,OEvent_coalpha,OEvent_cobeta,OEvent_colgamma,OEvent_cogamma,OEvent_cohgamma]
plot_cooccurences(all_data)
