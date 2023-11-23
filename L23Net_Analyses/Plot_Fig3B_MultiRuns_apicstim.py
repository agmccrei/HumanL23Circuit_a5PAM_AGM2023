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

N_seeds = 200
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

conds = ['healthy_apicstim','MDD_apicstim','MDD_a5PAM_apicstim']
paths = ['Saved_SpikesOnly/' + i + '/' for i in conds]

x_labels = ['Healthy','MDD','MDD\n'+r'$\alpha$'+'5-PAM']
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

def autoscale_y(ax,margin=0.1,multiplier=1):
	"""This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
	ax -- a matplotlib axes object
	margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""
	
	import numpy as np
	
	def get_bottom_top(line):
		xd = line.get_xdata()
		yd = line.get_ydata()
		lo,hi = ax.get_xlim()
		y_displayed = yd[((xd>lo) & (xd<hi))]
		h = np.max(y_displayed) - np.min(y_displayed)
		bot = np.min(y_displayed)-margin*h
		top = np.max(y_displayed)+margin*h*multiplier
		return bot,top
	
	lines = ax.get_lines()
	bot,top = np.inf, -np.inf
	
	for line in lines[:-1]:
		new_bot, new_top = get_bottom_top(line)
		if new_bot < bot: bot = new_bot
		if new_top > top: top = new_top
	
	ax.set_ylim(bot,top)

def barplot_annotate_brackets(fig, ax, num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
	"""
	Annotate barplot with p-values.
	
	:param num1: number of left bar to put bracket over
	:param num2: number of right bar to put bracket over
	:param data: string to write or number for generating asterixes
	:param center: centers of all bars (like plt.bar() input)
	:param height: heights of all bars (like plt.bar() input)
	:param yerr: yerrs of all bars (like plt.bar() input)
	:param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
	:param barh: bar height in axes coordinates (0 to 1)
	:param fs: font size
	:param maxasterix: maximum number of asterixes to write (for very small p-values)
	"""
	
	if type(data) is str:
		text = data
	else:
		# * is p < 0.05
		# ** is p < 0.005
		# *** is p < 0.0005
		# etc.
		text = ''
		p = .05
		
		while data < p:
			text += '*'
			p /= 10.
			
			if maxasterix and len(text) == maxasterix:
				break
		
		if len(text) == 0:
			text = 'n. s.'
	
	lx, ly = center[num1], height[num1]
	rx, ry = center[num2], height[num2]
	
	if yerr:
		ly += yerr[num1]
		ry += yerr[num2]
	
	ax_y0, ax_y1 = fig.gca().get_ylim()
	dh *= (ax_y1 - ax_y0)
	barh *= (ax_y1 - ax_y0)
	
	y = max(ly, ry) + dh
	
	barx = [lx, lx, rx, rx]
	bary = [y, y+barh/2, y+barh/2, y]
	mid = ((lx+rx)/2, y+barh*(3/4))
	
	ax.plot(barx, bary, c='black')
	
	kwargs = dict(ha='center', va='bottom')
	if fs is not None:
		kwargs['fontsize'] = fs
	
	ax.text(*mid, text, **kwargs)

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
		spikebinvec = np.histogram(popspikes_All,bins=np.arange(startsclice,stimtime+dt,dt))[0]
		spikebinvec_PN = np.histogram(popspikes_PN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
		spikebinvec_MN = np.histogram(popspikes_MN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
		spikebinvec_BN = np.histogram(popspikes_BN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
		spikebinvec_VN = np.histogram(popspikes_VN,bins=np.arange(startsclice,stimtime+dt,dt))[0]
		
		nperseg = len(tvec[t1:tstim])/2
		sampling_rate = (1/dt)*1000
		f_All, Pxx_den_All = ss.welch(spikebinvec, fs=sampling_rate, nperseg=nperseg)
		f_PN, Pxx_den_PN = ss.welch(spikebinvec_PN, fs=sampling_rate, nperseg=nperseg)
		f_MN, Pxx_den_MN = ss.welch(spikebinvec_MN, fs=sampling_rate, nperseg=nperseg)
		f_BN, Pxx_den_BN = ss.welch(spikebinvec_BN, fs=sampling_rate, nperseg=nperseg)
		f_VN, Pxx_den_VN = ss.welch(spikebinvec_VN, fs=sampling_rate, nperseg=nperseg)
		
		fmaxval = 101
		fmaxind = np.where(f_All>=fmaxval)[0][0]
		
		spikes[cind].append(Pxx_den_All[:fmaxind])
		spikes_PN[cind].append(Pxx_den_PN[:fmaxind])
		spikes_MN[cind].append(Pxx_den_MN[:fmaxind])
		spikes_BN[cind].append(Pxx_den_BN[:fmaxind])
		spikes_VN[cind].append(Pxx_den_VN[:fmaxind])

# Plot Rates
x = [[] for _ in x_labels_types]
ntypes = len(x_labels_types)
nconds = len(conds)
for i in range(0,ntypes):
	x[i] = np.linspace(0,nconds-1,nconds) + (nconds+1)*i

x = np.transpose(x)
x_PN = np.linspace(0,nconds-1,nconds)
x_types = np.linspace(0,ntypes-1,ntypes)

PN_SNRs_healthy_MDDdrug = []
maxPNrate=0
maxPNstimrate=0
maxPNsnr=0
maxPNpercentsilent=0

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

p_thresh = 0.05
c_thresh = 1.

fig_rates, ax_rates = plt.subplots(figsize=(8, 3.2))
fig_rates_PN, ax_rates_PN = plt.subplots(figsize=(8, 3.5))
fig_stimrates, ax_stimrates = plt.subplots(figsize=(8, 3.2))
fig_stimrates_PN, ax_stimrates_PN = plt.subplots(figsize=(8, 5))
fig_snr, ax_snr = plt.subplots(figsize=(8, 3.2))
fig_snr_PN, ax_snr_PN = plt.subplots(figsize=(6, 6))
fig_percentsilent, ax_percentsilent = plt.subplots(figsize=(6, 5))
fig_percentsilent_PN, ax_percentsilent_PN = plt.subplots(figsize=(6, 6))

fig_rates_bp, ax_rates_bp = plt.subplots(figsize=(8, 3.2))
fig_rates_PN_bp, ax_rates_PN_bp = plt.subplots(figsize=(8, 3.5))
fig_stimrates_bp, ax_stimrates_bp = plt.subplots(figsize=(8, 3.2))
fig_stimrates_PN_bp, ax_stimrates_PN_bp = plt.subplots(figsize=(8, 5))
fig_snr_bp, ax_snr_bp = plt.subplots(figsize=(8, 3.2))
fig_snr_PN_bp, ax_snr_PN_bp = plt.subplots(figsize=(6, 6))
fig_percentsilent_bp, ax_percentsilent_bp = plt.subplots(figsize=(6, 5))
fig_percentsilent_PN_bp, ax_percentsilent_PN_bp = plt.subplots(figsize=(6, 6))
for cind, cond in enumerate(conds):
	meanRates = np.mean(rates[cind],0)
	meanStimRates = np.mean(stimrates[cind],0)
	meanSNRs = np.mean(snrs[cind],0)
	meanSilent = np.mean(percentsilent[cind],0)
	stdevRates = np.std(rates[cind],0)
	stdevStimRates = np.std(stimrates[cind],0)
	stdevSNRs = np.std(snrs[cind],0)
	stdevSilent = np.std(percentsilent[cind],0)
	
	PN_SNRs_healthy_MDDdrug.append(np.transpose(snrs[cind])[0]) # PN only
	
	# vs healthy
	tstat_PN_0, pval_PN_0 = st.ttest_rel(np.transpose(rates[0])[0],np.transpose(rates[cind])[0])
	tstat_stimPN_0, pval_stimPN_0 = st.ttest_rel(np.transpose(stimrates[0])[0],np.transpose(stimrates[cind])[0])
	tstat_snrPN_0, pval_snrPN_0 = st.ttest_rel(np.transpose(snrs[0])[0],np.transpose(snrs[cind])[0])
	tstat_PNS_0, pval_PNS_0 = st.ttest_rel(np.transpose(percentsilent[0])[0],np.transpose(percentsilent[cind])[0])
	cd_PN_0 = cohen_d(np.transpose(rates[0])[0],np.transpose(rates[cind])[0])
	cd_stimPN_0 = cohen_d(np.transpose(stimrates[0])[0],np.transpose(stimrates[cind])[0])
	cd_snrPN_0 = cohen_d(np.transpose(snrs[0])[0],np.transpose(snrs[cind])[0])
	cd_PNS_0 = cohen_d(np.transpose(percentsilent[0])[0],np.transpose(percentsilent[cind])[0])
		
	tstat_SST_0, pval_SST_0 = st.ttest_rel(np.transpose(rates[0])[1],np.transpose(rates[cind])[1])
	tstat_stimSST_0, pval_stimSST_0 = st.ttest_rel(np.transpose(stimrates[0])[1],np.transpose(stimrates[cind])[1])
	tstat_snrSST_0, pval_snrSST_0 = st.ttest_rel(np.transpose(snrs[0])[1],np.transpose(snrs[cind])[1])
	tstat_SSTS_0, pval_SSTS_0 = st.ttest_rel(np.transpose(percentsilent[0])[1],np.transpose(percentsilent[cind])[1])
	cd_SST_0 = cohen_d(np.transpose(rates[0])[1],np.transpose(rates[cind])[1])
	cd_stimSST_0 = cohen_d(np.transpose(stimrates[0])[1],np.transpose(stimrates[cind])[1])
	cd_snrSST_0 = cohen_d(np.transpose(snrs[0])[1],np.transpose(snrs[cind])[1])
	cd_SSTS_0 = cohen_d(np.transpose(percentsilent[0])[1],np.transpose(percentsilent[cind])[1])
		
	tstat_PV_0, pval_PV_0 = st.ttest_rel(np.transpose(rates[0])[2],np.transpose(rates[cind])[2])
	tstat_stimPV_0, pval_stimPV_0 = st.ttest_rel(np.transpose(stimrates[0])[2],np.transpose(stimrates[cind])[2])
	tstat_snrPV_0, pval_snrPV_0 = st.ttest_rel(np.transpose(snrs[0])[2],np.transpose(snrs[cind])[2])
	tstat_PVS_0, pval_PVS_0 = st.ttest_rel(np.transpose(percentsilent[0])[2],np.transpose(percentsilent[cind])[2])
	cd_PV_0 = cohen_d(np.transpose(rates[0])[2],np.transpose(rates[cind])[2])
	cd_stimPV_0 = cohen_d(np.transpose(stimrates[0])[2],np.transpose(stimrates[cind])[2])
	cd_snrPV_0 = cohen_d(np.transpose(snrs[0])[2],np.transpose(snrs[cind])[2])
	cd_PVS_0 = cohen_d(np.transpose(percentsilent[0])[2],np.transpose(percentsilent[cind])[2])
		
	tstat_VIP_0, pval_VIP_0 = st.ttest_rel(np.transpose(rates[0])[3],np.transpose(rates[cind])[3])
	tstat_stimVIP_0, pval_stimVIP_0 = st.ttest_rel(np.transpose(stimrates[0])[3],np.transpose(stimrates[cind])[3])
	tstat_snrVIP_0, pval_snrVIP_0 = st.ttest_rel(np.transpose(snrs[0])[3],np.transpose(snrs[cind])[3])
	tstat_VIPS_0, pval_VIPS_0 = st.ttest_rel(np.transpose(percentsilent[0])[3],np.transpose(percentsilent[cind])[3])
	cd_VIP_0 = cohen_d(np.transpose(rates[0])[3],np.transpose(rates[cind])[3])
	cd_stimVIP_0 = cohen_d(np.transpose(stimrates[0])[3],np.transpose(stimrates[cind])[3])
	cd_snrVIP_0 = cohen_d(np.transpose(snrs[0])[3],np.transpose(snrs[cind])[3])
	cd_VIPS_0 = cohen_d(np.transpose(percentsilent[0])[3],np.transpose(percentsilent[cind])[3])
		
	# vs MDD
	tstat_PN, pval_PN = st.ttest_rel(np.transpose(rates[1])[0],np.transpose(rates[cind])[0])
	tstat_stimPN, pval_stimPN = st.ttest_rel(np.transpose(stimrates[1])[0],np.transpose(stimrates[cind])[0])
	tstat_snrPN, pval_snrPN = st.ttest_rel(np.transpose(snrs[1])[0],np.transpose(snrs[cind])[0])
	tstat_PNS, pval_PNS = st.ttest_rel(np.transpose(percentsilent[1])[0],np.transpose(percentsilent[cind])[0])
	cd_PN = cohen_d(np.transpose(rates[1])[0],np.transpose(rates[cind])[0])
	cd_stimPN = cohen_d(np.transpose(stimrates[1])[0],np.transpose(stimrates[cind])[0])
	cd_snrPN = cohen_d(np.transpose(snrs[1])[0],np.transpose(snrs[cind])[0])
	cd_PNS = cohen_d(np.transpose(percentsilent[1])[0],np.transpose(percentsilent[cind])[0])
	
	tstat_SST, pval_SST = st.ttest_rel(np.transpose(rates[1])[1],np.transpose(rates[cind])[1])
	tstat_stimSST, pval_stimSST = st.ttest_rel(np.transpose(stimrates[1])[1],np.transpose(stimrates[cind])[1])
	tstat_snrSST, pval_snrSST = st.ttest_rel(np.transpose(snrs[1])[1],np.transpose(snrs[cind])[1])
	tstat_SSTS, pval_SSTS = st.ttest_rel(np.transpose(percentsilent[1])[1],np.transpose(percentsilent[cind])[1])
	cd_SST = cohen_d(np.transpose(rates[1])[1],np.transpose(rates[cind])[1])
	cd_stimSST = cohen_d(np.transpose(stimrates[1])[1],np.transpose(stimrates[cind])[1])
	cd_snrSST = cohen_d(np.transpose(snrs[1])[1],np.transpose(snrs[cind])[1])
	cd_SSTS = cohen_d(np.transpose(percentsilent[1])[1],np.transpose(percentsilent[cind])[1])
	
	tstat_PV, pval_PV = st.ttest_rel(np.transpose(rates[1])[2],np.transpose(rates[cind])[2])
	tstat_stimPV, pval_stimPV = st.ttest_rel(np.transpose(stimrates[1])[2],np.transpose(stimrates[cind])[2])
	tstat_snrPV, pval_snrPV = st.ttest_rel(np.transpose(snrs[1])[2],np.transpose(snrs[cind])[2])
	tstat_PVS, pval_PVS = st.ttest_rel(np.transpose(percentsilent[1])[2],np.transpose(percentsilent[cind])[2])
	cd_PV = cohen_d(np.transpose(rates[1])[2],np.transpose(rates[cind])[2])
	cd_stimPV = cohen_d(np.transpose(stimrates[1])[2],np.transpose(stimrates[cind])[2])
	cd_snrPV = cohen_d(np.transpose(snrs[1])[2],np.transpose(snrs[cind])[2])
	cd_PVS = cohen_d(np.transpose(percentsilent[1])[2],np.transpose(percentsilent[cind])[2])
	
	tstat_VIP, pval_VIP = st.ttest_rel(np.transpose(rates[1])[3],np.transpose(rates[cind])[3])
	tstat_stimVIP, pval_stimVIP = st.ttest_rel(np.transpose(stimrates[1])[3],np.transpose(stimrates[cind])[3])
	tstat_snrVIP, pval_snrVIP = st.ttest_rel(np.transpose(snrs[1])[3],np.transpose(snrs[cind])[3])
	tstat_VIPS, pval_VIPS = st.ttest_rel(np.transpose(percentsilent[1])[3],np.transpose(percentsilent[cind])[3])
	cd_VIP = cohen_d(np.transpose(rates[1])[3],np.transpose(rates[cind])[3])
	cd_stimVIP = cohen_d(np.transpose(stimrates[1])[3],np.transpose(stimrates[cind])[3])
	cd_snrVIP = cohen_d(np.transpose(snrs[1])[3],np.transpose(snrs[cind])[3])
	cd_VIPS = cohen_d(np.transpose(percentsilent[1])[3],np.transpose(percentsilent[cind])[3])
	
	metrics = ["Baseline Rate","Response Rate", "SNR"]
	m0 = [np.mean(rates[0],0)[0],np.mean(stimrates[0],0)[0],np.mean(snrs[0],0)[0]]
	sd0 = [np.std(rates[0],0)[0],np.std(stimrates[0],0)[0],np.std(snrs[0],0)[0]]
	m1 = [np.mean(rates[1],0)[0],np.mean(stimrates[1],0)[0],np.mean(snrs[1],0)[0]]
	sd1 = [np.std(rates[1],0)[0],np.std(stimrates[1],0)[0],np.std(snrs[1],0)[0]]
	m2 = [np.mean(rates[cind],0)[0],np.mean(stimrates[cind],0)[0],np.mean(snrs[cind],0)[0]]
	sd2 = [np.std(rates[cind],0)[0],np.std(stimrates[cind],0)[0],np.std(snrs[cind],0)[0]]
	
	m0SST = [np.mean(rates[0],0)[1],np.mean(stimrates[0],0)[1],np.mean(snrs[0],0)[1]]
	sd0SST = [np.std(rates[0],0)[1],np.std(stimrates[0],0)[1],np.std(snrs[0],0)[1]]
	m1SST = [np.mean(rates[1],0)[1],np.mean(stimrates[1],0)[1],np.mean(snrs[1],0)[1]]
	sd1SST = [np.std(rates[1],0)[1],np.std(stimrates[1],0)[1],np.std(snrs[1],0)[1]]
	m2SST = [np.mean(rates[cind],0)[1],np.mean(stimrates[cind],0)[1],np.mean(snrs[cind],0)[1]]
	sd2SST = [np.std(rates[cind],0)[1],np.std(stimrates[cind],0)[1],np.std(snrs[cind],0)[1]]
	
	m0PV = [np.mean(rates[0],0)[2],np.mean(stimrates[0],0)[2],np.mean(snrs[0],0)[2]]
	sd0PV = [np.std(rates[0],0)[2],np.std(stimrates[0],0)[2],np.std(snrs[0],0)[2]]
	m1PV = [np.mean(rates[1],0)[2],np.mean(stimrates[1],0)[2],np.mean(snrs[1],0)[2]]
	sd1PV = [np.std(rates[1],0)[2],np.std(stimrates[1],0)[2],np.std(snrs[1],0)[2]]
	m2PV = [np.mean(rates[cind],0)[2],np.mean(stimrates[cind],0)[2],np.mean(snrs[cind],0)[2]]
	sd2PV = [np.std(rates[cind],0)[2],np.std(stimrates[cind],0)[2],np.std(snrs[cind],0)[2]]
	
	m0VIP = [np.mean(rates[0],0)[3],np.mean(stimrates[0],0)[3],np.mean(snrs[0],0)[3]]
	sd0VIP = [np.std(rates[0],0)[3],np.std(stimrates[0],0)[3],np.std(snrs[0],0)[3]]
	m1VIP = [np.mean(rates[1],0)[3],np.mean(stimrates[1],0)[3],np.mean(snrs[1],0)[3]]
	sd1VIP = [np.std(rates[1],0)[3],np.std(stimrates[1],0)[3],np.std(snrs[1],0)[3]]
	m2VIP = [np.mean(rates[cind],0)[3],np.mean(stimrates[cind],0)[3],np.mean(snrs[cind],0)[3]]
	sd2VIP = [np.std(rates[cind],0)[3],np.std(stimrates[cind],0)[3],np.std(snrs[cind],0)[3]]
	
	tstat = [tstat_PN,tstat_stimPN,tstat_snrPN]
	p = [pval_PN,pval_stimPN,pval_snrPN]
	cd = [cd_PN,cd_stimPN,cd_snrPN]
	
	tstatSST = [tstat_SST,tstat_stimSST,tstat_snrSST]
	pSST = [pval_SST,pval_stimSST,pval_snrSST]
	cdSST = [cd_SST,cd_stimSST,cd_snrSST]
	
	tstatPV = [tstat_PV,tstat_stimPV,tstat_snrPV]
	pPV = [pval_PV,pval_stimPV,pval_snrPV]
	cdPV = [cd_PV,cd_stimPV,cd_snrPV]
	
	tstatVIP = [tstat_VIP,tstat_stimVIP,tstat_snrVIP]
	pVIP = [pval_VIP,pval_stimVIP,pval_snrVIP]
	cdVIP = [cd_VIP,cd_stimVIP,cd_snrVIP]
	
	tstat_0 = [tstat_PN_0,tstat_stimPN_0,tstat_snrPN_0]
	p_0 = [pval_PN_0,pval_stimPN_0,pval_snrPN_0]
	cd_0 = [cd_PN_0,cd_stimPN_0,cd_snrPN_0]
	
	tstat_0SST = [tstat_SST_0,tstat_stimSST_0,tstat_snrSST_0]
	p_0SST = [pval_SST_0,pval_stimSST_0,pval_snrSST_0]
	cd_0SST = [cd_SST_0,cd_stimSST_0,cd_snrSST_0]
	
	tstat_0PV = [tstat_PV_0,tstat_stimPV_0,tstat_snrPV_0]
	p_0PV = [pval_PV_0,pval_stimPV_0,pval_snrPV_0]
	cd_0PV = [cd_PV_0,cd_stimPV_0,cd_snrPV_0]
	
	tstat_0VIP = [tstat_VIP_0,tstat_stimVIP_0,tstat_snrVIP_0]
	p_0VIP = [pval_VIP_0,pval_stimVIP_0,pval_snrVIP_0]
	cd_0VIP = [cd_VIP_0,cd_stimVIP_0,cd_snrVIP_0]
	
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
	for dfi in range(0,len(metrics)):
		df_SST = df_SST.append({"Comparison" : x_labels[0] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0SST[dfi],
					"Group1 SD" : sd0SST[dfi],
					"Group2 Mean" : m2SST[dfi],
					"Group2 SD" : sd2SST[dfi],
					"t-stat" : tstat_0SST[dfi],
					"p-value" : p_0SST[dfi],
					"Cohen's d" : cd_0SST[dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_PV = df_PV.append({"Comparison" : x_labels[0] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0PV[dfi],
					"Group1 SD" : sd0PV[dfi],
					"Group2 Mean" : m2PV[dfi],
					"Group2 SD" : sd2PV[dfi],
					"t-stat" : tstat_0PV[dfi],
					"p-value" : p_0PV[dfi],
					"Cohen's d" : cd_0PV[dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_VIP = df_VIP.append({"Comparison" : x_labels[0] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m0VIP[dfi],
					"Group1 SD" : sd0VIP[dfi],
					"Group2 Mean" : m2VIP[dfi],
					"Group2 SD" : sd2VIP[dfi],
					"t-stat" : tstat_0VIP[dfi],
					"p-value" : p_0VIP[dfi],
					"Cohen's d" : cd_0VIP[dfi]},
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
	for dfi in range(0,len(metrics)):
		df_SST = df_SST.append({"Comparison" : x_labels[1] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1SST[dfi],
					"Group1 SD" : sd1SST[dfi],
					"Group2 Mean" : m2SST[dfi],
					"Group2 SD" : sd2SST[dfi],
					"t-stat" : tstatSST[dfi],
					"p-value" : pSST[dfi],
					"Cohen's d" : cdSST[dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_PV = df_PV.append({"Comparison" : x_labels[1] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1PV[dfi],
					"Group1 SD" : sd1PV[dfi],
					"Group2 Mean" : m2PV[dfi],
					"Group2 SD" : sd2PV[dfi],
					"t-stat" : tstatPV[dfi],
					"p-value" : pPV[dfi],
					"Cohen's d" : cdPV[dfi]},
					ignore_index = True)
	for dfi in range(0,len(metrics)):
		df_VIP = df_VIP.append({"Comparison" : x_labels[1] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1VIP[dfi],
					"Group1 SD" : sd1VIP[dfi],
					"Group2 Mean" : m2VIP[dfi],
					"Group2 SD" : sd2VIP[dfi],
					"t-stat" : tstatVIP[dfi],
					"p-value" : pVIP[dfi],
					"Cohen's d" : cdVIP[dfi]},
					ignore_index = True)
	i1 = np.max(np.transpose(rates[cind-1])[0])
	i2 = np.max(np.transpose(rates[cind])[0])
	maxPNrate = np.max([i1, i2, maxPNrate])+0.2#+(cind-1)/20
	i1 = np.max(np.transpose(stimrates[cind-1])[0])
	i2 = np.max(np.transpose(stimrates[cind])[0])
	maxPNstimrate = np.max([i1, i2, maxPNstimrate])+0.2#+(cind-1)/20
	i1 = np.max(np.transpose(snrs[cind-1])[0])
	i2 = np.max(np.transpose(snrs[cind])[0])
	maxPNsnr = np.max([i1, i2, maxPNsnr])-0.5#+(cind-1)
	i1 = np.max(np.transpose(percentsilent[cind-1])[0])
	i2 = np.max(np.transpose(percentsilent[cind])[0])
	maxPNpercentsilent = np.max([i1, i2, maxPNpercentsilent])+0.2#+(cind-1)*3
	
	# baseline
	ax_rates.bar(x[cind][1:],height=meanRates[1:],
		   yerr=stdevRates[1:],    # error bars
		   capsize=8, # error bar cap width in points
		   width=1,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		  )
	ax_rates_bp.boxplot(np.transpose(rates[cind]).tolist()[1:],positions=x[cind][1:],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	if ((pval_SST_0 < p_thresh) & (abs(cd_SST_0) > c_thresh) & (cind>0)):
		ax_rates.text(x[cind][1],meanRates[1]+stdevRates[1]+3,'*',c='k',va='bottom' if ((pval_SST < p_thresh) & (abs(cd_SST) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_bp.text(x[cind][1],np.max(np.transpose(rates[cind])[1])+3,'*',c='k',va='bottom' if ((pval_SST < p_thresh) & (abs(cd_SST) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_SST < p_thresh) & (abs(cd_SST) > c_thresh) & (cind>0)):
		ax_rates.text(x[cind][1],meanRates[1]+stdevRates[1]+3,'*',c=colors_conds[1],va='top' if ((pval_SST_0 < p_thresh) & (abs(cd_SST_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_bp.text(x[cind][1],np.max(np.transpose(rates[cind])[1])+3,'*',c=colors_conds[1],va='top' if ((pval_SST_0 < p_thresh) & (abs(cd_SST_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_PV_0 < p_thresh) & (abs(cd_PV_0) > c_thresh) & (cind>0)):
		ax_rates.text(x[cind][2],meanRates[2]+stdevRates[1]+3,'*',c='k',va='bottom' if ((pval_PV < p_thresh) & (abs(cd_PV) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_bp.text(x[cind][2],np.max(np.transpose(rates[cind])[2])+3,'*',c='k',va='bottom' if ((pval_PV < p_thresh) & (abs(cd_PV) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_PV < p_thresh) & (abs(cd_PV) > c_thresh) & (cind>0)):
		ax_rates.text(x[cind][2],meanRates[2]+stdevRates[2]+3,'*',c=colors_conds[1],va='top' if ((pval_PV_0 < p_thresh) & (abs(cd_PV_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_bp.text(x[cind][2],np.max(np.transpose(rates[cind])[2])+3,'*',c=colors_conds[1],va='top' if ((pval_PV_0 < p_thresh) & (abs(cd_PV_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_VIP_0 < p_thresh) & (abs(cd_VIP_0) > c_thresh) & (cind>0)):
		ax_rates.text(x[cind][3],meanRates[3]+stdevRates[3]+3,'*',c='k',va='bottom' if ((pval_VIP < p_thresh) & (abs(cd_VIP) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_bp.text(x[cind][3],np.max(np.transpose(rates[cind])[3])+3,'*',c='k',va='bottom' if ((pval_VIP < p_thresh) & (abs(cd_VIP) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_VIP < p_thresh) & (abs(cd_VIP) > c_thresh) & (cind>0)):
		ax_rates.text(x[cind][3],meanRates[3]+stdevRates[3]+3,'*',c=colors_conds[1],va='top' if ((pval_VIP_0 < p_thresh) & (abs(cd_VIP_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_bp.text(x[cind][3],np.max(np.transpose(rates[cind])[3])+3,'*',c=colors_conds[1],va='top' if ((pval_VIP_0 < p_thresh) & (abs(cd_VIP_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	
	ax_rates_PN.bar(x_PN[cind],height=meanRates[0],
		   yerr=stdevRates[0],    # error bars
		   capsize=12, # error bar cap width in points
		   width=0.7,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		   )
	ax_rates_PN_bp.boxplot(np.transpose(rates[cind]).tolist()[0],positions=[x_PN[cind]],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh) & (cind>0)):
		ax_rates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.3,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_PN_bp.text(x_PN[cind],np.max(np.transpose(rates[cind])[0])+0.3,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh) & (cind>0)):
		ax_rates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.3,'*',c=colors_conds[1],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_rates_PN_bp.text(x_PN[cind],np.max(np.transpose(rates[cind])[0])+0.3,'*',c=colors_conds[1],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	# if ((pval_PN < 0.05) & (cind > 0)):
	# 	barplot_annotate_brackets(fig_rates_PN, ax_rates_PN, cind-1, cind, '*', x_PN, [maxPNrate for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
	
	# stimulation
	ax_stimrates.bar(x[cind][1:],height=meanStimRates[1:],
		   yerr=stdevStimRates[1:],    # error bars
		   capsize=8, # error bar cap width in points
		   width=1,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		  )
	ax_stimrates_bp.boxplot(np.transpose(stimrates[cind]).tolist()[1:],positions=x[cind][1:],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	if ((pval_stimSST_0 < p_thresh) & (abs(cd_stimSST_0) > c_thresh) & (cind>0)):
		ax_stimrates.text(x[cind][1],meanStimRates[1]+stdevStimRates[1]+3,'*',c='k',va='bottom' if ((pval_stimSST < p_thresh) & (abs(cd_stimSST) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_bp.text(x[cind][1],np.max(np.transpose(stimrates[cind])[1])+3,'*',c='k',va='bottom' if ((pval_stimSST < p_thresh) & (abs(cd_stimSST) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimSST < p_thresh) & (abs(cd_stimSST) > c_thresh) & (cind>0)):
		ax_stimrates.text(x[cind][1],meanStimRates[1]+stdevStimRates[1]+3,'*',c=colors_conds[1],va='top' if ((pval_stimSST_0 < p_thresh) & (abs(cd_stimSST_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_bp.text(x[cind][1],np.max(np.transpose(stimrates[cind])[1])+3,'*',c=colors_conds[1],va='top' if ((pval_stimSST_0 < p_thresh) & (abs(cd_stimSST_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimPV_0 < p_thresh) & (abs(cd_stimPV_0) > c_thresh) & (cind>0)):
		ax_stimrates.text(x[cind][2],meanStimRates[2]+stdevStimRates[2]+3,'*',c='k',va='bottom' if ((pval_stimPV < p_thresh) & (abs(cd_stimPV) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_bp.text(x[cind][2],np.max(np.transpose(stimrates[cind])[2])+3,'*',c='k',va='bottom' if ((pval_stimPV < p_thresh) & (abs(cd_stimPV) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimPV < p_thresh) & (abs(cd_stimPV) > c_thresh) & (cind>0)):
		ax_stimrates.text(x[cind][2],meanStimRates[2]+stdevStimRates[2]+3,'*',c=colors_conds[1],va='top' if ((pval_stimPV_0 < p_thresh) & (abs(cd_stimPV_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_bp.text(x[cind][2],np.max(np.transpose(stimrates[cind])[2])+3,'*',c=colors_conds[1],va='top' if ((pval_stimPV_0 < p_thresh) & (abs(cd_stimPV_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimVIP_0 < p_thresh) & (abs(cd_stimVIP_0) > c_thresh) & (cind>0)):
		ax_stimrates.text(x[cind][3],meanStimRates[3]+stdevStimRates[3]+3,'*',c='k',va='bottom' if ((pval_stimVIP < p_thresh) & (abs(cd_stimVIP) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_bp.text(x[cind][3],np.max(np.transpose(stimrates[cind])[3])+3,'*',c='k',va='bottom' if ((pval_stimVIP < p_thresh) & (abs(cd_stimVIP) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimVIP < p_thresh) & (abs(cd_stimVIP) > c_thresh) & (cind>0)):
		ax_stimrates.text(x[cind][3],meanStimRates[3]+stdevStimRates[3]+3,'*',c=colors_conds[1],va='top' if ((pval_stimVIP_0 < p_thresh) & (abs(cd_stimVIP_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_bp.text(x[cind][3],np.max(np.transpose(stimrates[cind])[3])+3,'*',c=colors_conds[1],va='top' if ((pval_stimVIP_0 < p_thresh) & (abs(cd_stimVIP_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	
	ax_stimrates_PN.bar(x_PN[cind],height=meanRates[0],
		   yerr=stdevRates[0],    # error bars
		   capsize=12, # error bar cap width in points
		   width=1,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		   )
	ax_stimrates_PN.bar(x_PN[cind]+4,height=meanStimRates[0],
		   yerr=stdevStimRates[0],    # error bars
		   capsize=12, # error bar cap width in points
		   width=1,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		   )
	ax_stimrates_PN_bp.boxplot(np.transpose(rates[cind]).tolist()[0],positions=[x_PN[cind]],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	ax_stimrates_PN_bp.boxplot(np.transpose(stimrates[cind]).tolist()[0],positions=[x_PN[cind]+4],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh) & (cind>0)):
		ax_stimrates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.4,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_PN_bp.text(x_PN[cind],np.max(np.transpose(rates[cind])[0])+0.4,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh) & (cind>0)):
		ax_stimrates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.4,'*',c=colors_conds[1],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_PN_bp.text(x_PN[cind],np.max(np.transpose(rates[cind])[0])+0.4,'*',c=colors_conds[1],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimPN_0 < p_thresh) & (abs(cd_stimPN_0) > c_thresh) & (cind>0)):
		ax_stimrates_PN.text(x_PN[cind]+4,meanStimRates[0]+stdevStimRates[0]+0.4,'*',c='k',va='bottom' if ((pval_stimPN < p_thresh) & (abs(cd_stimPN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_PN_bp.text(x_PN[cind]+4,np.max(np.transpose(stimrates[cind])[0])+0.4,'*',c='k',va='bottom' if ((pval_stimPN < p_thresh) & (abs(cd_stimPN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_stimPN < p_thresh) & (abs(cd_stimPN) > c_thresh) & (cind>0)):
		ax_stimrates_PN.text(x_PN[cind]+4,meanStimRates[0]+stdevStimRates[0]+0.4,'*',c=colors_conds[1],va='top' if ((pval_stimPN_0 < p_thresh) & (abs(cd_stimPN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_stimrates_PN_bp.text(x_PN[cind]+4,np.max(np.transpose(stimrates[cind])[0])+0.4,'*',c=colors_conds[1],va='top' if ((pval_stimPN_0 < p_thresh) & (abs(cd_stimPN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	# if ((pval_PN < 0.05) & (cind > 0)):
	# 	barplot_annotate_brackets(fig_stimrates_PN, ax_stimrates_PN, cind-1, cind, '*', x_PN, [maxPNrate for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
	# if ((pval_stimPN < 0.05) & (cind > 0)):
	# 	barplot_annotate_brackets(fig_stimrates_PN, ax_stimrates_PN, cind-1, cind, '*', [x0+4 for x0 in x_PN], [maxPNstimrate for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
	
	# snr
	ax_snr.bar(x[cind][1:],height=meanSNRs[1:],
		   yerr=stdevSNRs[1:],    # error bars
		   capsize=8, # error bar cap width in points
		   width=1,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		  )
	ax_snr_bp.boxplot(np.transpose(snrs[cind]).tolist()[1:],positions=x[cind][1:],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	ax_snr_PN.bar(x_PN[cind],height=meanSNRs[0],
		   yerr=stdevSNRs[0],    # error bars
		   capsize=12, # error bar cap width in points
		   width=0.6,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		   )
	ax_snr_PN_bp.boxplot(np.transpose(snrs[cind]).tolist()[0],positions=[x_PN[cind]],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.55
		  )
	if ((pval_snrPN_0 < p_thresh) & (abs(cd_snrPN_0) > c_thresh) & (cind>0)):
		ax_snr_PN.text(x_PN[cind],meanSNRs[0]+stdevSNRs[0]+0.6,'*',c='k',va='bottom' if ((pval_snrPN < p_thresh) & (abs(cd_snrPN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_snr_PN_bp.text(x_PN[cind],np.max(np.transpose(snrs[cind])[0])+0.6,'*',c='k',va='bottom' if ((pval_snrPN < p_thresh) & (abs(cd_snrPN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_snrPN < p_thresh) & (abs(cd_snrPN) > c_thresh) & (cind>0)):
		ax_snr_PN.text(x_PN[cind],meanSNRs[0]+stdevSNRs[0]+0.6,'*',c=colors_conds[1],va='top' if ((pval_snrPN_0 < p_thresh) & (abs(cd_snrPN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_snr_PN_bp.text(x_PN[cind],np.max(np.transpose(snrs[cind])[0])+0.6,'*',c=colors_conds[1],va='top' if ((pval_snrPN_0 < p_thresh) & (abs(cd_snrPN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	# if ((pval_snrPN < 0.05) & (cind > 0)):
	# 	barplot_annotate_brackets(fig_snr_PN, ax_snr_PN, cind-1, cind, '*', x_PN, [maxPNsnr for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
	
	# percent silent
	ax_percentsilent.bar(x[cind][1:],height=meanSilent[1:],
		   yerr=stdevSilent[1:],    # error bars
		   capsize=8, # error bar cap width in points
		   width=1,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		  )
	ax_percentsilent_bp.boxplot(np.transpose(percentsilent[cind]).tolist()[1:],positions=x[cind][1:],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	ax_percentsilent_PN.bar(x_PN[cind],height=meanSilent[0],
		   yerr=stdevSilent[0],    # error bars
		   capsize=12, # error bar cap width in points
		   width=0.6,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   edgecolor='k',
		   ecolor='black',
		   linewidth=1,
		   error_kw={'elinewidth':3,'markeredgewidth':3}
		  )
	ax_percentsilent_PN_bp.boxplot(np.transpose(percentsilent[cind]).tolist()[0],positions=[x_PN[cind]],
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
	if ((pval_PNS_0 < p_thresh) & (abs(cd_PNS_0) > c_thresh) & (cind>0)):
		ax_percentsilent_PN.text(x_PN[cind],meanSilent[0]+stdevSilent[0]+2,'*',c='k',va='bottom' if ((pval_PNS < p_thresh) & (abs(cd_PNS) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_percentsilent_PN_bp.text(x_PN[cind],np.max(np.transpose(percentsilent[cind])[0])+2,'*',c='k',va='bottom' if ((pval_PNS < p_thresh) & (abs(cd_PNS) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((pval_PNS < p_thresh) & (abs(cd_PNS) > c_thresh) & (cind>0)):
		ax_percentsilent_PN.text(x_PN[cind],meanSilent[0]+stdevSilent[0]+2,'*',c=colors_conds[1],va='top' if ((pval_PNS_0 < p_thresh) & (abs(cd_PNS_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_percentsilent_PN_bp.text(x_PN[cind],np.max(np.transpose(percentsilent[cind])[0])+2,'*',c=colors_conds[1],va='top' if ((pval_PNS_0 < p_thresh) & (abs(cd_PNS_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	# if ((pval_PNS < 0.05) & (cind > 0)):
	# 	barplot_annotate_brackets(fig_percentsilent_PN, ax_percentsilent_PN, cind-1, cind, '*', x_PN, [maxPNpercentsilent for _ in range(0,len(conds))], yerr=[0.05 for _ in range(0,len(conds))])

df.to_csv('figsV1_apicstim/stats_Rates.csv')
df_SST.to_csv('figsV1_apicstim/stats_Rates_SST.csv')
df_PV.to_csv('figsV1_apicstim/stats_Rates_PV.csv')
df_VIP.to_csv('figsV1_apicstim/stats_Rates_VIP.csv')

ax_rates.set_ylabel('Baseline Rate')
ax_rates.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_rates.set_xticklabels(x_labels_types[1:])
ax_rates.grid(False)
ax_rates.spines['right'].set_visible(False)
ax_rates.spines['top'].set_visible(False)
fig_rates.tight_layout()
fig_rates.savefig('figsV1_apicstim/Rates.png',dpi=300,transparent=True)

ax_rates_bp.set_ylabel('Baseline Rate')
ax_rates_bp.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_rates_bp.set_xticklabels(x_labels_types[1:])
ax_rates_bp.grid(False)
ax_rates_bp.spines['right'].set_visible(False)
ax_rates_bp.spines['top'].set_visible(False)
fig_rates_bp.tight_layout()
fig_rates_bp.savefig('figsV1_apicstim/Rates_bp.png',dpi=300,transparent=True)

ax_rates_PN.set_ylabel('Baseline (Hz)')
ax_rates_PN.set_xticks(x_PN)
ax_rates_PN.set_xlim(-0.6,2.6)
ax_rates_PN.set_xticklabels(x_labels)
ax_rates_PN.grid(False)
ax_rates_PN.spines['right'].set_visible(False)
ax_rates_PN.spines['top'].set_visible(False)
fig_rates_PN.tight_layout()
fig_rates_PN.savefig('figsV1_apicstim/Rates_PN.png',dpi=300,transparent=True)

ax_rates_PN_bp.set_ylabel('Baseline (Hz)')
ax_rates_PN_bp.set_xticks(x_PN)
ax_rates_PN_bp.set_xlim(-0.6,2.6)
ax_rates_PN_bp.set_xticklabels(x_labels)
ax_rates_PN_bp.grid(False)
ax_rates_PN_bp.spines['right'].set_visible(False)
ax_rates_PN_bp.spines['top'].set_visible(False)
fig_rates_PN_bp.tight_layout()
fig_rates_PN_bp.savefig('figsV1_apicstim/Rates_PN_bp.png',dpi=300,transparent=True)

ax_stimrates.set_ylabel('Response Rate')
ax_stimrates.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_stimrates.set_xticklabels(x_labels_types[1:])
ax_stimrates.grid(False)
ax_stimrates.spines['right'].set_visible(False)
ax_stimrates.spines['top'].set_visible(False)
fig_stimrates.tight_layout()
fig_stimrates.savefig('figsV1_apicstim/StimRates.png',dpi=300,transparent=True)

ax_stimrates_bp.set_ylabel('Response Rate')
ax_stimrates_bp.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_stimrates_bp.set_xticklabels(x_labels_types[1:])
ax_stimrates_bp.grid(False)
ax_stimrates_bp.spines['right'].set_visible(False)
ax_stimrates_bp.spines['top'].set_visible(False)
fig_stimrates_bp.tight_layout()
fig_stimrates_bp.savefig('figsV1_apicstim/StimRates_bp.png',dpi=300,transparent=True)

ax_stimrates_PN.set_ylabel('Pyr Firing Rate (Hz)')
ax_stimrates_PN.set_xticks([1,5])
ax_stimrates_PN.set_xlim(-1.5,7.5)
ax_stimrates_PN.set_xticklabels(['Pre-stimulus','Post-stimulus'])
ax_stimrates_PN.grid(False)
ax_stimrates_PN.spines['right'].set_visible(False)
ax_stimrates_PN.spines['top'].set_visible(False)
fig_stimrates_PN.tight_layout()
fig_stimrates_PN.savefig('figsV1_apicstim/StimRates_PN.png',dpi=300,transparent=True)

ax_stimrates_PN_bp.set_ylabel('Pyr Firing Rate (Hz)')
ax_stimrates_PN_bp.set_xticks([1,5])
ax_stimrates_PN_bp.set_xlim(-1.5,7.5)
ax_stimrates_PN_bp.set_xticklabels(['Pre-stimulus','Post-stimulus'])
ax_stimrates_PN_bp.grid(False)
ax_stimrates_PN_bp.spines['right'].set_visible(False)
ax_stimrates_PN_bp.spines['top'].set_visible(False)
fig_stimrates_PN_bp.tight_layout()
fig_stimrates_PN_bp.savefig('figsV1_apicstim/StimRates_PN_bp.png',dpi=300,transparent=True)

ax_snr.set_ylabel('SNR')
ax_snr.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_snr.set_xticklabels(x_labels_types[1:])
ax_snr.grid(False)
ax_snr.spines['right'].set_visible(False)
ax_snr.spines['top'].set_visible(False)
fig_snr.tight_layout()
fig_snr.savefig('figsV1_apicstim/SNR.png',dpi=300,transparent=True)

ax_snr_bp.set_ylabel('SNR')
ax_snr_bp.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_snr_bp.set_xticklabels(x_labels_types[1:])
ax_snr_bp.grid(False)
ax_snr_bp.spines['right'].set_visible(False)
ax_snr_bp.spines['top'].set_visible(False)
fig_snr_bp.tight_layout()
fig_snr_bp.savefig('figsV1_apicstim/SNR_bp.png',dpi=300,transparent=True)

ax_snr_PN.set_ylabel('SNR')
ax_snr_PN.set_xticks(x_PN)
ax_snr_PN.set_xlim(-0.6,2.6)
ax_snr_PN.set_xticklabels(x_labels)
ax_snr_PN.grid(False)
ax_snr_PN.spines['right'].set_visible(False)
ax_snr_PN.spines['top'].set_visible(False)
fig_snr_PN.tight_layout()
fig_snr_PN.savefig('figsV1_apicstim/SNR_PN.png',dpi=300,transparent=True)

ax_snr_PN_bp.set_ylabel('SNR')
ax_snr_PN_bp.set_xticks(x_PN)
ax_snr_PN_bp.set_xlim(-0.6,2.6)
ax_snr_PN_bp.set_xticklabels(x_labels)
ax_snr_PN_bp.grid(False)
ax_snr_PN_bp.spines['right'].set_visible(False)
ax_snr_PN_bp.spines['top'].set_visible(False)
fig_snr_PN_bp.tight_layout()
fig_snr_PN_bp.savefig('figsV1_apicstim/SNR_PN_bp.png',dpi=300,transparent=True)

ax_percentsilent.set_ylabel('% Silent')
ax_percentsilent.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_percentsilent.set_xticklabels(x_labels_types[1:])
ax_percentsilent.grid(False)
ax_percentsilent.spines['right'].set_visible(False)
ax_percentsilent.spines['top'].set_visible(False)
fig_percentsilent.tight_layout()
fig_percentsilent.savefig('figsV1_apicstim/PercentSilent.png',dpi=300,transparent=True)

ax_percentsilent_bp.set_ylabel('% Silent')
ax_percentsilent_bp.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i) for i in range(1,len(x_labels_types))])
ax_percentsilent_bp.set_xticklabels(x_labels_types[1:])
ax_percentsilent_bp.grid(False)
ax_percentsilent_bp.spines['right'].set_visible(False)
ax_percentsilent_bp.spines['top'].set_visible(False)
fig_percentsilent_bp.tight_layout()
fig_percentsilent_bp.savefig('figsV1_apicstim/PercentSilent_bp.png',dpi=300,transparent=True)

ax_percentsilent_PN.set_ylabel('Pyr % Silent')
ax_percentsilent_PN.set_xticks(x_PN)
ax_percentsilent_PN.set_xticklabels(x_labels)
ax_percentsilent_PN.grid(False)
ax_percentsilent_PN.spines['right'].set_visible(False)
ax_percentsilent_PN.spines['top'].set_visible(False)
fig_percentsilent_PN.tight_layout()
fig_percentsilent_PN.savefig('figsV1_apicstim/PercentSilent_PN.png',dpi=300,transparent=True)

ax_percentsilent_PN_bp.set_ylabel('Pyr % Silent')
ax_percentsilent_PN_bp.set_xticks(x_PN)
ax_percentsilent_PN_bp.set_xticklabels(x_labels)
ax_percentsilent_PN_bp.grid(False)
ax_percentsilent_PN_bp.spines['right'].set_visible(False)
ax_percentsilent_PN_bp.spines['top'].set_visible(False)
fig_percentsilent_PN_bp.tight_layout()
fig_percentsilent_PN_bp.savefig('figsV1_apicstim/PercentSilent_PN_bp.png',dpi=300,transparent=True)
plt.close()

# Calculate percent recovery relative to healthy (t0 - target) and MDD (t1 - baseline)
precov_SNR = []
precov_SNR_sd = []
recov_labels = ['Stress\nVehicle','Stress\n(10 mg/kg)']
recov_cols = ['tab:gray','lightgrey']

healthy_behav_m = 87
stressed_behav_m = 59
stressed_a5PAM_behav_m = 78
healthy_behav_sd = 1.5
stressed_behav_sd = 2
stressed_a5PAM_behav_sd = 3

precov_Behaviour = (stressed_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_behav_sd/stressed_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append(precov_Behaviour_sd)

precov_Behaviour = (stressed_a5PAM_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_a5PAM_behav_sd/stressed_a5PAM_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append(precov_Behaviour_sd)

for ind in range(1,len(conds)):
	print('testing recovery in '+conds[ind]+' condition')
	recovs = [(t0[ind]/t0[0])*100 for t0 in np.transpose(PN_SNRs_healthy_MDDdrug)]
	precov_SNR.append(np.mean(recovs,0))
	precov_SNR_sd.append(np.std(recovs,0))
	recov_labels.append(x_labels[ind])
	recov_cols.append(colors_conds[ind])

xra = np.arange(0,len(precov_SNR))
fig_recov, ax_recov = plt.subplots(figsize=(9, 8))
ax_recov.bar(xra,height=precov_SNR,
		yerr=precov_SNR_sd,    # error bars
		capsize=8, # error bar cap width in points
		width=0.8,    # bar width
		color=recov_cols,  # face color transparent
		edgecolor='k',
		ecolor='black',
		linewidth=4,
		error_kw={'elinewidth':3,'markeredgewidth':3}
		)
ax_recov.set_ylabel('% of Healthy')
ax_recov.set_xticks(xra)
ax_recov.set_xticklabels(recov_labels)
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figsV1_apicstim/PercentRecovery_PNvsBehaviour_SeedBySeed.png',dpi=300,transparent=True)
plt.close()


# Recalculate with means instead of seed by seed
precov_SNR = []
precov_SNR_sd = []
precov_Behaviour = (stressed_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_behav_sd/stressed_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append(precov_Behaviour_sd)
precov_Behaviour = (stressed_a5PAM_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_a5PAM_behav_sd/stressed_a5PAM_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append(precov_Behaviour_sd)

for ind in range(1,len(conds)):
	print('testing recovery in '+conds[ind]+' condition')
	t0 = PN_SNRs_healthy_MDDdrug
	m1 = np.mean(t0[0],0)
	m2 = np.mean(t0[ind],0)
	sd1 = np.std(t0[0],0)
	sd2 = np.std(t0[ind],0)
	recovs = (m2/m1)*100
	recovs_sd = recovs*np.sqrt((sd1/m1)**2+(sd2/m2)**2)
	precov_SNR.append(recovs)
	precov_SNR_sd.append(recovs_sd)
	recov_labels.append(x_labels[ind])
	recov_cols.append(colors_conds[ind])

xra = np.arange(0,len(precov_SNR))
fig_recov, ax_recov = plt.subplots(figsize=(9, 8))
ax_recov.bar(xra,height=precov_SNR,
		yerr=precov_SNR_sd,    # error bars
		capsize=8, # error bar cap width in points
		width=0.8,    # bar width
		color=recov_cols,  # face color transparent
		edgecolor='k',
		ecolor='black',
		linewidth=4,
		error_kw={'elinewidth':3,'markeredgewidth':3}
		)
ax_recov.set_ylabel('% of Healthy')
ax_recov.set_xticks(xra)
ax_recov.set_xticklabels(recov_labels)
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figsV1_apicstim/PercentRecovery_PNvsBehaviour_MeansDivision.png',dpi=300,transparent=True)
plt.close()

# Recalculate with means healthy vs seed by seed in conditions
precov_SNR = []
precov_SNR_sd = []
precov_Behaviour = (stressed_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_behav_sd/stressed_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append(precov_Behaviour_sd)
precov_Behaviour = (stressed_a5PAM_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_a5PAM_behav_sd/stressed_a5PAM_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append(precov_Behaviour_sd)

all_precovs = []
for ind in range(1,len(conds)):
	print('testing recovery in '+conds[ind]+' condition')
	m1 = np.mean(PN_SNRs_healthy_MDDdrug[0],0)
	recovs = [(t0[ind]/m1)*100 for t0 in np.transpose(PN_SNRs_healthy_MDDdrug)]
	all_precovs.append(recovs)
	precov_SNR.append(np.mean(recovs,0))
	precov_SNR_sd.append(np.std(recovs,0))
	recov_labels.append(x_labels[ind])
	recov_cols.append(colors_conds[ind])

xra = np.arange(0,len(precov_SNR))
fig_recov, ax_recov = plt.subplots(figsize=(9, 8))
ax_recov.bar(xra,height=precov_SNR,
		yerr=precov_SNR_sd,    # error bars
		capsize=8, # error bar cap width in points
		width=0.8,    # bar width
		color=recov_cols,  # face color transparent
		edgecolor='k',
		ecolor='black',
		linewidth=4,
		error_kw={'elinewidth':3,'markeredgewidth':3}
		)
ax_recov.set_ylabel('% of Healthy')
ax_recov.set_xticks(xra)
ax_recov.set_xticklabels(recov_labels)
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figsV1_apicstim/PercentRecovery_PNvsBehaviour_Hybrid.png',dpi=300,transparent=True)
plt.close()

fig_recov, ax_recov = plt.subplots(figsize=(9, 8))
ax_recov.hist(all_precovs[0], bins=15, color=recov_cols[2], alpha=0.8)
ax_recov.hist(all_precovs[1], bins=15, color=recov_cols[3], alpha=0.8)
ax_recov.set_ylabel('Count')
ax_recov.set_xlabel('% of Healthy')
fig_recov.tight_layout()
fig_recov.savefig('figsV1_apicstim/PercentRecovery_PNvsBehaviour_Distribution.png',dpi=300,transparent=True)
plt.close()

# Recalculate with bootstrapped means
precov_SNR = []
precov_SNR_sd = []
precov_Behaviour = (stressed_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_behav_sd/stressed_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append((precov_Behaviour_sd,precov_Behaviour_sd))
precov_Behaviour = (stressed_a5PAM_behav_m/healthy_behav_m)*100 # estimated from Prevot et al, fig 4a
precov_Behaviour_sd = precov_Behaviour*np.sqrt((healthy_behav_sd/healthy_behav_m)**2+(stressed_a5PAM_behav_sd/stressed_a5PAM_behav_m)**2)
precov_SNR.append(precov_Behaviour)
precov_SNR_sd.append((precov_Behaviour_sd,precov_Behaviour_sd))

for ind in range(1,len(conds)):
	print('testing recovery in '+conds[ind]+' condition')
	m1 = np.mean(PN_SNRs_healthy_MDDdrug[0],0)
	recovs = [(t0[ind]/m1)*100 for t0 in np.transpose(PN_SNRs_healthy_MDDdrug)]
	x = bs.bootstrap(np.array(recovs), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
	precov_SNR.append(x.value)
	precov_SNR_sd.append((x.value-x.lower_bound,x.upper_bound-x.value))
	recov_labels.append(x_labels[ind])
	recov_cols.append(colors_conds[ind])

xra = np.arange(0,len(precov_SNR))
fig_recov, ax_recov = plt.subplots(figsize=(9, 8))
ax_recov.bar(xra,height=precov_SNR,
		width=0.8,    # bar width
		edgecolor='k',
		color=recov_cols,  # face color transparent
		linewidth=4,
		)
ax_recov.errorbar(xra,precov_SNR,
		yerr=np.transpose(precov_SNR_sd),    # error bars
		capsize=8, # error bar cap width in points
		ecolor='black',
		linestyle='',
		elinewidth=3,
		capthick=3
		)
ax_recov.set_ylabel('% of Healthy')
ax_recov.set_xticks(xra)
ax_recov.set_xticklabels(recov_labels)
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figsV1_apicstim/PercentRecovery_PNvsBehaviour_HybridBootstrapped.png',dpi=300,transparent=True)
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
			x = bs.bootstrap(np.transpose(spikes_PN[cind])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=500)
			CI_means_PN.append(x.value)
			CI_lower_PN.append(x.lower_bound)
			CI_upper_PN.append(x.upper_bound)
	else:
		CI_means_PN = np.mean(spikes_PN[cind],0)
		CI_lower_PN = np.mean(spikes_PN[cind],0)-np.std(spikes_PN[cind],0)
		CI_upper_PN = np.mean(spikes_PN[cind],0)+np.std(spikes_PN[cind],0)
	
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

ax.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0,r'$\Theta$')
ax.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0,r'$\alpha$')
ax.text(13.25,0,r'$\beta$')
ylims = ax.get_ylim()
ax.plot([thetaband[0],thetaband[0]],ylims,c='dimgrey',ls=':')
ax.plot([alphaband[0],alphaband[0]],ylims,c='dimgrey',ls=':')
ax.plot([alphaband[1],alphaband[1]],ylims,c='dimgrey',ls=':')
ax.set_ylim(ylims)
ax.set_xlim(freq0,freq1)

fig.savefig('figsV1_apicstim/Spikes_PSD_Boot95CI_PN.png',bbox_inches='tight',dpi=300, transparent=True)
plt.close()
