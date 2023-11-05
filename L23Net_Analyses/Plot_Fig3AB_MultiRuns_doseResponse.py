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
from scipy.optimize import curve_fit
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

conds = ['MDD',
		'MDD_a5PAM_0.25',
		'MDD_a5PAM_0.50',
		'MDD_a5PAM_0.75',
		'MDD_a5PAM',
		'MDD_a5PAM_1.25',
		'MDD_a5PAM_1.50',
		'healthy']

paths = ['Saved_SpikesOnly/' + i + '/' for i in conds]

x_labels = ['MDD',
			'25%',
			'50%',
			'75%',
			'100%',
			'125%',
			'150%',
			'Healthy']

x_labels_types = ['Pyr','SST','PV','VIP']

colors_neurs = ['dimgrey', 'red', 'green', 'orange']
colors_conds = ['tab:purple',
				'dodgerblue',
				'dodgerblue',
				'dodgerblue',
				'dodgerblue',
				'dodgerblue',
				'dodgerblue',
				'tab:gray']
alphas_conds = [1, 1/6, 2/6, 3/6, 4/6, 5/6, 1, 1]

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

# Fit functions to dose-curves
def linear(x, slope, intercept):
	return slope*x + intercept

def expo(x, A, K, C):
	return A * np.exp(x * K) + C

def neg_expo(x, A, K, C):
	return A * np.exp(-x * K) + C

def sigmoid(x, k, L, b, x0):
	return (L / (1 + np.exp(k * (-x - x0)))) + b

def rev_sigmoid(x, k, L, b, x0):
	return (L / (1 + np.exp(k * (x - x0)))) + b

xvals = [0,0.25,0.5,0.75,1,1.25,1.5]
xvals_highres = np.arange(0.001,1.5,0.001)

rates_to_fit = [[],[]]
for d,rc in zip(xvals,rates[:-1]):
	for r in np.transpose(rc)[0]:
		rates_to_fit[0].append(d)
		rates_to_fit[1].append(r)

snrs_to_fit = [[],[]]
for d,rc in zip(xvals,snrs[:-1]):
	for r in np.transpose(rc)[0]:
		snrs_to_fit[0].append(d)
		snrs_to_fit[1].append(r)

tstat_r,pval_r = st.pearsonr(rates_to_fit[0],rates_to_fit[1])
tstat_s,pval_s = st.pearsonr(snrs_to_fit[0],snrs_to_fit[1])

p0_rl = [tstat_r, np.max(rates_to_fit[1])]
p0_re = [1, 1, 1]
p0_rs = [2, 1, 1, 0.75]
p0_sl = [tstat_s, np.max(snrs_to_fit[1])]
p0_se = [1, 1, 1]
p0_ss = [-2, 1, 1, 0.75]

coeff_rl = curve_fit(linear, rates_to_fit[0], rates_to_fit[1], p0 = p0_rl, maxfev = 100000000, full_output=True)
coeff_re = curve_fit(neg_expo, rates_to_fit[0], rates_to_fit[1], p0 = p0_re, maxfev = 100000000, full_output=True)
coeff_rs = curve_fit(rev_sigmoid, rates_to_fit[0], rates_to_fit[1], p0 = p0_rs, maxfev = 100000000, full_output=True)
coeff_sl = curve_fit(linear, snrs_to_fit[0], snrs_to_fit[1], p0 = p0_sl, maxfev = 100000000, full_output=True)
coeff_se = curve_fit(expo, snrs_to_fit[0], snrs_to_fit[1], p0 = p0_se, maxfev = 100000000, full_output=True)
coeff_ss = curve_fit(sigmoid, snrs_to_fit[0], snrs_to_fit[1], p0 = p0_ss, maxfev = 100000000, full_output=True)

SSE_rl = sum((coeff_rl[2]['fvec'])**2)
SSE_re = sum((coeff_re[2]['fvec'])**2)
SSE_rs = sum((coeff_rs[2]['fvec'])**2)
SSE_sl = sum((coeff_sl[2]['fvec'])**2)
SSE_se = sum((coeff_se[2]['fvec'])**2)
SSE_ss = sum((coeff_ss[2]['fvec'])**2)

l_fit_r = linear(xvals_highres, *coeff_rl[0])
e_fit_r = neg_expo(xvals_highres, *coeff_re[0])
s_fit_r = rev_sigmoid(xvals_highres, *coeff_rs[0])
l_fit_s = linear(xvals_highres, *coeff_sl[0])
e_fit_s = expo(xvals_highres, *coeff_se[0])
s_fit_s = sigmoid(xvals_highres, *coeff_ss[0])

fig_dosefits_r, ax_dosefits_r = plt.subplots(figsize=(20, 8))
ax_dosefits_r.scatter(rates_to_fit[0], rates_to_fit[1], c='k', label='Data')
ax_dosefits_r.plot(xvals_highres, l_fit_r, c='r', lw=8, ls=':', label='Linear SSE: '+str(np.round(SSE_rl,3)) + '; R,p = ' + str(np.round(tstat_r,3)) + ',' + str(np.round(pval_r,3)))
print('Rate Linear Pearson Correlation P-value = '+str(pval_r))
ax_dosefits_r.plot(xvals_highres, e_fit_r, c='b', lw=6, ls='--', label='Negative Exponential SSE: '+str(np.round(SSE_re,3)))
ax_dosefits_r.plot(xvals_highres, s_fit_r, c='g', lw=4, ls='-.', label='Reverse Sigmoidal SSE: '+str(np.round(SSE_rs,3)))
ax_dosefits_r.set_xlabel('Dose Mulitplier')
ax_dosefits_r.set_ylabel('Pyr Firing Rate (Hz)')
ax_dosefits_r.legend()
fig_dosefits_r.tight_layout()
fig_dosefits_r.savefig('figsV1_doseResponse/DoseFits_Rate.png',dpi=300,transparent=True)
plt.close()

fig_dosefits_s, ax_dosefits_s = plt.subplots(figsize=(20, 8))
ax_dosefits_s.scatter(snrs_to_fit[0], snrs_to_fit[1], c='k', label='Data')
ax_dosefits_s.plot(xvals_highres, l_fit_s, c='r', lw=8, ls=':', label='Linear SSE: '+str(np.round(SSE_sl,3)) + '; R,p = ' + str(np.round(tstat_s,3)) + ',' + str(np.round(pval_s,3)))
print('SNR Linear Pearson Correlation P-value = '+str(pval_s))
ax_dosefits_s.plot(xvals_highres, e_fit_s, c='b', lw=6, ls='--', label='Exponential SSE: '+str(np.round(SSE_se,3)))
ax_dosefits_s.plot(xvals_highres, s_fit_s, c='g', lw=4, ls='-.', label='Sigmoidal SSE: '+str(np.round(SSE_ss,3)))
ax_dosefits_s.set_xlabel('Dose Mulitplier')
ax_dosefits_s.set_ylabel('SNR')
ax_dosefits_s.legend()
fig_dosefits_s.tight_layout()
fig_dosefits_s.savefig('figsV1_doseResponse/DoseFits_SNR.png',dpi=300,transparent=True)
plt.close()

# Plot Rates
x = [[] for _ in x_labels_types]
ntypes = len(x_labels_types)
nconds = len(conds)
for i in range(0,ntypes):
	x[i] = np.linspace(0,nconds-1,nconds) + (nconds+1)*i

x = np.transpose(x)
x_PN = np.linspace(0,nconds-1,nconds)
x_PN2 = np.concatenate((x_PN[:-1],x_PN[:-1]+8))
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

p_thresh = 0.05
c_thresh = 1.

fig_rates, ax_rates = plt.subplots(figsize=(8, 3.2))
fig_rates_PN, ax_rates_PN = plt.subplots(figsize=(7, 7))
fig_stimrates, ax_stimrates = plt.subplots(figsize=(8, 3.2))
fig_stimrates_PN, ax_stimrates_PN = plt.subplots(figsize=(9.5, 6))
fig_snr, ax_snr = plt.subplots(figsize=(8, 3.2))
fig_snr_PN, ax_snr_PN = plt.subplots(figsize=(7, 7))
fig_percentsilent, ax_percentsilent = plt.subplots(figsize=(6, 5))
fig_percentsilent_PN, ax_percentsilent_PN = plt.subplots(figsize=(7, 7))
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
	tstat_PN_0, pval_PN_0 = st.ttest_rel(np.transpose(rates[-1])[0],np.transpose(rates[cind])[0])
	tstat_stimPN_0, pval_stimPN_0 = st.ttest_rel(np.transpose(stimrates[-1])[0],np.transpose(stimrates[cind])[0])
	tstat_snrPN_0, pval_snrPN_0 = st.ttest_rel(np.transpose(snrs[-1])[0],np.transpose(snrs[cind])[0])
	tstat_PNS_0, pval_PNS_0 = st.ttest_rel(np.transpose(percentsilent[-1])[0],np.transpose(percentsilent[cind])[0])
	cd_PN_0 = cohen_d(np.transpose(rates[-1])[0],np.transpose(rates[cind])[0])
	cd_stimPN_0 = cohen_d(np.transpose(stimrates[-1])[0],np.transpose(stimrates[cind])[0])
	cd_snrPN_0 = cohen_d(np.transpose(snrs[-1])[0],np.transpose(snrs[cind])[0])
	cd_PNS_0 = cohen_d(np.transpose(percentsilent[-1])[0],np.transpose(percentsilent[cind])[0])
	
	# vs MDD
	tstat_PN, pval_PN = st.ttest_rel(np.transpose(rates[0])[0],np.transpose(rates[cind])[0])
	tstat_stimPN, pval_stimPN = st.ttest_rel(np.transpose(stimrates[0])[0],np.transpose(stimrates[cind])[0])
	tstat_snrPN, pval_snrPN = st.ttest_rel(np.transpose(snrs[0])[0],np.transpose(snrs[cind])[0])
	tstat_PNS, pval_PNS = st.ttest_rel(np.transpose(percentsilent[0])[0],np.transpose(percentsilent[cind])[0])
	cd_PN = cohen_d(np.transpose(rates[0])[0],np.transpose(rates[cind])[0])
	cd_stimPN = cohen_d(np.transpose(stimrates[0])[0],np.transpose(stimrates[cind])[0])
	cd_snrPN = cohen_d(np.transpose(snrs[0])[0],np.transpose(snrs[cind])[0])
	cd_PNS = cohen_d(np.transpose(percentsilent[0])[0],np.transpose(percentsilent[cind])[0])
	
	metrics = ["Baseline Rate","Response Rate", "SNR"]
	m0 = [np.mean(rates[-1],0)[0],np.mean(stimrates[-1],0)[0],np.mean(snrs[-1],0)[0]]
	sd0 = [np.std(rates[-1],0)[0],np.std(stimrates[-1],0)[0],np.std(snrs[-1],0)[0]]
	m1 = [np.mean(rates[0],0)[0],np.mean(stimrates[0],0)[0],np.mean(snrs[0],0)[0]]
	sd1 = [np.std(rates[0],0)[0],np.std(stimrates[0],0)[0],np.std(snrs[0],0)[0]]
	m2 = [np.mean(rates[cind],0)[0],np.mean(stimrates[cind],0)[0],np.mean(snrs[cind],0)[0]]
	sd2 = [np.std(rates[cind],0)[0],np.std(stimrates[cind],0)[0],np.std(snrs[cind],0)[0]]
	
	tstat = [tstat_PN,tstat_stimPN,tstat_snrPN]
	p = [pval_PN,pval_stimPN,pval_snrPN]
	cd = [cd_PN,cd_stimPN,cd_snrPN]
	
	tstat_0 = [tstat_PN_0,tstat_stimPN_0,tstat_snrPN_0]
	p_0 = [pval_PN_0,pval_stimPN_0,pval_snrPN_0]
	cd_0 = [cd_PN_0,cd_stimPN_0,cd_snrPN_0]
	
	# vs Healthy
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels[-1] + ' vs. ' + x_labels[cind],
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
		df = df.append({"Comparison" : x_labels[0] + ' vs. ' + x_labels[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : m1[dfi],
					"Group1 SD" : sd1[dfi],
					"Group2 Mean" : m2[dfi],
					"Group2 SD" : sd2[dfi],
					"t-stat" : tstat[dfi],
					"p-value" : p[dfi],
					"Cohen's d" : cd[dfi]},
					ignore_index = True)
	i1 = np.max(np.transpose(rates[0])[0])
	i2 = np.max(np.transpose(rates[cind])[0])
	maxPNrate = np.max([i1, i2])+0.2#+(cind-1)/20
	i1 = np.max(np.transpose(stimrates[0])[0])
	i2 = np.max(np.transpose(stimrates[cind])[0])
	maxPNstimrate = np.max([i1, i2])+0.2#+(cind-1)/20
	i1 = np.max(np.transpose(snrs[0])[0])
	i2 = np.max(np.transpose(snrs[cind])[0])
	maxPNsnr = np.max([i1, i2])-0.5#+(cind-1)
	i1 = np.max(np.transpose(percentsilent[0])[0])
	i2 = np.max(np.transpose(percentsilent[cind])[0])
	
	# baseline
	if cind < len(conds[:-1]):
		ax_rates.bar(x[cind][1:],height=meanRates[1:],
			   yerr=stdevRates[1:],    # error bars
			   capsize=8, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			  )
		ax_rates_PN.bar(x_PN[cind],height=meanRates[0],
			   yerr=stdevRates[0],    # error bars
			   capsize=12, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			   )
		if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)):
			ax_rates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.3,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)):
			ax_rates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.3,'*',c=colors_conds[0],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
			# barplot_annotate_brackets(fig_rates_PN, ax_rates_PN, cind-1, cind, '*', x_PN, [maxPNrate for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
		
		# stimulation
		ax_stimrates.bar(x[cind][1:],height=meanStimRates[1:],
			   yerr=stdevStimRates[1:],    # error bars
			   capsize=8, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			  )
		ax_stimrates_PN.bar(x_PN[cind],height=meanRates[0],
			   yerr=stdevRates[0],    # error bars
			   capsize=12, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			   )
		ax_stimrates_PN.bar(x_PN[cind]+8,height=meanStimRates[0],
			   yerr=stdevStimRates[0],    # error bars
			   capsize=12, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			   )
		if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)):
			ax_stimrates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.3,'*',c='k',va='bottom' if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pval_PN < p_thresh) & (abs(cd_PN) > c_thresh)):
			ax_stimrates_PN.text(x_PN[cind],meanRates[0]+stdevRates[0]+0.3,'*',c=colors_conds[0],va='top' if ((pval_PN_0 < p_thresh) & (abs(cd_PN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		# if ((pval_stimPN_0 < 0.05) & (abs(cd_stimPN_0) > 0.5)):
		# 	ax_stimrates_PN.text(x_PN[cind]+8,meanStimRates[0]+stdevStimRates[0]+0.1,'*',c='k',ha='right' if ((pval_stimPN < 0.05) & (abs(cd_stimPN) > 0.5)) else 'center', va='bottom')
		# if ((pval_stimPN < 0.05) & (abs(cd_stimPN) > 0.5)):
		# 	ax_stimrates_PN.text(x_PN[cind]+8,meanStimRates[0]+stdevStimRates[0]+0.1,'*',c=colors_conds[0],ha='left' if ((pval_stimPN_0 < 0.05) & (abs(cd_stimPN_0) > 0.5)) else 'center', va='bottom')
			# barplot_annotate_brackets(fig_stimrates_PN, ax_stimrates_PN, cind-1, cind, '*', x_PN, [maxPNrate for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
		
		# snr
		ax_snr.bar(x[cind][1:],height=meanSNRs[1:],
			   yerr=stdevSNRs[1:],    # error bars
			   capsize=8, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			  )
		ax_snr_PN.bar(x_PN[cind],height=meanSNRs[0],
			   yerr=stdevSNRs[0],    # error bars
			   capsize=12, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			   )
		if ((pval_snrPN_0 < p_thresh) & (abs(cd_snrPN_0) > c_thresh)):
			ax_snr_PN.text(x_PN[cind],meanSNRs[0]+stdevSNRs[0]+0.6,'*',c='k',va='bottom' if ((pval_snrPN < p_thresh) & (abs(cd_snrPN) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pval_snrPN < p_thresh) & (abs(cd_snrPN) > c_thresh)):
			ax_snr_PN.text(x_PN[cind],meanSNRs[0]+stdevSNRs[0]+0.6,'*',c=colors_conds[0],va='top' if ((pval_snrPN_0 < p_thresh) & (abs(cd_snrPN_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
			# barplot_annotate_brackets(fig_snr_PN, ax_snr_PN, cind-1, cind, '*', x_PN, [maxPNsnr for _ in range(0,len(conds))], yerr=[0.01 for _ in range(0,len(conds))])
		
		# percent silent
		ax_percentsilent.bar(x[cind][1:],height=meanSilent[1:],
			   yerr=stdevSilent[1:],    # error bars
			   capsize=8, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			  )
		ax_percentsilent_PN.bar(x_PN[cind],height=meanSilent[0],
			   yerr=stdevSilent[0],    # error bars
			   capsize=12, # error bar cap width in points
			   width=1,    # bar width
			   color=colors_conds[cind],  # face color transparent
			   edgecolor='k',
			   ecolor='black',
			   alpha=alphas_conds[cind],
			   linewidth=1,
			   error_kw={'elinewidth':3,'markeredgewidth':3}
			  )
		if ((pval_PNS_0 < p_thresh) & (abs(cd_PNS_0) > c_thresh)):
			ax_percentsilent_PN.text(x_PN[cind],meanSilent[0]+stdevSilent[0]+0.3,'*',c='k',va='bottom' if ((pval_PNS < p_thresh) & (abs(cd_PNS) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		if ((pval_PNS < p_thresh) & (abs(cd_PNS) > c_thresh)):
			ax_percentsilent_PN.text(x_PN[cind],meanSilent[0]+stdevSilent[0]+0.3,'*',c=colors_conds[0],va='top' if ((pval_PNS_0 < p_thresh) & (abs(cd_PNS_0) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
			# barplot_annotate_brackets(fig_percentsilent_PN, ax_percentsilent_PN, cind-1, cind, '*', x_PN, [maxPNpercentsilent for _ in range(0,len(conds))], yerr=[0.05 for _ in range(0,len(conds))])
		
	else:
		for sind in range(1,len(x[cind])):
			ax_rates.plot([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5],[meanRates[sind],meanRates[sind]],'k',ls='dashed',alpha=1)
			ax_rates.fill_between([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5], y1 = [meanRates[sind]+stdevRates[sind],meanRates[sind]+stdevRates[sind]], y2 = [meanRates[sind]-stdevRates[sind],meanRates[sind]-stdevRates[sind]], color='k', alpha=0.2, zorder=2)
			
			ax_stimrates.plot([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5],[meanStimRates[sind],meanStimRates[sind]],'k',ls='dashed',alpha=1)
			ax_stimrates.fill_between([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5], y1 = [meanStimRates[sind]+stdevStimRates[sind],meanStimRates[sind]+stdevStimRates[sind]], y2 = [meanStimRates[sind]-stdevStimRates[sind],meanStimRates[sind]-stdevStimRates[sind]], color='k', alpha=0.2, zorder=2)
			
			ax_snr.plot([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5],[meanSNRs[sind],meanSNRs[sind]],'k',ls='dashed',alpha=1)
			ax_snr.fill_between([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5], y1 = [meanSNRs[sind]+stdevSNRs[sind],meanSNRs[sind]+stdevSNRs[sind]], y2 = [meanSNRs[sind]-stdevSNRs[sind],meanSNRs[sind]-stdevSNRs[sind]], color='k', alpha=0.2, zorder=2)
			
			ax_percentsilent.plot([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5],[meanSilent[sind],meanSilent[sind]],'k',ls='dashed',alpha=1)
			ax_percentsilent.fill_between([x[cind][sind]-len(conds[:-1])-0.5, x[cind][sind]-0.5], y1 = [meanSilent[sind]+stdevSilent[sind],meanSilent[sind]+stdevSilent[sind]], y2 = [meanSilent[sind]-stdevSilent[sind],meanSilent[sind]-stdevSilent[sind]], color='k', alpha=0.2, zorder=2)
		
		ax_rates_PN.plot([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5],[meanRates[0],meanRates[0]],'k',ls='dashed',alpha=1)
		ax_rates_PN.fill_between([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5], y1 = [meanRates[0]+stdevRates[0],meanRates[0]+stdevRates[0]], y2 = [meanRates[0]-stdevRates[0],meanRates[0]-stdevRates[0]], color='k', alpha=0.2, zorder=2)
		
		ax_stimrates_PN.plot([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5],[meanRates[0],meanRates[0]],'k',ls='dashed',alpha=1)
		ax_stimrates_PN.plot([x_PN[cind]+8-len(conds[:-1])-0.5, x_PN[cind]+8-0.5],[meanStimRates[0],meanStimRates[0]],'k',ls='dashed',alpha=1)
		ax_stimrates_PN.fill_between([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5], y1 = [meanRates[0]+stdevRates[0],meanRates[0]+stdevRates[0]], y2 = [meanRates[0]-stdevRates[0],meanRates[0]-stdevRates[0]], color='k', alpha=0.2, zorder=2)
		ax_stimrates_PN.fill_between([x_PN[cind]+8-len(conds[:-1])-0.5, x_PN[cind]+8-0.5], y1 = [meanStimRates[0]+stdevStimRates[0],meanStimRates[0]+stdevStimRates[0]], y2 = [meanStimRates[0]-stdevStimRates[0],meanStimRates[0]-stdevStimRates[0]], color='k', alpha=0.2, zorder=2)
		
		ax_snr_PN.plot([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5],[meanSNRs[0],meanSNRs[0]],'k',ls='dashed',alpha=1)
		ax_snr_PN.fill_between([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5], y1 = [meanSNRs[0]+stdevSNRs[0],meanSNRs[0]+stdevSNRs[0]], y2 = [meanSNRs[0]-stdevSNRs[0],meanSNRs[0]-stdevSNRs[0]], color='k', alpha=0.2, zorder=2)
		
		ax_percentsilent_PN.plot([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5],[meanSilent[0],meanSilent[0]],'k',ls='dashed',alpha=1)
		ax_percentsilent_PN.fill_between([x_PN[cind]-len(conds[:-1])-0.5, x_PN[cind]-0.5], y1 = [meanSilent[0]+stdevSilent[0],meanSilent[0]+stdevSilent[0]], y2 = [meanSilent[0]-stdevSilent[0],meanSilent[0]-stdevSilent[0]], color='k', alpha=0.2, zorder=2)




df.to_csv('figsV1_doseResponse/stats_Rates.csv')

ax_rates.set_ylabel('Baseline Rate')
ax_rates.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i - 1) for i in range(1,len(x_labels_types))])
ax_rates.set_xticklabels(x_labels_types[1:])
ax_rates.grid(False)
ax_rates.spines['right'].set_visible(False)
ax_rates.spines['top'].set_visible(False)
fig_rates.tight_layout()
fig_rates.savefig('figsV1_doseResponse/Rates.png',dpi=300,transparent=True)

ax_rates_PN.set_ylabel('Baseline (Hz)')
ax_rates_PN.set_xticks(x_PN[:-1])
ax_rates_PN.set_xlim(-1.5,7.5)
ax_rates_PN.set_xticklabels(x_labels[:-1], rotation = 45, ha="center")
ax_rates_PN.grid(False)
ax_rates_PN.spines['right'].set_visible(False)
ax_rates_PN.spines['top'].set_visible(False)
fig_rates_PN.tight_layout()
fig_rates_PN.savefig('figsV1_doseResponse/Rates_PN.png',dpi=300,transparent=True)

ax_stimrates.set_ylabel('Response Rate')
ax_stimrates.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i - 1) for i in range(1,len(x_labels_types))])
ax_stimrates.set_xticklabels(x_labels_types[1:])
ax_stimrates.grid(False)
ax_stimrates.spines['right'].set_visible(False)
ax_stimrates.spines['top'].set_visible(False)
fig_stimrates.tight_layout()
fig_stimrates.savefig('figsV1_doseResponse/StimRates.png',dpi=300,transparent=True)

ax_stimrates_PN.set_ylabel('Pyr Firing Rate (Hz)')
ax_stimrates_PN.set_xticks(x_PN2)
ax_stimrates_PN.set_xlim(-1.6,15.6)
ax_stimrates_PN.set_xticklabels(['MDD','25%','50%','75%','100%','125%','150%','MDD','25%','50%','75%','100%','125%','150%'], rotation = 45, ha="center")
ax_stimrates_PN.grid(False)
ax_stimrates_PN.spines['right'].set_visible(False)
ax_stimrates_PN.spines['top'].set_visible(False)
fig_stimrates_PN.tight_layout()
fig_stimrates_PN.savefig('figsV1_doseResponse/StimRates_PN.png',dpi=300,transparent=True)

ax_snr.set_ylabel('SNR')
ax_snr.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i - 1) for i in range(1,len(x_labels_types))])
ax_snr.set_xticklabels(x_labels_types[1:])
ax_snr.grid(False)
ax_snr.spines['right'].set_visible(False)
ax_snr.spines['top'].set_visible(False)
fig_snr.tight_layout()
fig_snr.savefig('figsV1_doseResponse/SNR.png',dpi=300,transparent=True)

ax_snr_PN.set_ylabel('SNR')
ax_snr_PN.set_xticks(x_PN[:-1])
ax_snr_PN.set_xlim(-1.5,7.5)
ax_snr_PN.set_ylim(0,8.5)
ax_snr_PN.set_xticklabels(x_labels[:-1], rotation = 45, ha="center")
ax_snr_PN.grid(False)
ax_snr_PN.spines['right'].set_visible(False)
ax_snr_PN.spines['top'].set_visible(False)
fig_snr_PN.tight_layout()
fig_snr_PN.savefig('figsV1_doseResponse/SNR_PN.png',dpi=300,transparent=True)

ax_percentsilent.set_ylabel('% Silent')
ax_percentsilent.set_xticks([int(x_types[i]*len(x)+(len(x))/2 + i - 1) for i in range(1,len(x_labels_types))])
ax_percentsilent.set_xticklabels(x_labels_types[1:])
ax_percentsilent.grid(False)
ax_percentsilent.spines['right'].set_visible(False)
ax_percentsilent.spines['top'].set_visible(False)
fig_percentsilent.tight_layout()
fig_percentsilent.savefig('figsV1_doseResponse/PercentSilent.png',dpi=300,transparent=True)

ax_percentsilent_PN.set_ylabel('Pyr % Silent')
ax_percentsilent_PN.set_xticks(x_PN[:-1])
ax_percentsilent_PN.set_xlim(-1.5,7.5)
ax_percentsilent_PN.set_xticklabels(x_labels[:-1], rotation = 45, ha="center")
ax_percentsilent_PN.grid(False)
ax_percentsilent_PN.spines['right'].set_visible(False)
ax_percentsilent_PN.spines['top'].set_visible(False)
fig_percentsilent_PN.tight_layout()
fig_percentsilent_PN.savefig('figsV1_doseResponse/PercentSilent_PN.png',dpi=300,transparent=True)
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
	
	ax.plot(f_PN[f1:f2], CI_means_PN[f1:f2], colors_conds[cind],alpha=alphas_conds[cind])
	ax.fill_between(f_PN[f1:f2], CI_lower_PN[f1:f2], CI_upper_PN[f1:f2],color=colors_conds[cind],alpha=alphas_conds[cind]-0.1)
	ax.tick_params(axis='x', which='major', bottom=True)
	ax.tick_params(axis='y', which='major', left=True)
	inset.plot(f_PN[:f3], CI_means_PN[:f3], colors_conds[cind],alpha=alphas_conds[cind])
	inset.set_xscale('log')
	inset.set_yscale('log')
	inset.tick_params(axis='x', which='major', bottom=True)
	inset.tick_params(axis='y', which='major', left=True)
	inset.tick_params(axis='x', which='minor', bottom=True)
	inset.tick_params(axis='y', which='minor', left=True)
	inset.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
	inset.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
	inset.fill_between(f_PN[:f3], CI_lower_PN[:f3], CI_upper_PN[:f3],color=colors_conds[cind],alpha=alphas_conds[cind]-0.1)
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

fig.savefig('figsV1_doseResponse/Spikes_PSD_Boot95CI_PN.png',bbox_inches='tight',dpi=300, transparent=True)
plt.close()
