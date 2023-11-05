################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as mticker
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
from scipy.optimize import curve_fit
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import math as math
import scipy.special as sp
import LFPy
import pandas as pd

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def skewnorm(x, sigmag, mu, alpha,a):
	#normal distribution
	normpdf = (1/(sigmag*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigmag,2))))
	normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigmag))/(np.sqrt(2)))))
	return 2*a*normpdf*normcdf

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

def barplot_annotate_brackets(fig, ax, num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None, barcolor = 'black'):
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
	mid = ((lx+rx)/2, y+barh*(1/8))
	
	ax.plot(barx, bary, c=barcolor)
	
	kwargs = dict(ha='center', va='bottom', c=barcolor, fontweight='bold')
	if fs is not None:
		kwargs['fontsize'] = fs
	
	ax.text(*mid, text, **kwargs)

N_seeds = 200
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000.
N_HL23PN = int(0.8*N_cells)
N_HL23MN = int(0.05*N_cells)
N_HL23BN = int(0.07*N_cells)
N_HL23VN = int(0.08*N_cells)
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
tstop = 4500 # ms
response_window = 50 # ms
startsclice = 1000 # ms
stimtime = 4000 # ms
rate_window = 5.000001 # Hz
ratebins = np.arange(0,rate_window,0.2)
bin_centres_highres = np.arange(0.001,rate_window,0.001)

conds = ['healthy','MDD','MDD_a5PAM','MDD_benzo']
paths = ['Saved_SpikesOnly/' + i + '/' for i in conds]

x_labels = ['Healthy','MDD','MDD\n'+r'$\alpha$'+'5-PAM','MDD\nBenzo.']
x_labels_types = ['Pyr','SST','PV','VIP']

colors_neurs = ['dimgrey', 'red', 'green', 'orange']
colors_conds = ['tab:gray', 'tab:purple', 'dodgerblue', 'chocolate']
colors_conds2 = ['k', 'tab:purple', 'dodgerblue', 'chocolate']

rates_fits = [[] for _ in conds]
stimrates = [[] for _ in conds]

for seed in N_seedsList:
	for cind, path in enumerate(paths):
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		# Load outputs
		temp_s = np.load(path + 'SPIKES_Seed' + str(seed) + '.npy',allow_pickle=True)
		
		# PSTH to generate train of binned rates for each seed/condition
		SPIKES1 = [x for _,x in sorted(zip(temp_s.item()['gids'][0],temp_s.item()['times'][0]))]
		popspikes = np.concatenate(SPIKES1).ravel()
		
		spikeratevec = []
		for slide in np.arange(0,response_window,1):
			pre_bins = np.arange(startsclice+slide,stimtime+dt,response_window)
			spikeratevec.extend((np.histogram(popspikes,bins=pre_bins)[0]/(response_window/1000))/N_HL23PN)
		
		post_bins = np.arange(stimtime+5,stimtime+response_window+5+dt,response_window)
		spikeratevec_stim = (np.histogram(popspikes,bins=post_bins)[0]/(response_window/1000))/N_HL23PN
		
		# Fit baseline spike rate distribution to Gaussian
		s_hist, s_bins = np.histogram(spikeratevec,bins = ratebins, density=True)
		bin_centres = (s_bins[:-1] + s_bins[1:])/2
		p0 = [np.std(s_hist), np.mean(s_hist), 1, 1]
		coeff, var_matrix = curve_fit(skewnorm, bin_centres, s_hist, p0 = p0, bounds = [0,(np.std(s_hist)*5,np.mean(s_hist)*10,2,200)], maxfev = 100000000)
		hist_fit = skewnorm(bin_centres_highres, *coeff)
		
		plt.plot(bin_centres, s_hist, label='Test data')
		plt.plot(bin_centres_highres, hist_fit, label='Fitted data')
		plt.savefig('figs_ratedistributions_benzo/'+conds[cind]+str(seed)+'_Pre.png',dpi=300,transparent=True)
		plt.close()
		
		rates_fits[cind].append(hist_fit)
		stimrates[cind].append(spikeratevec_stim[0])

# Fit baseline spike rate distribution to Gaussian
hist_fit_stim = [[] for _ in conds]
failed_detections = [[] for _ in conds]
false_detections = [[] for _ in conds]
for cind, sr in enumerate(stimrates):
	s_hist, s_bins = np.histogram(sr,bins = ratebins, density=True)
	bin_centres_stim = (s_bins[:-1] + s_bins[1:])/2
	p0 = [np.std(s_hist), np.mean(s_hist), 1, 1]
	coeff, var_matrix = curve_fit(skewnorm, bin_centres, s_hist, p0=p0, bounds = [0,(np.std(s_hist)*5,np.mean(s_hist)*10,2,200)], maxfev = 100000000)
	post_fit = skewnorm(bin_centres_highres, *coeff)
	hist_fit_stim[cind] = post_fit
	
	plt.plot(bin_centres_stim, s_hist, label='Test data')
	plt.plot(bin_centres_highres, post_fit, label='Fitted data')
	plt.savefig('figs_ratedistributions_benzo/'+conds[cind]+'_Post.png',dpi=300,transparent=True)
	plt.close()
	
	for pre_fit in rates_fits[cind]:
		idx = np.argwhere(np.diff(np.sign(pre_fit - post_fit))).flatten()
		idx = idx[np.where(np.logical_and(idx>1100, idx<=2200))][0]
		failed_detections[cind].append((np.trapz(post_fit[:idx])/np.trapz(post_fit))*100)
		false_detections[cind].append((np.trapz(pre_fit[idx:])/np.trapz(pre_fit))*100)

failed_d_m = []
failed_d_sd_l = []
failed_d_sd_u = []
false_d_m = []
false_d_sd_l = []
false_d_sd_u = []
bootci = False
if bootci:
	for v in failed_detections:
		x = bs.bootstrap(np.array(v), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		failed_d_m.append(x.value)
		failed_d_sd_l.append(x.value-x.lower_bound)
		failed_d_sd_u.append(x.upper_bound-x.value)

	for v in false_detections:
		x = bs.bootstrap(np.array(v), stat_func=bs_stats.mean, alpha=0.05, num_iterations=500)
		false_d_m.append(x.value)
		false_d_sd_l.append(x.value-x.lower_bound)
		false_d_sd_u.append(x.upper_bound-x.value)
else:
	for v in failed_detections:
		failed_d_m.append(np.mean(np.array(v)))
		failed_d_sd_l.append(np.std(np.array(v)))
		failed_d_sd_u.append(np.std(np.array(v)))
	for v in false_detections:
		false_d_m.append(np.mean(np.array(v)))
		false_d_sd_l.append(np.std(np.array(v)))
		false_d_sd_u.append(np.std(np.array(v)))

data_m = np.transpose([failed_d_m,false_d_m])
data_sd_l = np.transpose([failed_d_sd_l,false_d_sd_l])
data_sd_u = np.transpose([failed_d_sd_u,false_d_sd_u])

tstats_failed = []
pvals_failed = []
cd_failed = []
for f in failed_detections[0:-1]:
	t0, p0 = st.ttest_rel(f,failed_detections[-1])
	tstats_failed.append(t0)
	pvals_failed.append(p0)
	c0 = cohen_d(f,failed_detections[-1])
	cd_failed.append(c0)
tstats_false = []
pvals_false = []
cd_false = []
for f in false_detections[0:-1]:
	t0, p0 = st.ttest_rel(f,false_detections[-1])
	tstats_false.append(t0)
	pvals_false.append(p0)
	c0 = cohen_d(f,false_detections[-1])
	cd_false.append(c0)

tstat = [[t1,t2] for t1,t2 in zip(tstats_failed,tstats_false)]
p = [[p1,p2] for p1,p2 in zip(pvals_failed,pvals_false)]
cd = [[c1,c2] for c1,c2 in zip(cd_failed,cd_false)]
metrics = ["Failed Detections","False Detections"]
df = pd.DataFrame(columns=["Comparison",
			"Metric",
			"Group1 Mean",
			"Group1 95% CI",
			"Group2 Mean",
			"Group2 95% CI",
			"t-stat",
			"p-value",
			"Cohen's d"])

for cind in range(0,len(conds)-1):
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels[cind] + ' vs. ' + x_labels[-1],
					"Metric" : metrics[dfi],
					"Group1 Mean" : data_m[cind][dfi],
					"Group1 95% CI" : (data_m[cind][dfi]-data_sd_l[cind][dfi],data_m[cind][dfi]+data_sd_u[cind][dfi]),
					"Group2 Mean" : data_m[-1][dfi],
					"Group2 95% CI" : (data_m[-1][dfi]-data_sd_l[-1][dfi],data_m[-1][dfi]+data_sd_u[-1][dfi]),
					"t-stat" : tstat[cind][dfi],
					"p-value" : p[cind][dfi],
					"Cohen's d" : cd[cind][dfi]},
					ignore_index = True)

df.to_csv('figs_ratedistributions_benzo/stats_FailedFalse.csv')

x = [0.,5]
width = 1.
fig_det, ax_det = plt.subplots(figsize=(7, 6))
for cind,c in enumerate(conds):
	if not len(conds) % 2:
		x2 = [xr+cind*width+width/2 for xr in x]
		x3 = [0.5,1.5,2.5,3.5]
	else:
		x2 = [xr+cind*width for xr in x]
		x3 = [0,1,2,3]
	
	ax_det.bar(x2,height=data_m[cind],
		   width=width,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   label=x_labels[cind],
		   edgecolor='k',
		   linewidth=1,
		  )
	ax_det.errorbar(x2,data_m[cind],
			yerr=(data_sd_l[cind],data_sd_u[cind]),    # error bars
			capsize=8, # error bar cap width in points
			ecolor='black',
			linestyle='',
			elinewidth=3,
			capthick=3
			)

p_thresh = 0.05
c_thresh = 1.

pi1 = 0
pi2 = 0
for cind in range(0,len(conds)-1):
	if ((p[cind][0] < p_thresh) & (abs(cd[cind][0]) > c_thresh)):
		barplot_annotate_brackets(fig_det, ax_det, cind, len(conds)-1, '*', x3, [11+pi1*1.4 for _ in data_m], yerr= [0.25 for _ in data_sd_u], barcolor=colors_conds2[cind])
		pi1 += 1
	if ((p[cind][1] < p_thresh) & (abs(cd[cind][1]) > c_thresh)):
		barplot_annotate_brackets(fig_det, ax_det, cind, len(conds)-1, '*', [xx+x[1] for xx in x3], [12+pi2*1.4 for _ in data_m], yerr=[1. for _ in data_sd_u], barcolor=colors_conds2[cind])
		pi2 += 1

# ax_det.legend(loc='upper left')
ax_det.set_ylabel('Probability (%)')
ax_det.set_xticks([xx+len(conds)/2 for xx in x])
ax_det.set_xticklabels(['Failed\nDetection','False\nDetection'])
ax_det.set_xlim(-1.1,10.1)
ax_det.grid(False)
ax_det.spines['right'].set_visible(False)
ax_det.spines['top'].set_visible(False)
fig_det.tight_layout()
fig_det.savefig('figs_ratedistributions_benzo/DetectionProbability.png',dpi=300,transparent=True)
plt.close()

# Average baseline fits
hist_fit_base = [[] for _ in conds]
for cind, br in enumerate(rates_fits):
	hist_fit_base[cind] = np.mean(br,axis=0)

# Plot
cind = 0
fig_ov, ax_ov = plt.subplots(figsize=(8, 8))
for br, sr in zip(hist_fit_base,hist_fit_stim):
	idx = np.argwhere(np.diff(np.sign(br - sr))).flatten()
	idx = idx[np.where(np.logical_and(idx>1100, idx<=2200))][0]
	ax_ov.plot(bin_centres_highres, br, label='Pre', color = colors_conds[cind], ls='dashed', alpha = 0.5)
	ax_ov.plot(bin_centres_highres, sr, label='Post', color = colors_conds[cind], alpha = 0.8)
	ax_ov.fill_between(bin_centres_highres[:idx],sr[:idx], color = colors_conds[cind], alpha = 0.5)
	ax_ov.fill_between(bin_centres_highres[idx:],br[idx:], color = colors_conds[cind], alpha = 0.8)
	ax_ov.set_ylim(bottom=0)
	ylims = ax_ov.get_ylim()
	ax_ov.plot([bin_centres_highres[idx],bin_centres_highres[idx]], ylims, color = colors_conds[cind], linewidth=2, ls='dotted', alpha=0.3)
	ax_ov.set_ylim(ylims)
	cind += 1
ax_ov.set_xlabel('Spike Rate (Hz)')
ax_ov.set_ylabel('Proportion')
ax_ov.set_xlim(0,bin_centres_highres[-1])
fig_ov.tight_layout()
fig_ov.savefig('figs_ratedistributions_benzo/RateDistribution.png',dpi=300,transparent=True)
plt.close()

x_labels2 = ['Healthy','MDD','MDD + '+r'$\alpha$'+'5-PAM','MDD + Benzodiazepine']
cind = 0
for br, sr in zip(hist_fit_base,hist_fit_stim):
	fig_ov, ax_ov = plt.subplots(figsize=(7, 6))
	idx = np.argwhere(np.diff(np.sign(br - sr))).flatten()
	idx = idx[np.where(np.logical_and(idx>1100, idx<=2200))][0]
	ax_ov.plot(bin_centres_highres, br, label='Pre', color = colors_conds[cind], ls='dashed', alpha = 0.5, linewidth=3)
	ax_ov.plot(bin_centres_highres, sr, label='Post', color = colors_conds[cind], alpha = 0.8, linewidth=3)
	ax_ov.fill_between(bin_centres_highres[:idx],sr[:idx], color = colors_conds[cind], alpha = 0.5)
	ax_ov.fill_between(bin_centres_highres[idx:],br[idx:], color = colors_conds[cind], alpha = 0.8)
	ax_ov.set_ylim(ylims)
	ylims = ax_ov.get_ylim()
	ax_ov.plot([bin_centres_highres[idx],bin_centres_highres[idx]], ylims, color = 'k', linewidth=2, ls='dotted', alpha=0.8)
	ax_ov.set_ylim(ylims)
	ax_ov.set_xlabel('Spike Rate (Hz)')
	ax_ov.set_ylabel('Proportion')
	ax_ov.set_xlim(0,bin_centres_highres[-1])
	ax_ov.set_title(x_labels2[cind])
	fig_ov.tight_layout()
	fig_ov.savefig('figs_ratedistributions_benzo/RateDistribution_'+conds[cind]+'.png',dpi=300,transparent=True)
	plt.close()
	cind += 1
