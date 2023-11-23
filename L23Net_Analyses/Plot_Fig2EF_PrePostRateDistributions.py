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
fsize = 30

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

conds = ['healthy','MDD','MDD_a5PAM']
paths = ['Saved_SpikesOnly/' + i + '/' for i in conds]

x_labels = ['Healthy','MDD','MDD\n'+r'$\alpha$'+'5-PAM']
x_labels_types = ['Pyr','SST','PV','VIP']

colors_neurs = ['dimgrey', 'red', 'green', 'orange']
colors_conds = ['tab:gray', 'tab:purple', 'dodgerblue']

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
		plt.savefig('figs_ratedistributions/'+conds[cind]+str(seed)+'_Pre.png',dpi=300,transparent=True)
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
	plt.savefig('figs_ratedistributions/'+conds[cind]+'_Post.png',dpi=300,transparent=True)
	plt.close()
	
	for pre_fit in rates_fits[cind]:
		idx = np.argwhere(np.diff(np.sign(pre_fit - post_fit))).flatten()
		idx = idx[np.where(np.logical_and(idx>1100, idx<=2000))][0]
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

# vs healthy
tstats_failed0 = []
pvals_failed0 = []
cd_failed0 = []
for f in failed_detections:
	t_0, p_0 = st.ttest_rel(failed_detections[0],f)
	tstats_failed0.append(t_0)
	pvals_failed0.append(p_0)
	c_0 = cohen_d(failed_detections[0],f)
	cd_failed0.append(c_0)
tstats_false0 = []
pvals_false0 = []
cd_false0 = []
for f in false_detections:
	t_0, p_0 = st.ttest_rel(false_detections[0],f)
	tstats_false0.append(t_0)
	pvals_false0.append(p_0)
	c_0 = cohen_d(false_detections[0],f)
	cd_false0.append(c_0)

tstat0 = [[t1,t2] for t1,t2 in zip(tstats_failed0,tstats_false0)]
p0 = [[p1,p2] for p1,p2 in zip(pvals_failed0,pvals_false0)]
cd0 = [[c1,c2] for c1,c2 in zip(cd_failed0,cd_false0)]

# vs MDD
tstats_failed = []
pvals_failed = []
cd_failed = []
for f in failed_detections:
	t_0, p_0 = st.ttest_rel(failed_detections[1],f)
	tstats_failed.append(t_0)
	pvals_failed.append(p_0)
	c_0 = cohen_d(failed_detections[1],f)
	cd_failed.append(c_0)
tstats_false = []
pvals_false = []
cd_false = []
for f in false_detections:
	t_0, p_0 = st.ttest_rel(false_detections[1],f)
	tstats_false.append(t_0)
	pvals_false.append(p_0)
	c_0 = cohen_d(false_detections[1],f)
	cd_false.append(c_0)

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

x_labels2 = ['Healthy','MDD','MDD + a5-PAM']

for cind in range(0,len(conds)):
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels2[1] + ' vs. ' + x_labels2[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : data_m[1][dfi],
					"Group1 95% CI" : (data_m[1][dfi]-data_sd_l[1][dfi],data_m[1][dfi]+data_sd_u[1][dfi]),
					"Group2 Mean" : data_m[cind][dfi],
					"Group2 95% CI" : (data_m[cind][dfi]-data_sd_l[cind][dfi],data_m[cind][dfi]+data_sd_u[cind][dfi]),
					"t-stat (vs MDD)" : tstat[cind][dfi],
					"p-value (vs MDD)" : p[cind][dfi],
					"Cohen's d (vs MDD)" : cd[cind][dfi]},
					ignore_index = True)
		df = df.append({"Comparison" : x_labels2[0] + ' vs. ' + x_labels2[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : data_m[0][dfi],
					"Group1 95% CI" : (data_m[0][dfi]-data_sd_l[0][dfi],data_m[0][dfi]+data_sd_u[0][dfi]),
					"Group2 Mean" : data_m[cind][dfi],
					"Group2 95% CI" : (data_m[cind][dfi]-data_sd_l[cind][dfi],data_m[cind][dfi]+data_sd_u[cind][dfi]),
					"t-stat (vs MDD)" : tstat0[cind][dfi],
					"p-value (vs MDD)" : p0[cind][dfi],
					"Cohen's d (vs MDD)" : cd0[cind][dfi]},
					ignore_index = True)

df.to_csv('figs_ratedistributions/stats_FailedFalse.csv')

x = [0.,4.]
width = 1.
fig_det, ax_det = plt.subplots(figsize=(7, 6))
fig_det_bp, ax_det_bp = plt.subplots(figsize=(7, 6))
for cind,c in enumerate(conds):
	if not len(conds) % 2:
		x2 = [xr+cind*width+width/2 for xr in x]
		x3 = [n+0.5 for n in range(len(conds))]
	else:
		x2 = [xr+cind*width for xr in x]
		x3 = [n for n in range(len(conds))]
	
	x2 = [xr+cind*width for xr in x]
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
	ax_det_bp.boxplot([failed_detections[cind],false_detections[cind]],positions=x2,
		   boxprops=dict(color=colors_conds[cind],linewidth=3),
		   capprops=dict(color=colors_conds[cind],linewidth=3),
		   whiskerprops=dict(color=colors_conds[cind],linewidth=3),
		   flierprops=dict(color=colors_conds[cind], markeredgecolor=colors_conds[cind],linewidth=3),
		   medianprops=dict(color=colors_conds[cind],linewidth=3),
		   widths=0.9
		  )
p_thresh = 0.05
c_thresh = 1.

for cind in range(0,len(conds)):
	if ((p[cind][0] < p_thresh) & (abs(cd[cind][0]) > c_thresh) & (cind>0)):
		ax_det.text(x3[cind],data_m[cind][0]+data_sd_u[cind][0]+1.,'*',c=colors_conds[1],va='top' if ((p0[cind][0] < p_thresh) & (abs(cd0[cind][0]) >c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_det_bp.text(x3[cind],np.max(failed_detections[cind])+1.,'*',c=colors_conds[1],va='top' if ((p0[cind][0] < p_thresh) & (abs(cd0[cind][0]) >c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((p[cind][1] < p_thresh) & (abs(cd[cind][1]) > c_thresh) & (cind>0)):
		ax_det.text(x3[cind]+x[1],data_m[cind][1]+data_sd_u[cind][1]+1.,'*',c=colors_conds[1],va='top' if ((p0[cind][1] < p_thresh) & (abs(cd0[cind][1]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_det_bp.text(x3[cind]+x[1],np.max(false_detections[cind])+1.,'*',c=colors_conds[1],va='top' if ((p0[cind][1] < p_thresh) & (abs(cd0[cind][1]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((p0[cind][0] < p_thresh) & (abs(cd0[cind][0]) > c_thresh) & (cind>0)):
		ax_det.text(x3[cind],data_m[cind][0]+data_sd_u[cind][0]+1.,'*',c='k',va='bottom' if ((p[cind][0] < p_thresh) & (abs(cd[cind][0]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_det_bp.text(x3[cind],np.max(failed_detections[cind])+1.,'*',c='k',va='bottom' if ((p[cind][0] < p_thresh) & (abs(cd[cind][0]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((p0[cind][1] < p_thresh) & (abs(cd0[cind][1]) > c_thresh) & (cind>0)):
		ax_det.text(x3[cind]+x[1],data_m[cind][1]+data_sd_u[cind][1]+1.,'*',c='k',va='bottom' if ((p[cind][1] < p_thresh) & (abs(cd[cind][1]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
		ax_det_bp.text(x3[cind]+x[1],np.max(false_detections[cind])+1.,'*',c='k',va='bottom' if ((p[cind][1] < p_thresh) & (abs(cd[cind][1]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)

# if (pval_failed_HvsM < 0.05):
# 	barplot_annotate_brackets(fig_det, ax_det, 0, 1, '*', [0,1], [data_m[0][0],data_m[1][0]], yerr=[data_sd_u[0][0],data_sd_u[1][0]])
# if (pval_failed_MvsD < 0.05):
# 	barplot_annotate_brackets(fig_det, ax_det, 0, 1, '*', [1,2], [data_m[1][0],data_m[2][0]], yerr=[data_sd_u[1][0],data_sd_u[2][0]])
# if (pval_false_HvsM < 0.05):
# 	barplot_annotate_brackets(fig_det, ax_det, 0, 1, '*', [4,5], [data_m[0][1],data_m[1][1]], yerr=[data_sd_u[0][1],data_sd_u[1][1]])
# if (pval_false_MvsD < 0.05):
# 	barplot_annotate_brackets(fig_det, ax_det, 0, 1, '*', [5,6], [data_m[1][1],data_m[2][1]], yerr=[data_sd_u[1][1],data_sd_u[2][1]])

# ax_det.legend(loc='upper left')
ax_det.set_ylabel('Probability (%)')
ax_det.set_xticks([xx+1 for xx in x])
ax_det.set_xlim(-1.1,7.1)
ax_det.set_xticklabels(['Failed\nDetection','False\nDetection'])
ax_det.grid(False)
ax_det.spines['right'].set_visible(False)
ax_det.spines['top'].set_visible(False)
fig_det.tight_layout()
fig_det.savefig('figs_ratedistributions/DetectionProbability.png',dpi=300,transparent=True)

ax_det_bp.set_ylabel('Probability (%)')
ax_det_bp.set_xticks([xx+1 for xx in x])
ax_det_bp.set_xlim(-1.1,7.1)
ax_det_bp.set_xticklabels(['Failed\nDetection','False\nDetection'])
ax_det_bp.grid(False)
ax_det_bp.spines['right'].set_visible(False)
ax_det_bp.spines['top'].set_visible(False)
fig_det_bp.tight_layout()
fig_det_bp.savefig('figs_ratedistributions/DetectionProbability_bp.png',dpi=300,transparent=True)
plt.close()

# Average baseline fits
hist_fit_base = [[] for _ in conds]
for cind, br in enumerate(rates_fits):
	hist_fit_base[cind] = np.mean(br,axis=0)

# Plot
cind = 0
fig_ov, ax_ov = plt.subplots(figsize=(8, 8))
np.save('figs_ratedistributions/post_stimulus_default_circuit.npy',hist_fit_stim[0])
for br, sr in zip(hist_fit_base,hist_fit_stim):
	idx = np.argwhere(np.diff(np.sign(br - sr))).flatten()
	idx = idx[np.where(np.logical_and(idx>1100, idx<=2000))][0]
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
fig_ov.savefig('figs_ratedistributions/RateDistribution.png',dpi=300,transparent=True)
plt.close()

x_labels2 = ['Healthy','MDD','MDD + '+r'$\alpha$'+'5-PAM']
cind = 0
for br, sr in zip(hist_fit_base,hist_fit_stim):
	fig_ov, ax_ov = plt.subplots(figsize=(4.5, 6))
	idx = np.argwhere(np.diff(np.sign(br - sr))).flatten()
	idx = idx[np.where(np.logical_and(idx>1100, idx<=2000))][0]
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
	fig_ov.savefig('figs_ratedistributions/RateDistribution_'+conds[cind]+'.png',dpi=300,transparent=True)
	plt.close()
	cind += 1
