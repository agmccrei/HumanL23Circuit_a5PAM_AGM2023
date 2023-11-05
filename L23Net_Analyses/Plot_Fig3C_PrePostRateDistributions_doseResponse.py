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
fsize = 30

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
			'MDD\n'+r'25% $\alpha$'+'5-PAM',
			'MDD\n'+r'50% $\alpha$'+'5-PAM',
			'MDD\n'+r'75% $\alpha$'+'5-PAM',
			'MDD\n'+r'100% $\alpha$'+'5-PAM',
			'MDD\n'+r'125% $\alpha$'+'5-PAM',
			'MDD\n'+r'150% $\alpha$'+'5-PAM',
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
		plt.savefig('figs_ratedistributions_doseResponse/'+conds[cind]+str(seed)+'_Pre.png',dpi=300,transparent=True)
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
	plt.savefig('figs_ratedistributions_doseResponse/'+conds[cind]+'_Post.png',dpi=300,transparent=True)
	plt.close()
	
	for pre_fit in rates_fits[cind]:
		idx = np.argwhere(np.diff(np.sign(pre_fit - post_fit))).flatten()
		idx = idx[np.where(np.logical_and(idx>1000, idx<=2200))][0]
		failed_detections[cind].append((np.trapz(post_fit[:idx])/np.trapz(post_fit))*100)
		false_detections[cind].append((np.trapz(pre_fit[idx:])/np.trapz(pre_fit))*100)

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

failed_to_fit = [[],[]]
for d,rc in zip(xvals,failed_detections[:-1]):
	for r in rc:
		failed_to_fit[0].append(d)
		failed_to_fit[1].append(r)

false_to_fit = [[],[]]
for d,rc in zip(xvals,false_detections[:-1]):
	for r in rc:
		false_to_fit[0].append(d)
		false_to_fit[1].append(r)

tstat_r,pval_r = st.pearsonr(failed_to_fit[0],failed_to_fit[1])
tstat_s,pval_s = st.pearsonr(false_to_fit[0],false_to_fit[1])

p0_rl = [tstat_r, np.max(failed_to_fit[1])]
p0_re = [1, 1, 1]
p0_rs = [2, 1, 1, 0.75]
p0_sl = [tstat_s, np.max(false_to_fit[1])]
p0_se = [1, 1, 1]
p0_ss = [2, 1, 1, 0.75]

coeff_rl = curve_fit(linear, failed_to_fit[0], failed_to_fit[1], p0 = p0_rl, maxfev = 100000000, full_output=True)
coeff_re = curve_fit(neg_expo, failed_to_fit[0], failed_to_fit[1], p0 = p0_re, maxfev = 100000000, full_output=True)
coeff_rs = curve_fit(rev_sigmoid, failed_to_fit[0], failed_to_fit[1], p0 = p0_rs, maxfev = 200000000, full_output=True)
coeff_sl = curve_fit(linear, false_to_fit[0], false_to_fit[1], p0 = p0_sl, maxfev = 100000000, full_output=True)
coeff_se = curve_fit(neg_expo, false_to_fit[0], false_to_fit[1], p0 = p0_se, maxfev = 100000000, full_output=True)
coeff_ss = curve_fit(rev_sigmoid, false_to_fit[0], false_to_fit[1], p0 = p0_ss, maxfev = 200000000, full_output=True)

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
e_fit_s = neg_expo(xvals_highres, *coeff_se[0])
s_fit_s = rev_sigmoid(xvals_highres, *coeff_ss[0])

fig_dosefits_r, ax_dosefits_r = plt.subplots(figsize=(20, 8))
ax_dosefits_r.scatter(failed_to_fit[0], failed_to_fit[1], c='k', label='Data')
ax_dosefits_r.plot(xvals_highres, l_fit_r, c='r', lw=8, ls=':', label='Linear SSE: '+str(np.round(SSE_rl,3)) + '; R,p = ' + str(np.round(tstat_r,3)) + ',' + str(np.round(pval_r,3)))
print('Failed Linear Pearson Correlation P-value = '+str(pval_r))
ax_dosefits_r.plot(xvals_highres, e_fit_r, c='b', lw=6, ls='--', label='Negative Exponential SSE: '+str(np.round(SSE_re,3)))
ax_dosefits_r.plot(xvals_highres, s_fit_r, c='g', lw=4, ls='-.', label='Reverse Sigmoidal SSE: '+str(np.round(SSE_rs,3)))
ax_dosefits_r.set_xlabel('Dose Mulitplier')
ax_dosefits_r.set_ylabel('Failed Detections (%)')
ax_dosefits_r.legend()
fig_dosefits_r.tight_layout()
fig_dosefits_r.savefig('figs_ratedistributions_doseResponse/DoseFits_FailedDetection.png',dpi=300,transparent=True)
plt.close()

fig_dosefits_s, ax_dosefits_s = plt.subplots(figsize=(20, 8))
ax_dosefits_s.scatter(false_to_fit[0], false_to_fit[1], c='k', label='Data')
ax_dosefits_s.plot(xvals_highres, l_fit_s, c='r', lw=8, ls=':', label='Linear SSE: '+str(np.round(SSE_sl,3)) + '; R,p = ' + str(np.round(tstat_s,3)) + ',' + str(np.round(pval_s,3)))
print('False Linear Pearson Correlation P-value = '+str(pval_s))
ax_dosefits_s.plot(xvals_highres, e_fit_s, c='b', lw=6, ls='--', label='Negative Exponential SSE: '+str(np.round(SSE_se,3)))
ax_dosefits_s.plot(xvals_highres, s_fit_s, c='g', lw=4, ls='-.', label='Reverse Sigmoidal SSE: '+str(np.round(SSE_ss,3)))
ax_dosefits_s.set_xlabel('Dose Mulitplier')
ax_dosefits_s.set_ylabel('False Detections (%)')
ax_dosefits_s.legend()
fig_dosefits_s.tight_layout()
fig_dosefits_s.savefig('figs_ratedistributions_doseResponse/DoseFits_FalseDetection.png',dpi=300,transparent=True)
plt.close()




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
	t_0, p_0 = st.ttest_rel(failed_detections[-1],f)
	tstats_failed0.append(t_0)
	pvals_failed0.append(p_0)
	c_0 = cohen_d(failed_detections[-1],f)
	cd_failed0.append(c_0)
tstats_false0 = []
pvals_false0 = []
cd_false0 = []
for f in false_detections:
	t_0, p_0 = st.ttest_rel(false_detections[-1],f)
	tstats_false0.append(t_0)
	pvals_false0.append(p_0)
	c_0 = cohen_d(false_detections[-1],f)
	cd_false0.append(c_0)

tstat0 = [[t1,t2] for t1,t2 in zip(tstats_failed0,tstats_false0)]
p0 = [[p1,p2] for p1,p2 in zip(pvals_failed0,pvals_false0)]
cd0 = [[c1,c2] for c1,c2 in zip(cd_failed0,cd_false0)]

# vs MDD
tstats_failed = []
pvals_failed = []
cd_failed = []
for f in failed_detections:
	t_0, p_0 = st.ttest_rel(failed_detections[0],f)
	tstats_failed.append(t_0)
	pvals_failed.append(p_0)
	c_0 = cohen_d(failed_detections[0],f)
	cd_failed.append(c_0)
tstats_false = []
pvals_false = []
cd_false = []
for f in false_detections:
	t_0, p_0 = st.ttest_rel(false_detections[0],f)
	tstats_false.append(t_0)
	pvals_false.append(p_0)
	c_0 = cohen_d(false_detections[0],f)
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
			"t-stat (vs MDD)",
			"p-value (vs MDD)",
			"Cohen's d (vs MDD)"])

x_labels2 = ['MDD',
			'MDD + 25% a5-PAM',
			'MDD + 50% a5-PAM',
			'MDD + 75% a5-PAM',
			'MDD + 100% a5-PAM',
			'MDD + 125% a5-PAM',
			'MDD + 150% a5-PAM',
			'Healthy']
for cind in range(0,len(conds)):
	for dfi in range(0,len(metrics)):
		df = df.append({"Comparison" : x_labels2[0] + ' vs. ' + x_labels2[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : data_m[0][dfi],
					"Group1 95% CI" : (data_m[0][dfi]-data_sd_l[0][dfi],data_m[0][dfi]+data_sd_u[0][dfi]),
					"Group2 Mean" : data_m[cind][dfi],
					"Group2 95% CI" : (data_m[cind][dfi]-data_sd_l[cind][dfi],data_m[cind][dfi]+data_sd_u[cind][dfi]),
					"t-stat (vs MDD)" : tstat[cind][dfi],
					"p-value (vs MDD)" : p[cind][dfi],
					"Cohen's d (vs MDD)" : cd[cind][dfi]},
					ignore_index = True)
		df = df.append({"Comparison" : x_labels2[-1] + ' vs. ' + x_labels2[cind],
					"Metric" : metrics[dfi],
					"Group1 Mean" : data_m[-1][dfi],
					"Group1 95% CI" : (data_m[-1][dfi]-data_sd_l[-1][dfi],data_m[-1][dfi]+data_sd_u[-1][dfi]),
					"Group2 Mean" : data_m[cind][dfi],
					"Group2 95% CI" : (data_m[cind][dfi]-data_sd_l[cind][dfi],data_m[cind][dfi]+data_sd_u[cind][dfi]),
					"t-stat (vs MDD)" : tstat0[cind][dfi],
					"p-value (vs MDD)" : p0[cind][dfi],
					"Cohen's d (vs MDD)" : cd0[cind][dfi]},
					ignore_index = True)

df.to_csv('figs_ratedistributions_doseResponse/stats_FailedFalse.csv')

x_labels2 = ['MDD',
			r'25% $\alpha$'+'5-PAM',
			r'50% $\alpha$'+'5-PAM',
			r'75% $\alpha$'+'5-PAM',
			r'100% $\alpha$'+'5-PAM',
			r'125% $\alpha$'+'5-PAM',
			r'150% $\alpha$'+'5-PAM',
			'Healthy']
x = [0.,8.5]
width = 1.
fig_det, ax_det = plt.subplots(figsize=(9.5, 6))
xt = [0 for _ in range(0,len(conds)*2-2)]
for cind,c in enumerate(conds[:-1]):
	if not len(conds) % 2:
		x2 = [xr+cind*width+width/2 for xr in x]
		x3 = [n+0.5 for n in range(len(conds))]
	else:
		x2 = [xr+cind*width for xr in x]
		x3 = [n for n in range(len(conds))]
	
	xt[cind] = x2[0]
	xt[cind+7] = x2[1]
	
	ax_det.bar(x2,height=data_m[cind],
		   width=width,    # bar width
		   color=colors_conds[cind],  # face color transparent
		   alpha=alphas_conds[cind],
		   label=x_labels2[cind],
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

ax_det.plot([x[0], x[0]+len(conds[:-1])],[data_m[-1][0],data_m[-1][0]],'k',ls='dashed',alpha=1)
ax_det.plot([x[1], x[1]+len(conds[:-1])],[data_m[-1][1],data_m[-1][1]],'k',ls='dashed',alpha=1)
ax_det.fill_between([x[0], x[0]+len(conds[:-1])], y1 = [data_m[-1][0]+data_sd_u[-1][0],data_m[-1][0]+data_sd_u[-1][0]], y2 = [data_m[-1][0]-data_sd_l[-1][0],data_m[-1][0]-data_sd_l[-1][0]], color='k', alpha=0.2, zorder=2)
ax_det.fill_between([x[1], x[1]+len(conds[:-1])], y1 = [data_m[-1][1]+data_sd_u[-1][1],data_m[-1][1]+data_sd_u[-1][1]], y2 = [data_m[-1][1]-data_sd_l[-1][1],data_m[-1][1]-data_sd_l[-1][1]], color='k', alpha=0.2, zorder=2)

p_thresh = 0.05
c_thresh = 1.

for cind in range(0,len(conds[:-1])):
	if ((p[cind][0] < p_thresh) & (abs(cd[cind][0]) > c_thresh)):
		ax_det.text(x3[cind],data_m[cind][0]+data_sd_u[cind][0]+1.,'*',c=colors_conds[0],va='top' if ((p0[cind][0] < p_thresh) & (abs(cd0[cind][0]) >c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((p[cind][1] < p_thresh) & (abs(cd[cind][1]) > c_thresh)):
		ax_det.text(x3[cind]+x[1],data_m[cind][1]+data_sd_u[cind][1]+1.,'*',c=colors_conds[0],va='top' if ((p0[cind][1] < p_thresh) & (abs(cd0[cind][1]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((p0[cind][0] < p_thresh) & (abs(cd0[cind][0]) > c_thresh)):
		ax_det.text(x3[cind],data_m[cind][0]+data_sd_u[cind][0]+1.,'*',c='k',va='bottom' if ((p[cind][0] < p_thresh) & (abs(cd[cind][0]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)
	if ((p0[cind][1] < p_thresh) & (abs(cd0[cind][1]) > c_thresh)):
		ax_det.text(x3[cind]+x[1],data_m[cind][1]+data_sd_u[cind][1]+1.,'*',c='k',va='bottom' if ((p[cind][1] < p_thresh) & (abs(cd[cind][1]) > c_thresh)) else 'top', ha='center',fontweight='bold',fontsize=fsize)

x_labels = ['MDD',
			'25%',
			'50%',
			'75%',
			'100%',
			'125%',
			'150%',
			'MDD',
			'25%',
			'50%',
			'75%',
			'100%',
			'125%',
			'150%']

# ax_det.legend(loc='upper right', prop={'size': 10})
ax_det.set_ylabel('Probability (%)')
# ax_det.set_xticks([xx+len(conds)/2-width/2 for xx in x])
# ax_det.set_xticklabels(['Failed\nDetection','False\nDetection'])
ax_det.set_xticks(xt)
ax_det.set_xticklabels(x_labels, rotation = 45, ha="center")
ax_det.grid(False)
ax_det.spines['right'].set_visible(False)
ax_det.spines['top'].set_visible(False)
ax_det.set_xlim(-1.1,16.1)
fig_det.tight_layout()
fig_det.savefig('figs_ratedistributions_doseResponse/DetectionProbability.png',dpi=300,transparent=True)
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
	idx = idx[np.where(np.logical_and(idx>1000, idx<=2200))][0]
	ax_ov.plot(bin_centres_highres, br, label='Pre', color = colors_conds[cind], ls='dashed', alpha = 0.5*alphas_conds[cind])
	ax_ov.plot(bin_centres_highres, sr, label='Post', color = colors_conds[cind], alpha = 0.8*alphas_conds[cind])
	ax_ov.fill_between(bin_centres_highres[:idx],sr[:idx], color = colors_conds[cind], alpha = 0.5*alphas_conds[cind])
	ax_ov.fill_between(bin_centres_highres[idx:],br[idx:], color = colors_conds[cind], alpha = 0.8*alphas_conds[cind])
	ax_ov.set_ylim(bottom=0)
	ylims = ax_ov.get_ylim()
	ax_ov.plot([bin_centres_highres[idx],bin_centres_highres[idx]], ylims, color = colors_conds[cind], linewidth=2, ls='dotted', alpha=0.3)
	ax_ov.set_ylim(ylims)
	cind += 1
ax_ov.set_xlabel('Spike Rate (Hz)')
ax_ov.set_ylabel('Proportion')
ax_ov.set_xlim(0,bin_centres_highres[-1])
fig_ov.tight_layout()
fig_ov.savefig('figs_ratedistributions_doseResponse/RateDistribution.png',dpi=300,transparent=True)
plt.close()

x_labels2 = ['MDD',
			'MDD + '+r'25% $\alpha$'+'5-PAM',
			'MDD + '+r'50% $\alpha$'+'5-PAM',
			'MDD + '+r'75% $\alpha$'+'5-PAM',
			'MDD + '+r'100% $\alpha$'+'5-PAM',
			'MDD + '+r'125% $\alpha$'+'5-PAM',
			'MDD + '+r'150% $\alpha$'+'5-PAM',
			'Healthy']
cind = 0
for br, sr in zip(hist_fit_base,hist_fit_stim):
	fig_ov, ax_ov = plt.subplots(figsize=(7, 6))
	idx = np.argwhere(np.diff(np.sign(br - sr))).flatten()
	idx = idx[np.where(np.logical_and(idx>1000, idx<=2200))][0]
	ax_ov.plot(bin_centres_highres, br, label='Pre', color = colors_conds[cind], ls='dashed', alpha = 0.5*alphas_conds[cind], linewidth=3)
	ax_ov.plot(bin_centres_highres, sr, label='Post', color = colors_conds[cind], alpha = 0.8*alphas_conds[cind], linewidth=3)
	ax_ov.fill_between(bin_centres_highres[:idx],sr[:idx], color = colors_conds[cind], alpha = 0.5*alphas_conds[cind])
	ax_ov.fill_between(bin_centres_highres[idx:],br[idx:], color = colors_conds[cind], alpha = 0.8*alphas_conds[cind])
	ax_ov.set_ylim(ylims)
	ylims = ax_ov.get_ylim()
	ax_ov.plot([bin_centres_highres[idx],bin_centres_highres[idx]], ylims, color = 'k', linewidth=2, ls='dotted', alpha=0.8)
	ax_ov.set_ylim(ylims)
	ax_ov.set_xlabel('Spike Rate (Hz)')
	ax_ov.set_ylabel('Proportion')
	ax_ov.set_xlim(0,bin_centres_highres[-1])
	ax_ov.set_title(x_labels2[cind])
	fig_ov.tight_layout()
	fig_ov.savefig('figs_ratedistributions_doseResponse/RateDistribution_'+conds[cind]+'.png',dpi=300,transparent=True)
	plt.close()
	cind += 1
