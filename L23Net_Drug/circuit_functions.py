#================================================================================
#= Import
#================================================================================
import os
import time
tic = time.perf_counter()
from os.path import join
import sys
import zipfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
from mpi4py import MPI
import math
import neuron
from neuron import h, gui
import LFPy
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, StimIntElectrode
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools

#================================================================================
#= Controls
#================================================================================

# Frank ===============

# Alex ===============
plotnetworksomas = True
plotrasterandrates = True
plotephistimseriesandPSD = True
plotsomavs = True # Specify cell indices to plot in 'cell_indices_to_plot' - Note: plotting too many cells can randomly cause TCP connection errors

# Kant ===============

#================================================================================
#= Analysis
#================================================================================
#===============================
#= Analysis Parameters
#===============================
transient = 2000 #used for plotting and analysis

radii = [79000., 80000., 85000., 90000.] #4sphere model
sigmas = [0.3, 1.5, 0.015, 0.3] #conductivity
L23_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])
ECoG_sensor = np.array([[0., 0., 79000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)
ECoG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, ECoG_sensor)

sampling_rate = (1/0.025)*1000
nperseg = int(sampling_rate/2)
t1 = int(transient/0.025)
#===============================

def bandPassFilter(signal,low=.1, high=100.,order = 2):
	order = 2
	# z, p, k = ss.butter(order, [low,high],btype='bandpass',fs=sampling_rate,output='zpk')
	# sos = ss.zpk2sos(z, p, k)
	# y = ss.sosfiltfilt(sos, signal)
	b, a = ss.butter(order, [low,high],btype='bandpass',fs=sampling_rate)
	y = ss.filtfilt(b, a, signal)
	return y

#================================================================================
#= Plotting
#================================================================================
#===============================
#= Frank
#===============================
pop_colors = {'HL23PYR':'k', 'HL23SST':'red', 'HL23PV':'green', 'HL23VIP':'yellow'}
popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
#===============================

#===============================
#= Alex
#===============================
pop_colors = {'HL23PYR':'k', 'HL23SST':'crimson', 'HL23PV':'green', 'HL23VIP':'darkorange'}
popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
poplabels = ['PN', 'MN', 'BN', 'VN']

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})
#===============================


# Plot soma positions
def plot_network_somas(OUTPUTPATH):
	filename = os.path.join(OUTPUTPATH,'cell_positions_and_rotations.h5')
	popDataArray = {}
	popDataArray[popnames[0]] = pd.read_hdf(filename,popnames[0])
	popDataArray[popnames[0]] = popDataArray[popnames[0]].sort_values('gid')
	popDataArray[popnames[1]] = pd.read_hdf(filename,popnames[1])
	popDataArray[popnames[1]] = popDataArray[popnames[1]].sort_values('gid')
	popDataArray[popnames[2]] = pd.read_hdf(filename,popnames[2])
	popDataArray[popnames[2]] = popDataArray[popnames[2]].sort_values('gid')
	popDataArray[popnames[3]] = pd.read_hdf(filename,popnames[3])
	popDataArray[popnames[3]] = popDataArray[popnames[3]].sort_values('gid')

	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=5)
	for pop in popnames:
		for i in range(0,len(popDataArray[pop]['gid'])):
			ax.scatter(popDataArray[pop]['x'][i],popDataArray[pop]['y'][i],popDataArray[pop]['z'][i], c=pop_colors[pop], s=5)
			ax.set_xlim([-300, 300])
			ax.set_ylim([-300, 300])
			ax.set_zlim([-1200, -400])
	return fig

# Plot spike raster plots & spike rates
def plot_raster_and_rates(SPIKES,tstart_plot=transient,tstop_plot=network.tstop,popnames=popnames,network=network,OUTPUTPATH=OUTPUTPATH,GLOBALSEED=GLOBALSEED,tstim=network.tstop):
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	fig = plt.figure(figsize=(10, 8))
	ax1 =fig.add_subplot(111)
	
	N_cells = 0
	for pop in cell_names:
		N_cells += circuit_params['SING_CELL_PARAM'].at['cell_num',pop]
		print('Number of cells = '+str(N_cells))
	
	for name, spts, gids in zip(popnames, SPIKES['times'], SPIKES['gids']):
		t = []
		g = []
		for spt, gid in zip(spts, gids):
			t = np.r_[t, spt]
			g = np.r_[g, np.zeros(spt.size)+gid]
			ax1.plot(t[t >= tstart_plot], g[t >= tstart_plot], '|', color=pop_colors[name])
		ax1.set_ylim(0,N_cells)
		ax1.set_xlim(tstart_plot,tstop_plot)
		ax1.set_xlabel('Time (ms)')
		ax1.set_ylabel('Cell Number')

	PN = []
	MN = []
	BN = []
	VN = []
	SPIKE_list = [PN ,MN, BN, VN]
	SPIKE_MEANS = []
	SPIKE_STDEV = []
	for i in range(4):
		for j in range(len(SPIKES['times'][i])):
			scount = SPIKES['times'][i][j][(SPIKES['times'][i][j]>transient) & (SPIKES['times'][i][j]<=tstim)]
			Hz = np.array([(scount.size)/((int(tstim)-transient)/1000)])
			SPIKE_list[i].append(Hz)
		SPIKE_MEANS.append(np.mean(SPIKE_list[i]))
		SPIKE_STDEV.append(np.std(SPIKE_list[i]))

	meanstdevstr1 = '\n' + str(np.around(SPIKE_MEANS[0], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[0], decimals=2))
	meanstdevstr2 = '\n' + str(np.around(SPIKE_MEANS[1], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[1], decimals=2))
	meanstdevstr3 = '\n' + str(np.around(SPIKE_MEANS[2], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[2], decimals=2))
	meanstdevstr4 = '\n' + str(np.around(SPIKE_MEANS[3], decimals=2)) + r' $\pm$ '+ str(np.around(SPIKE_STDEV[3], decimals=2))
	names = [poplabels[0]+meanstdevstr1,poplabels[1]+meanstdevstr2,poplabels[2]+meanstdevstr3,poplabels[3]+meanstdevstr4]

	Hzs_mean = np.array(SPIKE_MEANS)
	np.savetxt(os.path.join(OUTPUTPATH,'spikerates_Seed' + str(int(GLOBALSEED)) + '.txt'),Hzs_mean)
	w = 0.8
	fig2 = plt.figure(figsize=(10, 8))
	ax2 = fig2.add_subplot(111)
	ax2.bar(x = [0, 1, 2, 3],
	height=[pop for pop in SPIKE_MEANS],
	yerr=[np.std(pop) for pop in SPIKE_list],
	capsize=12,
	width=w,
	tick_label=names,
	color=[clr for clr in colors],
	edgecolor='k',
	ecolor='black',
	linewidth=4,
	error_kw={'elinewidth':3,'markeredgewidth':3})
	ax2.set_ylabel('Spike Frequency (Hz)')
	ax2.grid(False)

	return fig, fig2

# Plot spike time histograms
def plot_spiketimehists(SPIKES,network):
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	binsize = 10 # ms
	numbins = int((network.tstop - transient)/binsize)
	fig, axarr = plt.subplots(len(colors),1)
	for i in range(len(colors)):
		popspikes = list(itertools.chain.from_iterable(SPIKES['times'][i]))
		popspikes = [i for i in popspikes if i > transient]
		axarr[i].hist(popspikes,bins=numbins,color=colors[i],linewidth=0,edgecolor='none',range=(2000,7000))
		axarr[i].set_xlim(transient,network.tstop)
		if i < len(colors)-1:
			axarr[i].set_xticks([])
	axarr[-1:][0].set_xlabel('Time (ms)')

	return fig

# Plot EEG & ECoG voltages & PSDs
def plot_eeg(network,DIPOLEMOMENT,low=.1,high=100.,order=2,tstop=network.tstop):
	t2 = int(tstop/0.025)
	DP = DIPOLEMOMENT['HL23PYR']
	for pop in popnames[1:]:
		DP = np.add(DP,DIPOLEMOMENT[pop])
	
	EEG = EEG_args.calc_potential(DP, L23_pos)
	EEG = EEG[0]
	
	EEG_filt = bandPassFilter(EEG[t1:t2],low,high,order)
	
	EEG_freq, EEG_ps = ss.welch(EEG_filt, fs=sampling_rate, nperseg=nperseg)
	EEGraw_freq, EEGraw_ps = ss.welch(EEG[t1:t2], fs=sampling_rate, nperseg=nperseg)
	
	tvec = np.arange((network.tstop)/(1000/sampling_rate)+1)*(1000/sampling_rate)
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	ax1.plot(tvec[t1:t2], EEG_filt, c='k')
	ax1.set_xlim(transient,tstop)
	ax1.set_ylabel('EEG (mV)')
	ax2 = fig.add_subplot(212)
	ax2.plot(EEG_freq, EEG_ps, c='k')
	ax2.set_xlim(0,100)
	ax2.set_xlabel('Frequency (Hz)')
	
	fig2 = plt.figure(figsize=(10,10))
	ax21 = fig2.add_subplot(221)
	ax21.plot(tvec[t1:], EEG[t1:], c='k')
	ax21.set_xlim(transient,tstop)
	ax21.set_ylabel('EEG (mV)')
	ax22 = fig2.add_subplot(222)
	ax22.plot(EEGraw_freq, EEGraw_ps, c='k')
	ax22.set_xlim(0,100)
	ax22.set_xlabel('Frequency (Hz)')

	return fig, fig2

# Plot LFP voltages & PSDs
def plot_lfp(network,OUTPUT,tstop=network.tstop):
	t2 = int(tstop/0.025)
	LFP1_freq, LFP1_ps = ss.welch(OUTPUT[0]['imem'][0][t1:t2], fs=sampling_rate, nperseg=nperseg)
	
	tvec = np.arange((network.tstop)/(1000/sampling_rate)+1)*(1000/sampling_rate)
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(211)
	ax1.plot(tvec[t1:],OUTPUT[0]['imem'][0][t1:],'k')
	ax1.set_xlim(transient,network.tstop)
	ax1.set_xlabel('Time (ms)')
	
	ax2 = fig.add_subplot(212)
	ax2.plot(LFP1_freq,LFP1_ps,'k')
	ax2.set_xlim(0,100)
	ax2.set_xlabel('Frequency (Hz)')
	fig.tight_layout()

	return fig

# Collect Somatic Voltages Across Ranks
def somavCollect(network,cellindices,RANK,SIZE,COMM):
	if RANK == 0:
		volts = []
		gids2 = []
		for i, pop in enumerate(network.populations):
			svolts = []
			sgids = []
			for gid, cell in zip(network.populations[pop].gids, network.populations[pop].cells):
				if gid in cellindices:
					svolts.append(cell.somav)
					sgids.append(gid)

			volts.append([])
			gids2.append([])
			volts[i] += svolts
			gids2[i] += sgids

			for j in range(1, SIZE):
				volts[i] += COMM.recv(source=j, tag=15)
				gids2[i] += COMM.recv(source=j, tag=16)
	else:
		volts = None
		gids2 = None
		for i, pop in enumerate(network.populations):
			svolts = []
			sgids = []
			for gid, cell in zip(network.populations[pop].gids, network.populations[pop].cells):
				if gid in cellindices:
					svolts.append(cell.somav)
					sgids.append(gid)
			COMM.send(svolts, dest=0, tag=15)
			COMM.send(sgids, dest=0, tag=16)

	return dict(volts=volts, gids2=gids2)

# Plot somatic voltages for each population
def plot_somavs(network,VOLTAGES,gstart=transient,gstop=network.tstop,tstim=0):
	tvec = np.arange(network.tstop/network.dt+1)*network.dt-tstim
	fig = plt.figure(figsize=(10,5))
	cls = ['black','crimson','green','darkorange']
	for i, pop in enumerate(network.populations):
		for v in range(0,len(VOLTAGES['volts'][i])):
			ax = plt.subplot2grid((len(VOLTAGES['volts']), len(VOLTAGES['volts'][i])), (i, v), rowspan=1, colspan=1, frameon=False)
			ax.plot(tvec,VOLTAGES['volts'][i][v], c=cls[i])
			ax.set_xlim(gstart,gstop)
			ax.set_ylim(-85,40)
			if i < len(VOLTAGES['volts'])-1:
				ax.set_xticks([])
			if v > 0:
				ax.set_yticks([])

	return fig

# Run Plot Functions
if plotsomavs:
	N_HL23PYR = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23PYR']
	N_HL23SST = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23SST']
	N_HL23PV = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23PV']
	N_HL23VIP = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23VIP']
	cell_indices_to_plot = [0, N_HL23PYR, N_HL23PYR+N_HL23SST, N_HL23PYR+N_HL23SST+N_HL23PV] # Plot first cell from each population
	VOLTAGES = somavCollect(network,cell_indices_to_plot,RANK,SIZE,COMM)

if RANK ==0:
	if plotnetworksomas:
		fig = plot_network_somas(OUTPUTPATH)
		fig.savefig(os.path.join(OUTPUTPATH,'network_somas_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotrasterandrates:
		fig, fig2 = plot_raster_and_rates(SPIKES,tstim=4000)
		fig.savefig(os.path.join(OUTPUTPATH,'raster_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig2.savefig(os.path.join(OUTPUTPATH,'rates_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_spiketimehists(SPIKES,network)
		fig.savefig(os.path.join(OUTPUTPATH,'spiketimes_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotephistimseriesandPSD:
		fig, fig2 = plot_eeg(network,DIPOLEMOMENT,.1,100.,2)
		fig.savefig(os.path.join(OUTPUTPATH,'eeg_filt_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig2.savefig(os.path.join(OUTPUTPATH,'eeg_raw_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_lfp(network,OUTPUT)
		fig.savefig(os.path.join(OUTPUTPATH,'lfps_traces_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotsomavs:
		fig = plot_somavs(network,VOLTAGES)
		fig.savefig(os.path.join(OUTPUTPATH,'somav_'+str(GLOBALSEED)),bbox_inches='tight', dpi=300, transparent=True)
#===============================
# Kant
#===============================
pop_colors = {'HL23PYR':'k', 'HL23SST':'red', 'HL23PV':'green', 'HL23VIP':'yellow'}
popnames = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
#===============================
