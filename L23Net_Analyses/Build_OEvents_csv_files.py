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

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

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
conds = ['MDD_a5PAM_long']#['healthy_long','MDD_long','MDD_a5PAM_long']
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

for cind, path in enumerate(paths):
	for seed in N_seedsList:
		print('Analyzing seed #'+str(seed)+' for '+conds[cind])
		# Load outputs
		temp_e = np.load(path + 'DIPOLEMOMENT_Seed' + str(seed) + '.npy')
		
		# EEG
		temp_e2 = temp_e['HL23PYR']
		temp_e2 = np.add(temp_e2,temp_e['HL23SST'])
		temp_e2 = np.add(temp_e2,temp_e['HL23PV'])
		temp_e2 = np.add(temp_e2,temp_e['HL23VIP'])
		
		potential = EEG_args.calc_potential(temp_e2, L23_pos)
		EEG = potential[0][t1:t2]
		EEG = np.array([[e for e in EEG]])
		
		dout = getIEIstatsbyBand(EEG,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,MUA,overlapth,getphase=True,savespec=True)
		df = GetDFrame(dout,sampr, EEG, MUA, haveMUA=False) # convert the oscillation event data into a pandas dataframe
		
		print('Normalizing wavelet spectrograms. . .')
		for ms in dout[chan]['lms']: ms.TFR = mednorm(ms.TFR)
		
		dlms={chan:dout[chan]['lms'] for chan in lchan}
		
		tt = np.linspace(0,len(EEG[0])/sampr,len(EEG[0]))
		evv = eventviewer(df,EEG,None,tt,sampr,winsz,dlms)
		evv.specrange = (0,10)
		
		# select and view a beta event
		#dfs = df[(df.filtsigcor>0.5)]
		evv.highlightevents(0,0,df.index)
		evv.fig.set_size_inches(12,10)
		ax = evv.fig.get_axes()
		ax[0].set_xlim(0,6000)
		ax[1].set_xlim(0,6000)
		ax[1].set_xlabel('Time (ms)')
		ax[1].set_ylabel('EEG (mV)')
		evv.fig.tight_layout()
		evv.fig.savefig('Saved_OEvents_thresh4/Spectrograms/'+conds[cind]+'_seed'+str(seed)+'.png',dpi=300,transparent=True)
		ax[0].set_ylim(freqmin,30)
		evv.fig.savefig('Saved_OEvents_thresh4/Spectrograms/'+conds[cind]+'_seed'+str(seed)+'_lowFreq.png',dpi=300,transparent=True)
		plt.close()
		
		df.to_csv('Saved_OEvents_thresh4/'+conds[cind]+'_seed'+str(seed)+'_OEvents.csv')

