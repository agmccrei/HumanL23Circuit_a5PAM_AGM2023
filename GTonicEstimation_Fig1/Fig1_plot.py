import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neuron import h, gui
from scipy import stats as st

h.load_file("nrngui.hoc")
h.load_file("import3d.hoc")

h.load_file("models/biophys_HL23MN1.hoc")
h.load_file("models/biophys_HL23PN1.hoc")
h.load_file("models/biophys_HL23BN1.hoc")
h.load_file("models/biophys_HL23VN1.hoc")

h.load_file("models/NeuronTemplate.hoc")
h.load_file("cell_functions.hoc")
h.load_file("net_functions.hoc")
h.load_file("net_params.hoc")

h('strdef modelname')

# Experimental values
# exp_PTX_current_m =
# exp_PTX_current_sd =

exp_baseline_all = [-43.80,-42.20,-78.20,-119.10,-109.10,-41.40,-70.80,-35.80,-52.37]
exp_basline_current_m = np.mean(exp_baseline_all)#-65.86
exp_basline_current_sd =  np.std(exp_baseline_all)#30.84

exp_drug_all = [-148.30,-89.80,-85.60,-142.80,-157.50,-42.00,-60.40,-53.30,-77.82]
exp_drug_current_m = np.mean(exp_drug_all)#-95.28
exp_drug_current_sd = np.std(exp_drug_all)#43.57

stat,p = st.ttest_rel(exp_baseline_all,exp_drug_all,alternative='greater')
print(p)

modelname = "HL23PN1"
Gtonic = 0.000938
GtonicDrug = 0.001498
IncreaseFactor_drug = ((GtonicDrug/Gtonic)-1)*100
IncreaseFactor_gene = (40/100)*((GtonicDrug/Gtonic)-1)*100

xra = [0,0.75]
fig_recov, ax_recov = plt.subplots(figsize=(2.5,2.6))
# fig_recov, ax_recov = plt.subplots(figsize=(3.4,2.6))
ax_recov.bar(xra,height=[Gtonic,GtonicDrug],
		width=0.35,    # bar width
		color=['lightgray','dodgerblue'],  # face color transparent
		edgecolor='k',
		ecolor='black',
		linewidth=1,
		error_kw={'elinewidth':3,'markeredgewidth':3}
		)
ax_recov.locator_params(axis='y', nbins=4)
ax_recov.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
ax_recov.set_ylabel(r'$G_{tonic,apical}$ (S/cm$^{2}$)')
ax_recov.set_xticks(xra)
ax_recov.set_xlim(-0.5,1.25)
ax_recov.set_xticklabels(['GABA','GABA\n+\n'+r'$\alpha_{5}$-PAM'])
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figures/Gtonic.png',dpi=300,transparent=True)
plt.close()

xra = np.arange(0,2)
fig_recov, ax_recov = plt.subplots(figsize=(2.5, 3))
ax_recov.bar(xra,height=[IncreaseFactor_drug,IncreaseFactor_gene],
		width=0.6,    # bar width
		color=['dodgerblue','lightgreen'],  # face color transparent
		edgecolor='k',
		ecolor='black',
		linewidth=2,
		error_kw={'elinewidth':3,'markeredgewidth':3}
		)
ax_recov.set_ylabel(r'$\alpha_{5}$-PAM Boost (%)')
ax_recov.set_xticks(xra)
ax_recov.set_xticklabels(['Drug\nEffect',r'$\alpha_{5}$ Expression' '\nCompensated'])
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figures/a5PAM_boost.png',dpi=300,transparent=True)
plt.close()

h.tstop = 10000
h.v_init = -75
t1 = 3000
t2 = 7000

cell = h.init_cell(modelname)

cvec = h.Vector()
tvec = h.Vector()

vc = h.VClamp(cell.soma[0](0.5))
vc.dur[0] = h.tstop
vc.amp[0] = -75
tvec.record(h._ref_t)
cvec.record(vc._ref_i)

gtonicvec_somabasal = h.Vector(6)
gtonic_apic = h.Vector(6)
timepointvec = h.Vector(6)

timepointvec.x[0] = 0
timepointvec.x[1] = t1
timepointvec.x[2] = t1 + h.dt
timepointvec.x[3] = t2
timepointvec.x[4] = t2 + h.dt
timepointvec.x[5] = h.tstop

gtonicvec_somabasal.x[0] = Gtonic
gtonicvec_somabasal.x[1] = Gtonic
gtonicvec_somabasal.x[2] = Gtonic
gtonicvec_somabasal.x[3] = Gtonic
gtonicvec_somabasal.x[4] = 0
gtonicvec_somabasal.x[5] = 0

gtonic_apic.x[0] = Gtonic
gtonic_apic.x[1] = Gtonic
gtonic_apic.x[2] = GtonicDrug
gtonic_apic.x[3] = GtonicDrug
gtonic_apic.x[4] = 0
gtonic_apic.x[5] = 0

for seg in cell.somatic:
	seg.insert("tonic")
	for n in range(1,seg.nseg+1):
		f = n/(seg.nseg+1)
		gtonicvec_somabasal.play(seg(f)._ref_g_tonic, timepointvec, 1)
		seg(f).e_gaba_tonic = -5
for seg in cell.basal:
	seg.insert("tonic")
	for n in range(1,seg.nseg+1):
		f = n/(seg.nseg+1)
		gtonicvec_somabasal.play(seg(f)._ref_g_tonic, timepointvec, 1)
		seg(f).e_gaba_tonic = -5
for seg in cell.apical:
	seg.insert("tonic")
	for n in range(1,seg.nseg+1):
		f = n/(seg.nseg+1)
		gtonic_apic.play(seg(f)._ref_g_tonic, timepointvec, 1)
		seg(f).e_gaba_tonic = -5

h.run()

i_GABA = np.array(cvec)[int(t1/h.dt-100)]*1000
i_PAM = np.array(cvec)[int(t2/h.dt-100)]*1000
i_PTX = np.array(cvec)[int(h.tstop/h.dt-100)]*1000

i_GABA_net = i_PTX-i_GABA
i_PAM_net = i_PTX-i_PAM

print('Baseline current (PTX) = '+str(i_PTX) + ' pA')
print('Baseline tonic current = '+str(i_GABA_net) + ' pA')
print('a5-PAM tonic current = '+str(i_PAM_net) + ' pA')

exp_basline_current_m
exp_basline_current_sd
exp_drug_current_m
exp_drug_current_sd

xra = [0-0.35/3,0+0.35/3,0.75-0.35/3,0.75+0.35/3]
fig_recov, ax_recov = plt.subplots(figsize=(2.5,2.6))
# fig_recov, ax_recov = plt.subplots(figsize=(3.4,2.6))
ax_recov.bar(xra,height=[-exp_basline_current_m,i_GABA_net,-exp_drug_current_m,i_PAM_net],
		width=0.35/1.5,    # bar width
		color=['white','lightgray','white','dodgerblue'],  # face color transparent
		edgecolor=['k','k','dodgerblue','dodgerblue'],
		linewidth=1)
ax_recov.errorbar(x=xra[0],
				y=-exp_basline_current_m,
				yerr=exp_basline_current_sd,
				ecolor='black',
				alpha=1,
				capsize=4,
				capthick=1,
				elinewidth=1)
ax_recov.errorbar(x=xra[2],
				y=-exp_drug_current_m,
				yerr=exp_drug_current_sd,
				ecolor='dodgerblue',
				alpha=1,
				capsize=4,
				capthick=1,
				elinewidth=1)
x_dat1 = xra[0]+(np.random.random(len(exp_baseline_all))-0.5)*0#(0.35/4)
x_dat2 = xra[2]+(np.random.random(len(exp_drug_all))-0.5)*0#(0.35/4)
ind=4
for x1,d1,x2,d2 in zip(x_dat1,np.abs(exp_baseline_all),x_dat2,np.abs(exp_drug_all)):
	ax_recov.plot([x1,x2],[d1,d2],c='darkgrey',ls='dashed',lw=0.8,zorder=ind)
	ind+=1
ax_recov.scatter(x_dat1,np.abs(exp_baseline_all),s=4**2,c='white',edgecolors='k',zorder=ind+1)
ax_recov.scatter(x_dat2,np.abs(exp_drug_all),s=4**2,c='white',edgecolors='dodgerblue',zorder=ind+2)
ax_recov.locator_params(axis='y', nbins=4)
ax_recov.set_ylabel('Current (pA)')
ax_recov.set_xticks(xra)
ax_recov.set_xlim(-0.5,1.25)
ax_recov.set_xticks(xra)
ax_recov.set_xticklabels(['exp','sim','exp','sim'], rotation = 45, ha="center")
ax_recov.tick_params(axis='both', which='both', length=0)
ax_recov.grid(False)
ax_recov.spines['right'].set_visible(False)
ax_recov.spines['top'].set_visible(False)

fig_recov.tight_layout()
fig_recov.savefig('figures/Itonic.png',dpi=300,transparent=True)
plt.close()

# Plot
fig, ax = plt.subplots(figsize=(4.3,2.5))

ax.fill_between([0,t1/1000],i_GABA,i_PTX,facecolor='lightgray',edgecolor='none',alpha=0.5)
ax.fill_between([t1/1000,t2/1000],i_PAM,i_PTX,facecolor='dodgerblue',edgecolor='none',alpha=0.5)

ax.fill_between([0,t1/1000],-180,-165,facecolor='lightgray')
ax.fill_between([t1/1000,t2/1000],-180,-165,facecolor='dodgerblue')
ax.fill_between([t2/1000,h.tstop/1000],-180,-165,facecolor='lightseagreen')

ax.text((0+100)/1000,-3,'PTX - GABA:')
ax.text((t1+100)/1000,-3,r'PTX - $\alpha_{5}$-PAM:')

ax.text((0+100)/1000,i_PTX-13,str(round(i_GABA_net,1))+' pA')
ax.text((t1+100)/1000,i_PTX-13,str(round(i_PAM_net,1))+' pA')

ax.text((0+100)/1000,-178,r'GABA (5 $\mu$M)')
ax.text((t1+100)/1000,-178,r'$\alpha_{5}$-PAM (3 $\mu$M)')
ax.text((t2+100)/1000,-178,r'PTX (100 $\mu$M)')

t0_tvec = np.array(tvec)[:int(t1/h.dt)]/1000
t1_tvec = np.array(tvec)[int((t1/h.dt)+h.dt):int(t2/h.dt)]/1000
t2_tvec = np.array(tvec)[int((t2/h.dt)+h.dt):]/1000
t0_cvec = np.array(cvec)[:int(t1/h.dt)]*1000
t1_cvec = np.array(cvec)[int((t1/h.dt)+h.dt):int(t2/h.dt)]*1000
t2_cvec = np.array(cvec)[int((t2/h.dt)+h.dt):]*1000
ax.plot(tvec/1000,cvec*1000,'k',linewidth=2)
ax.plot(t0_tvec,t0_cvec,color='lightgray',linewidth=1)
ax.plot(t1_tvec,t1_cvec,color='dodgerblue',linewidth=1)
ax.plot(t2_tvec,t2_cvec,color='lightseagreen',linewidth=1)

# ax.scatter(x=(t1/1000)/2,y=exp_basline_current_m+i_PTX,color='k',s=10)
# ax.errorbar(x=(t1/1000)/2,y=exp_basline_current_m+i_PTX,yerr=exp_basline_current_sd,ecolor='k',alpha=1,capsize=5,capthick=2,elinewidth=2)
#
# ax.scatter(x=(t1/1000)+(t2/1000-t1/1000)/2,y=exp_drug_current_m+i_PTX,color='k',s=10)
# ax.errorbar(x=(t1/1000)+(t2/1000-t1/1000)/2,y=exp_drug_current_m+i_PTX,yerr=exp_drug_current_sd,ecolor='k',alpha=1,capsize=5,capthick=2,elinewidth=2)

ax.set_xlim(0,h.tstop/1000)
ax.set_ylim(-180,5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current (pA)')
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('figures/tonic_estimation.png',dpi=300,transparent=True)
