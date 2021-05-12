import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 16}
matplotlib.rc('font', **font)
import numpy as np

# Prepare bounds
lyman = 2*10**-21 #eV
SN1987a = 6*10**-12 #GeV^-1 and 10.8 M_sol Jansson and Farrar model. Just guessed at the valeu
cast = 0.66*10**-10 #GeV^-1

# Constants
k = 1
rho = 0.3 #GeV/cm^3
hbar =  6.582*10**-16# eV*s
c = 2.998 * 10**8 # SI

# functions
def m_to_hz(m):
    """"""
    return m/(2*np.pi*hbar)

def hz_to_m(hz):
    return hz*2*np.pi*hbar

def g(angle, m, k, rho):
    """Input axion mass in eV, dark matter fraction k, and critical density rho in GeV/cm^3. Return coupling constant."""
    return 4.58*10**-10 * (2*angle*np.pi/180) * (m/10**-21) * (k*rho/0.3)**(-0.5)

def angle_plot(lims, angle, k, rho):
    """Generate the coupling constant vs mass arrays for plotting"""
    masses = np.linspace(lims[0], lims[1], num=100)
    gs = g(angle, masses, k, rho)
    return masses, gs

# Angle limits and plot domains
bicep_lims = [1.5*10**-21, hz_to_m((3600*24)**-1)]
bicep_angle = 0.68
bicep_masses, bicep_gs = angle_plot(bicep_lims, bicep_angle, k, rho) 

SA_lims = [1.5*10**-21/3, hz_to_m((3600*24)**-1)]
SA_angle = 4*2*1.3/60. #2x 68% lim plus 30% degredation
SA_masses, SA_gs = angle_plot(SA_lims, SA_angle, k, rho) 

SO_lims = [1.5*10**-21/2, hz_to_m((3600*24)**-1)]
SO_angle = 0.32*2*1.3/60. #2x 68% lim plus 30% degredation
SO_masses, SO_gs = angle_plot(SO_lims, SO_angle, k, rho) 

ACT_lims = [1.5*10**-21/2, hz_to_m((3600*24)**-1)]
ACT_angle = 0.06*2*1.3 #2x 68% lim plus 30% degredation
ACT_masses, ACT_gs = angle_plot(ACT_lims, ACT_angle, k, rho) 

tenth_lims = [10**-23, 10**-18]
tenth_angle = 0.1
tenth_masses, tenth_gs = angle_plot(tenth_lims, tenth_angle, k, rho) 

hund_lims = [10**-23, 10**-18]
hund_angle = 0.01
hund_masses, hund_gs = angle_plot(hund_lims, hund_angle, k, rho) 

# Washout
wash_masses = np.linspace(10**-23, 10**-18, num=100)
wash_gs = 9.6*10**-13 * (wash_masses/10**-21)*k

# Plot 
fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

xmin = 10**-23
xmax = 10**-18
ymin, ymax = 10**-14, 10**-9

# Bounds
# ax1.axvline(lyman, c='k', ls=':')
# ax1.axvspan(xmin, lyman, alpha=0.2, color='grey')
# ax1.axhline(SN1987a, c='k', ls='--')
# ax1.axhline(cast, c='k', ls='-.')
# ax1.axhspan(SN1987a, ymax, alpha=0.2, color='grey')

# Angles
ax1.loglog(bicep_masses, bicep_gs, c='b', label='Keck 2012 (smoothed)')
ax1.loglog(SA_masses, SA_gs, c='r', ls='--', label='SA (projected)')
ax1.loglog(SO_masses, SO_gs, c='c', ls='--', label='SO (projected)')
ax1.loglog(ACT_masses, ACT_gs, c='tab:orange', ls='--', label='ACT (projected from 2014/2015 data)')
# ax1.loglog(wash_masses, wash_gs, c='tab:green', label=r'Washout (Fedderke et al.)')

ax1.set_xlabel(r"$m_{\phi}$ [eV]", fontsize=14)
ax1.set_ylabel(r"$g_{\phi \gamma}$ $[GeV^{-1}]$", fontsize=16)
ax1.set_xlim(2*10**-22, 10**-19)
ax1.set_ylim(10**-13, 10**-9)
ax1.legend(prop={'size':10})

#Arrows
ax1.annotate('', xy=(0.95*np.max(bicep_masses), np.max(bicep_gs)),
             xytext=(0.95*np.max(bicep_masses), 0.6*np.max(bicep_gs)),
             arrowprops=dict(arrowstyle= '<|-', color='b', lw=1.5))
ax1.annotate('', xy=(0.95*np.max(SA_masses), np.max(SA_gs)),
             xytext=(0.95*np.max(SA_masses), 0.6*np.max(SA_gs)),
             arrowprops=dict(arrowstyle= '<|-', color='r', lw=1.5))
ax1.annotate('', xy=(0.95*np.max(ACT_masses), np.max(ACT_gs)),
             xytext=(0.95*np.max(ACT_masses), 0.6*np.max(ACT_gs)),
             arrowprops=dict(arrowstyle= '<|-', color='tab:orange', lw=1.5))
ax1.annotate('', xy=(0.95*np.max(SO_masses), np.max(SO_gs)),
             xytext=(0.95*np.max(SO_masses), 0.6*np.max(SO_gs)),
             arrowprops=dict(arrowstyle= '<|-', color='c', lw=1.5))

#New ticks
def tick_function(m):
    hz = m_to_hz(np.array(m))
    return ["%.3f" % h for h in hz]
hz_ticks = np.array([10**-7, 10**-6, 10**-5])
m_ticks = [hz_to_m(h) for h in hz_ticks]
ax2.set_xscale('log')
ax2.set_xlim(10**-22, 10**-19)
ax2.set_xticks(m_ticks)
ax2.set_xticklabels([ r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$'])
ax2.set_xlabel(r"Oscillation frequency (Hz)", fontsize=16)

plt.savefig("outputs/CPR_projections_axion_only.png", bbox_inches='tight')