# Compute the 2d posterior of full-season angle data
# Make sure to run 'module load python' first to get into an environment where python3 has scipy

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from datetime import datetime
import json
import sys
import yaml
from fractions import Fraction
import h5py

def log_prior(b, prior_lims):
    """A uniform log prior that returns 0 if within prior bounds, -inf otherwise"""
    for param, lim_pair in zip(b, prior_lims):
        low = lim_pair[0]
        high = lim_pair[1]
        if param < low or param > high:
            return -np.inf
    else:
        return 0
    
def posterior_log_np(b, data):
    """Returns the log of the posterior value without the prior """
    ll = 0
    a, f, ph, st = b
     #Iterate over the steps and add the posterior contribution
    for step in data.keys():
        angs = data[step]['angs']
        t_os = data[step]['days_start']
        t_fs = data[step]['days_end']
        rvar = data[step]['rvar']
        stds = data[step]['stds']
        N = len(angs)

        #Calculate some variables
        sred = (st**-2 + rvar)**-0.5
        x_avg = np.mean([(angs[i] - sineavg(t_os[i], t_fs[i], a, f, ph))/(stds[i]**2) for i in range(N)])
        xsq_avg = np.mean([((angs[i] - sineavg(t_os[i], t_fs[i], a, f, ph))/stds[i])**2 for i in range(N)])

        ll += 0.5*(sred*x_avg)**2 - 0.5*xsq_avg + np.log(sred)-np.log(st)
   
    return ll    

def posterior_log(b, data, prior_lims):
    """Returns the log of the posterior value. If the prior limits exist, 
    make sure the posterior is zero if outisde the max/min limits"""
    lp = log_prior(b, prior_lims)
    if not np.isfinite(lp):
        return -np.inf
    return lp + posterior_log_np(b, data)

def sineavg(t_o, t_f, a, f, ph):
    """Average of sine wave with params "a, f, ph", over times t_o to t_f"""
    t_o = np.array(t_o)
    t_f = np.array(t_f)
    return (a/(2*np.pi*f*(t_f-t_o)))*(np.cos(2*np.pi*f*t_o + ph) - np.cos(2*np.pi*f*t_f + ph))

def date_to_ctime(date):
    """Given a date in the folder notation (yyyymmdd_hhmmss), return
    the time in seconds since the beginning of 2012"""
    datetime_object = datetime.strptime(date, '%Y%m%d_%H%M%S')
    seconds = (datetime_object-datetime(1970,1,1)).total_seconds()
    return seconds

def step_dict(hwp_file, angle_file):
    """
    Given a hwp file containing the steps and their times, and an angle file containing
    the angles, times, and errors, generate a dictionary containing all the angles and 
    their error values, indexed by the HWP steps
    """
    #Import the hwp steps
    hwp_list = []
    f = open(hwp_file, "r")
    for x in f:
        hwp_list.append(x[:-1])
    f.close()

    with open(angle_file, 'r') as file:
        angle_dir = json.load(file)

    #Generate a dictionary with the times and whether they correspond to a hwp step or observation
    time_dir = {}
    for date in angle_dir.keys():
        t = date_to_ctime(date)
        time_dir[t] = [date, 'obs']
    for date in hwp_list:
        t = date_to_ctime(date)
        time_dir[t] = [date, 'hwp']

    #Inject the signal if desired
    inject_signal = False
    # aT = 2
    # fT = 1/30.
    # phT = 1    

    #Create a dictionary indexed by hwp steps. Structure is {'step 0': {'first_day'=day at start of step, 
    # 'days_start' = [times at which each data point was taken, in days], 'days_end' = days_start + 8 hours, 
    # 'angs' = [angles], 'errs' = [standard deviation of each angle]}, ....}
    typ_prev = 'hwp'
    step = 0
    dt = {}
    for i, t in enumerate(np.sort(list(time_dir.keys()))):
        typ = time_dir[t][1]
        if step == 0 or (typ == 'hwp' and typ_prev == 'obs'):
            step += 1
            dt[f"step {step}"] = {}
            dt[f"step {step}"]['days_start'] = []
            dt[f"step {step}"]['days_end'] = []
            dt[f"step {step}"]['angs'] = []
            dt[f"step {step}"]['stds'] = []
            dt[f"step {step}"]['rvar'] = 0
        if typ == 'obs':
            date = time_dir[t][0]
            day = t/(3600.*24)
            if not dt[f"step {step}"]['angs']:
                dt[f"step {step}"]['first_day'] = day 
            dt[f"step {step}"]['days_start'].append(day) #in hours
            dt[f"step {step}"]['days_end'].append(day + 8/24.) #NOT CORRECT: need actual end time
            if inject_signal == True:
                angle = angle_dir[date]['angle']*180./np.pi + sineavg(day, day + 8/24, aT, fT, phT)
            else:
                angle = angle_dir[date]['angle']*180/np.pi
            dt[f"step {step}"]['angs'].append(angle) #angle in degrees
    #         std = angle_dir[date]['std'] #NEED TO FIX TO INCORPORTE ACTUAL ANGLE STDS
            std = max(5,np.abs(angle)/3)
            dt[f"step {step}"]['stds'].append(std)
            dt[f"step {step}"]['rvar'] += std**-2
        typ_prev = typ
        
    return dt

    """ With the priors, angle and frequency domains, and an output filename, compute
    the 2d posterior via numerical integration.
    """

if __name__ == '__main__':
    
    #Get the step/angle dictionary
    hwp_file = "/global/cscratch1/sd/jspisak/pipeline/data/axion/HWP_dates/HWP_dates.txt"
    angle_file = "/global/cscratch1/sd/jspisak/pipeline/analysis/output/signal_angles_sep17.pkl"
    dt = step_dict(hwp_file, angle_file)

    #Input values
    accuracy = 1e-2
    a = 2
    f = 0.05
    prior_lims = [[0,10], [1/200., 1], [0, 2*np.pi], [1e-4, 5]]
    
    #Calculate the integrals
    sl = []
    def posterior_quad(ph, st):
        """Just a function of the phase and sigma_hwp to be integrated over"""
        return np.exp(posterior_log([a, f, ph, st], dt, prior_lims)+40)
    #Integrate with scipy quad over ph.
    ans = dblquad(posterior_quad, 1e-4, 5, lambda x: 0, lambda x: 2*np.pi, epsrel=accuracy)
    print(f"Answer {ans[0]}, error estimate {ans[1]}")
