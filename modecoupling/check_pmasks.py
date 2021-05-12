# Usage: python2 check_pmasks.py inputs/check_pmasks.yaml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
matplotlib.rcParams.update({'font.size': 14})
from AnalysisBackend.hdf5 import modecoupling_hdf5 as mode
from AnalysisBackend.hdf5 import window_hdf5
import glob
import sys
import yaml
import os
from AnalysisBackend.mapping import maprep
from AnalysisBackend.mapping import polwindow
from AnalysisBackend.powerspectrum import psestimator

def make_pmask(path, source, mask_cut=0.01, window=0):
    """
    Make the pmask from a hdf5 file.
    INPUTS:
    path (str): path to hdf5 file
    source (str): 'PB1RA23HAB', 'PB1RA12HAB', or 'PB1LST4p5'
    OUTPUTS:
    pmask (array)
    """
    m,tweight,pweight = mode.prepare_map(path,source)
    pmask = window_hdf5.standard_pol_mask(m,pweight, mask_cut=mask_cut)
    if window:
        for i in range(len(pmask)):
            for j in range(len(pmask[i])):
                if pmask[i,j]<window:
                    pmask[i,j] = 0
    
    return pmask

def get_pweight(path):
    """
    Gets the polarization weights from a hdf5 file. Doesn't do any filtering, but does pad the array.
    INPUTS:
    path (str): path to hdf5 file
    OUTPUTS:
    pweights (array)
    """
    m = maprep.MapMakingVectors.load(path)
    cc = m.mapinfo.view2d(m.cc)
    cs = m.mapinfo.view2d(m.cs)
    ss = m.mapinfo.view2d(m.ss)
    pweight = polwindow.qu_weight_mineig(cc,cs,ss)
    pweight = psestimator.extend_mask_fft(pweight)
    return pweight

def make_padded_map(path, source):
    """
    Make the padded I, Q, and U maps from a hdf5 file.
    INPUTS:
    path (str): path to hdf5 file
    source (str): 'PB1RA23HAB', 'PB1RA12HAB', or 'PB1LST4p5'
    OUTPUTS:
    I, Q, U (arrays
    """
    m,tweight,pweight = mode.prepare_map(path,source)
    I, Q, U = window_hdf5.get_iqu(m)
    
    return I, Q, U
    
def pmasks_plot_diff(path1, path2, source, day, output_file=False):
    """
    Plot the coadd pmask, daily pmask, and their difference.
    INPUTS:
    path1 (str): path to coadd hdf5
    path2 (str): path to daily map hdf5
    source (str): 'PB1RA23HAB', 'PB1RA12HAB', or 'PB1LST4p5'
    day (str): Day timestamp
    output_file: Save filepath for plot
    OUTPUTS:
    Saved or shown plot
    """
    pmask1 = make_pmask(path1, source)
    pmask2 = make_pmask(path2, source)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    ax0.set_title("pmask coadd {}".format(source))
    im0 = ax0.imshow(pmask1)
    ax0.set_xlabel('pixels')
    ax0.set_ylabel('pixels')
    fig.colorbar(im0, ax=ax0)
    
    ax1.set_title("pmask "+day)
    im1 = ax1.imshow(pmask2)
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')
    fig.colorbar(im1, ax=ax1)
    
    ax2.set_title("Difference")
    im2 = ax2.imshow(pmask1-pmask2)
    ax2.set_xlabel('pixels')
    ax2.set_ylabel('pixels')
    fig.colorbar(im2, ax=ax2)
    
    if output_file:
        plt.savefig(output_file)
    
def max_diff(pmask1, pmask2, thresh):
    """
    Get the maximum difference across all pixels between 2 pmasks.
    INPUTS:
    pmask1 (array)
    pmask2 (array)
    thresh (float): Only consider pixels with a pmask1 value > thresh
    OUTPUTS:
    max_diff: Maximum difference (float)
    """
    max_diff = 0
    for p1, p2 in zip(np.ndarray.flatten(pmask1), np.ndarray.flatten(pmask2)):
        if p1>thresh:
            if np.abs(p1-p2)>np.abs(max_diff):
                max_diff = p1-p2
    return max_diff

def avg_diff(pmask1, pmask2, thresh):
    """
    Get the average (absolute value) difference across all pixels between 2 pmasks.
    INPUTS:
    pmask1 (array)
    pmask2 (array)
    thresh (float): Only consider pixels with a pmask1 value > thresh
    OUTPUTS:
    Average abs(difference) (float)
    """
    sum_diff = 0
    counter = 0
    for p1, p2 in zip(np.ndarray.flatten(pmask1), np.ndarray.flatten(pmask2)):
        if p1>thresh:
            counter += 1
            sum_diff += np.abs(p1-p2)        
    return sum_diff/counter

def pixel_vars(pweight):
    pweight = pweight/np.max(pweight)
#    var = np.ones(np.shape(pweight))*1000/np.min(pweight[np.nonzero(pweight)])
    var = np.zeros(np.shape(pweight))
    for i in range(len(pweight)):
        for j in range(len(pweight[i])):
            if pweight[i,j] != 0.0:
                var[i,j] = 1/pweight[i,j]
    return var

def compute_map_avg_var(weights, pixel_vars):
    N = np.sum(weights)
    pixel_contributions = weights**2*pixel_vars/N**2
    map_avg_var = np.sum(pixel_contributions)
    
    return map_avg_var, pixel_contributions

def compute_fractional_var_increase(pmask_coadd, pmask_day, pweight, window=False):
    pmask_coadd = pmask_coadd.copy()
    if window:
        for i in range(len(pmask_coadd)):
            for j in range(len(pmask_coadd[i])):
                if pmask_coadd[i,j]<window:
                    pmask_coadd[i,j] = 0
        
    pvars = pixel_vars(pweight)
    
    map_avg_var_day, pixel_contributions_day = compute_map_avg_var(pmask_day, pvars)
    map_avg_var_coadd, pixel_contributions_coadd = compute_map_avg_var(pmask_coadd, pvars)
    
    frac_var = map_avg_var_coadd/map_avg_var_day-1
    frac_var_pixel = (pixel_contributions_coadd-pixel_contributions_day)/map_avg_var_day

    return frac_var, frac_var_pixel

def days_list():
    """Return a list of all day names"""
    days = []
    for patch in ['pb1lst4p5_3x3', 'pb1ra12hab_3x3', 'pb1ra23hab_3x3']:
        days.extend(patch_days_list(patch))
    return days

def patch_days_list(patch):
    """Given a patch folder (e.g. pb1lst4p5), return a list of days"""
    days_dir = "/global/cscratch1/sd/jspisak/pipeline/data/sim_signal_only/{}/sim_signal_only/".format(patch)
    return [day_dir.split('/')[-1] for day_dir in glob.glob(days_dir + "201*")]

if __name__ == "__main__":
    input_yaml = sys.argv[1]
    with open(input_yaml) as file:
        inputs = yaml.load(file, Loader=yaml.FullLoader)
        
    hdf5 = inputs['hdf5']
    patches = inputs['patches']
    sources = inputs['sources']
    thresh = inputs['thresh']
    plot_num = inputs['plot_num']
    output_dir = inputs['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #Iterate over all patches
    for source, patch in zip(sources, patches):
        coadd_path = "/global/cscratch1/sd/jspisak/pipeline/data/sim_signal_only/{}/sim_signal_only/coadd/{}".format(patch, hdf5)
        days_dir = "/global/cscratch1/sd/jspisak/pipeline/data/sim_signal_only/{}/sim_signal_only/".format(patch)
        days = [day_dir.split('/')[-1] for day_dir in glob.glob(days_dir + "201*")]
        avg_diffs = []
        max_diffs = []
        pmask1 = make_pmask(coadd_path, source)
    
        #Iterate through each day in the patch and compute the max and average absolute value difference between pmasks 
        for day in days:
            day_path = days_dir + day + '/coadd/' + hdf5 
            pmask2 = make_pmask(day_path, source)
            max_diffs.append(max_diff(pmask1, pmask2, thresh))
            avg_diffs.append(avg_diff(pmask1, pmask2, thresh))
        
        # Plot the worst offenders for each patch
        worst_avg = np.flip(np.argsort(avg_diffs))[0:plot_num]
        worst_max = np.flip(np.argsort(max_diffs))[0:plot_num]
        for idx in np.unique(np.concatenate((worst_avg, worst_max), axis=0)):
            day = days[idx]
            day_path = days_dir + day + '/coadd/' + hdf5 
            pmasks_plot_diff(coadd_path, day_path, source, day, output_file="{}/pmask_{}_{}".format(output_dir, source, day))