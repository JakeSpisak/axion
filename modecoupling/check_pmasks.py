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

def make_pmask(path, source):
    """
    Make the pmask from a hdf5 file.
    INPUTS:
    path (str): path to hdf5 file
    source (str): 'PB1RA23HAB', 'PB1RA12HAB', or 'PB1LST4p5'
    OUTPUTS:
    pmask (array)
    """
    m,tweight,pweight = mode.prepare_map(path,source)
    pmask = window_hdf5.standard_pol_mask(m,pweight)
    
    return pmask
    
def pmasks_plot_diff(path1, path2, source, day, output_file):
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
    im2 = ax2.imshow(pmask1-pmask2, vmin=-1, vmax=1)
    ax2.set_xlabel('pixels')
    ax2.set_ylabel('pixels')
    fig.colorbar(im2, ax=ax2)
    
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