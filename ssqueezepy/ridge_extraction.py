<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""Authors: David Bondesson, OverLordGoldDragon

Ridge extraction from time-frequency representations (STFT, CWT, synchrosqueezed).
"""
import numpy as np
from numba import jit

EPS = np.finfo(np.float64).eps


def extract_ridges(Tf, scales, penalty=2., n_ridges=1, bw=15, transform='cwt',
                   get_params=False):
    """Tracks time-frequency ridges by performing forward-backward ridge tracking
    algorithm, based on ref [1] (a version of Eq. III.4).

    Also see: https://www.mathworks.com/help/signal/ref/tfridge.html

    # Arguments:
        Tf: np.ndarray
            Complex time-frequency representation.

        scales:
            Frequency scales to calculate distance penalty term.

        penalty: float
            Value to penalize frequency jumps; multiplies the square of change
            in frequency. Trialworthy values: 0.5, 2, 5, 20, 40. Higher reduces
            odds of a ridge derailing to noise, but makes harder to track fast
            frequency changes.

        n_ridges: int
            Number of ridges to be calculated.

        bw: int
            Decides how many bins will be subtracted around max energy frequency
            bins when extracting multiple ridges (2 is standard for ssq'd).
            See "bw selection".

        transform: str['cwt', 'stft']
            Treats `scales` logarithmically if 'cwt', else linearly.
            `ssq_cwt` & `ssq_stft` are still 'cwt' & 'stft'.

        get_params: bool (default False)
            Whether to also compute and return `fridge` & `max_energy`.

    # Returns
        ridge_idxs: np.ndarray
            Indices for maximum frequency ridge(s).
        fridge: np.ndarray
            Frequencies tracking maximum frequency ridge(s).
        max_energy: np.ndarray [n_timeshifts x n_ridges]
            Energy maxima vectors along time axis.

    **bw selection**

    When a component is extracted, a region around it (a number of bins above
    and below the ridge) is zeroed and no longer affects next ridge's extraction.
        - higher: more bins subtracted, lesser chance of selecting the same
        component as the ridge.
        - lower:  less bins subtracted, lesser chance of dropping an unrelated
        component before the component is considered.
        - In general, set higher if more `scales` (or greater `nv`), or lower
        frequency resolution:
            - cwt:  `wavelets.freq_resolution(wavelet, N, nondim=False)`
            - stft: `utils.window_resolution(window)`
            - `N = utils.p2up(len(x))[0]`

    # References
        1. On the extraction of instantaneous frequencies from ridges in
        time-frequency representations of signals.
        D. Iatsenko, P. V. E. McClintock, A. Stefanovska.
        https://arxiv.org/pdf/1310.7276.pdf
    """
    def generate_penalty_matrix(scales, penalty):
        """Penalty matrix describes all potential penalties of  jumping from
        current frequency (first axis) to one or several new frequencies (second
        axis)

        `scales`: frequency scale vector from time-freq transform
        `penalty`: user-set penalty for freqency jumps (standard = 1.0)
        """
        # subtract.outer(A, B) = [[A[0] - B[0], A[0] - B[1], ...],
        #                         [A[1] - B[0], A[1] - B[1], ...],]
        dist_matrix = penalty * np.subtract.outer(scales, scales)**2
        return dist_matrix.squeeze()

    def fw_bw_ridge_tracking(energy_to_track, penalty_matrix):
        """Calculates acummulated penalty in forward (t=end...0) followed by
        backward (t=end...0) direction

        `energy`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies

        Returns: `ridge_idxs_fw_bw`: estimated forward backward frequency
                                     ridge indices
        """
        (penalized_energy_fw, ridge_idxs_fw
         ) = _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix)
        # backward calculation of frequency ridge (min log negative energy)
        ridge_idxs_fw_bw = _accumulated_penalty_energy_bw(
            energy_to_track, penalty_matrix, penalized_energy_fw, ridge_idxs_fw)

        return ridge_idxs_fw_bw

    scales = (np.log(scales) if transform == 'cwt' else
              scales)
    scales = scales.squeeze()
    energy = np.abs(Tf)**2
    n_timeshifts = Tf.shape[1]

    ridge_idxs = np.zeros((n_timeshifts, n_ridges), dtype=int)
    if get_params:
        fridge     = np.zeros((n_timeshifts, n_ridges))
        max_energy = np.zeros((n_timeshifts, n_ridges))

    penalty_matrix = generate_penalty_matrix(scales, penalty)

    for i in range(n_ridges):
        energy_max = energy.max(axis=0)
        energy_neg_log_norm = -np.log(energy / energy_max + EPS)

        ridge_idxs[:, i] = fw_bw_ridge_tracking(energy_neg_log_norm,
                                                penalty_matrix)
        if get_params:
            max_energy[:, i] = energy[ridge_idxs[:, i], range(n_timeshifts)]
            fridge[:, i] = scales[ridge_idxs[:, i]]

        for time_idx in range(n_timeshifts):
            ridx = ridge_idxs[time_idx, i]
            energy[int(ridx - bw):int(ridx + bw), time_idx] = 0

    return ((ridge_idxs, fridge, max_energy) if get_params else
            ridge_idxs)


def _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix):
    """Calculates acummulated penalty in forward direction (t=0...end).

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre-calculated penalty for all potential jumps between
                      two frequencies

    # Returns:
        `penalized_energy`: new energy with added forward penalty
        `ridge_idxs`: calculated initial ridge with only forward penalty
    """
    penalized_energy = energy_to_track.copy()
    penalized_energy = __accumulated_penalty_energy_fw(penalized_energy,
                                                       penalty_matrix)
    ridge_idxs = np.unravel_index(np.argmin(penalized_energy, axis=0),
                                  penalized_energy.shape)[1]
    return penalized_energy, ridge_idxs


@jit(nopython=True, cache=True)
def __accumulated_penalty_energy_fw(penalized_energy, penalty_matrix):
    for idx_time in range(1, penalized_energy.shape[1]):
        for idx_freq in range(0, penalized_energy.shape[0]):
            penalized_energy[idx_freq, idx_time
                             ] += np.amin(penalized_energy[:, idx_time - 1] +
                                          penalty_matrix[idx_freq, :])
    return penalized_energy


def _accumulated_penalty_energy_bw(energy_to_track, penalty_matrix,
                                   penalized_energy_fw, ridge_idxs_fw):
    """Calculates acummulated penalty in backward direction (t=end...0)

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre calculated penalty for all potential jumps between
                      two frequencies
    `ridge_idxs_fw`: calculated forward ridge

    Returns: `ridge_idxs_fw`: new ridge with added backward penalty, int array
    """
    pen_e = penalized_energy_fw
    e = energy_to_track
    ridge_idxs_fw = __accumulated_penalty_energy_bw(e, penalty_matrix, pen_e,
                                                    ridge_idxs_fw)
    ridge_idxs_fw = np.asarray(ridge_idxs_fw).astype(int)
    return ridge_idxs_fw


@jit(nopython=True, cache=True)
def __accumulated_penalty_energy_bw(e, penalty_matrix, pen_e, ridge_idxs_fw):
    for idx_time in range(e.shape[1] - 2, -1, -1):
        val = (pen_e[ridge_idxs_fw[idx_time + 1], idx_time + 1] -
               e[    ridge_idxs_fw[idx_time + 1], idx_time + 1])
        for idx_freq in range(e.shape[0]):
            new_penalty = penalty_matrix[ridge_idxs_fw[idx_time + 1], idx_freq]

            if abs(val - (pen_e[idx_freq, idx_time] + new_penalty)) < EPS:
                ridge_idxs_fw[idx_time] = idx_freq
    return ridge_idxs_fw
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:01:45 2020

@author: David Bondesson
"""

import numpy as np


def extract_fridges(tf_transf,frequency_scales,penalty=2.0,num_ridges=1,BW=25):
#   tracks frequency ridges by performing forward-backward 
#   ridge tracking algorithm:
#   Arguments:  tf_transf - complex time frequency representation
#               fs - frequency scales to calculate distance penalty term
#               penalty - integer value to penalise frequency jumps
#               num_ridges - number of ridges to be calculated
#               BW - decides how many bins will be subtracted around max 
#                    energy frequency bins when extracting multiple ridges (2 is standard value for syncrosqueezed transform)
#   outputs:    max_Energy (vector) along time axis
#               ridge_idx - indexes for maximum frequency ridge(s)
#               fridge - frequencies traccking maximum frequency ridge(s)
    
    
    def generate_penalty_matrix(frequency_scales,penalty):
    #   penalty matrix describes all potential penalties of  jumping from current frequency
    #   (first axis) to one or several new frequencies (second axis) 
    #   Arguments: frequency_scales - Frequency scale vector from time-freq transform
    #              penalty - user set penalty for freqency jumps (standard =1.0)
    #   outputs:   dist_matrix -penalty matrix
    #
        freq_scale=frequency_scales.copy()
        dist_matrix= np.square(np.subtract.outer(freq_scale,freq_scale))*penalty
    
        return dist_matrix
      
    def calculate_accumulated_penalty_energy_forwards(Energy_to_track,penalty_matrix):
    #   Calculates acummulated penalty in forward direction (t=0...end)
    #   Arguments:  Energy - squared abs time-frequency transform
    #               penalty_matrix - pre calculated penalty for all potential jumps between two frequencies
    #   outputs:    penalised_energy - new energy with added forward penalty
    #               ridge_idx - calculated initial ridge with only forward penalty
        penalised_energy=Energy_to_track.copy()     
       
        for idx_time in range(1,np.shape(penalised_energy)[1],1):
            for idx_freq in range(0,np.shape(penalised_energy)[0],1):     
                penalised_energy[idx_freq,idx_time]+=np.amin(penalised_energy[:,idx_time-1]+penalty_matrix[idx_freq,:])            
        
        ridge_idx=np.unravel_index(np.argmin(penalised_energy, axis=0), penalised_energy.shape)[1]
        
        return  penalised_energy,ridge_idx  
    
    def calculate_accumulated_penalty_energy_backwards(Energy_to_track,penalty_matrix,penalised_energy_frwd,ridge_idx_frwd):
    #   Calculates acummulated penalty in backward direction (t=end...0)
    #   Arguments:  Energy - squared abs time-frequency transform
    #               penalty_matrix - pre calculated penalty for all potential jumps between two frequencies
    #               ridge_idx_frwd - Calculated forward ridge 
    #   outputs:    ridge_idx_frwd - new ridge with added backward penalty
    #
        pen_e=penalised_energy_frwd.copy()
        e=Energy_to_track.copy()
        for idx_time in range(np.shape(e)[1]-2,-1,-1):
            val=pen_e[ridge_idx_frwd[idx_time+1],idx_time+1]-e[ridge_idx_frwd[idx_time+1],idx_time+1]
            for idx_freq in range(0,np.shape(e)[0],1):
                new_penalty=penalty_matrix[ridge_idx_frwd[idx_time+1],idx_freq]
        
                
                if(abs(val-(pen_e[idx_freq,idx_time]+new_penalty))<np.finfo(np.float64).eps):
                    ridge_idx_frwd[idx_time]=idx_freq
                    
        return  ridge_idx_frwd
    
    
    
    def frwd_bckwd_ridge_tracking(Energy_to_track,penalty_matrix):
    #   Calculates acummulated penalty in forward (t=end...0) followed by backward (t=end...0) direction 
    #   Arguments:  Energy - squared abs time-frequency transform
    #               penalty_matrix - pre calculated penalty for all potential jumps between two frequencies 
    #   outputs:    ridge_idx_frwd_bck - Estimated forward backward frequency ridge indices
      
        penalised_energy_frwd,ridge_idx_frwd=calculate_accumulated_penalty_energy_forwards(Energy_to_track,penalty_matrix)
        #    backward calculation of frequency ridge (min log negative energy)
        ridge_idx_frwd_bck=calculate_accumulated_penalty_energy_backwards(Energy_to_track,penalty_matrix,penalised_energy_frwd,ridge_idx_frwd)   
        
        return ridge_idx_frwd_bck
    
    
    
    Energy=np.square(np.abs(tf_transf))
    dim= Energy.shape
    ridge_idx = np.zeros((dim[1],num_ridges))
    max_Energy = np.zeros((dim[1],num_ridges))
    fridge = np.zeros((dim[1],num_ridges))
    
    penalty_matrix= np.squeeze(generate_penalty_matrix(frequency_scales,penalty))
    eps= np.finfo(np.float64).eps

     
    for current_ridge_index in range(0,num_ridges):
        energy_max=np.max(Energy,axis=0) 
        Energy_neg_log_norm=-np.log((Energy/energy_max)+eps)
 
      
        ridge_idx[:,current_ridge_index]=np.array(frwd_bckwd_ridge_tracking(Energy_neg_log_norm,penalty_matrix))
        ridge_idx=ridge_idx.astype(int)
       
        max_Energy[:,current_ridge_index]=Energy[ridge_idx[:,current_ridge_index],np.arange(len(ridge_idx[:,current_ridge_index]))]          
        fridge[:,current_ridge_index]=np.squeeze(frequency_scales[ridge_idx[:,current_ridge_index]])

        for time_idx in range(0,dim[1]):            
               Energy[int(ridge_idx[time_idx,current_ridge_index]-BW):int(ridge_idx[time_idx,current_ridge_index]+BW),time_idx]=0
    return max_Energy,ridge_idx,fridge
    

>>>>>>> b6bad10... updated ridge_tracking test with appropriate signal padding and test images
