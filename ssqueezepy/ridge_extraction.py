<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""Authors: David Bondesson, OverLordGoldDragon
<<<<<<< HEAD

=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
Ridge extraction from time-frequency representations (STFT, CWT, synchrosqueezed).
"""
import numpy as np
from numba import jit

EPS = np.finfo(np.float64).eps


def extract_ridges(Tf, scales, penalty=2., n_ridges=1, bw=15, transform='cwt',
                   get_params=False):
    """Tracks time-frequency ridges by performing forward-backward ridge tracking
    algorithm, based on ref [1] (a version of Eq. III.4).
<<<<<<< HEAD

    Also see: https://www.mathworks.com/help/signal/ref/tfridge.html

=======
    Also see: https://www.mathworks.com/help/signal/ref/tfridge.html
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
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
<<<<<<< HEAD

=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
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

<<<<<<< HEAD
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
=======
>>>>>>> 66ee952... added example for changing penalty term and rewrote README for example purposes
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
<<<<<<< HEAD
<<<<<<< HEAD
    algorithm, based on ref [1].
<<<<<<< HEAD
<<<<<<< HEAD
    # Arguments
=======
=======
=======
    algorithm, based on ref [1] (version of eq. III.4 in publication).
>>>>>>> 28772d9... added equation ref in ridge_extraction.py
=======
    algorithm, based on ref [1] (a version of Eq. III.4).

    Also see: https://www.mathworks.com/help/signal/ref/tfridge.html
>>>>>>> 2b676de... Performance optimizations; added author

>>>>>>> 984d549... Add STFT support, reformat code
    # Arguments:
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
# -*- coding: utf-8 -*-
import numpy as np

EPS = np.finfo(np.float64).eps

<<<<<<< HEAD

def extract_ridges(Tf, scales, penalty=2., n_ridges=1, BW=25):
    """Tracks time-frequency ridges by performing forward-backward ridge tracking
    algorithm, based on ref [1].

    # Arguments
        Tf: np.ndarray
            Complex time-frequency representation.

        scales:
            Frequency scales to calculate distance penalty term.

        penalty: float
            Value to penalise frequency jumps.

        n_ridges: int
            Number of ridges to be calculated.

>>>>>>> 037161f... Update ridge_extraction.py
=======
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======

>>>>>>> 984d549... Add STFT support, reformat code
        BW: int
            Decides how many bins will be subtracted around max
            energy frequency bins when extracting multiple ridges
            (2 is standard value for syncrosqueezed transform).
=======
        bw: int
            Decides how many bins will be subtracted around max energy frequency
            bins when extracting multiple ridges (2 is standard for ssq'd).
            See "bw selection".
>>>>>>> 0bff8ec... `bw` docstring, styling

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 037161f... Update ridge_extraction.py
=======
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======
        transform: str['cwt', 'stft']
            Treats `scales` logarithmically if 'cwt', else linearly.
            `ssq_cwt` & `ssq_stft` are still 'cwt' & 'stft'.

        get_params: bool (default False)
            Whether to also compute and return `fridge` & `max_energy`.

>>>>>>> 984d549... Add STFT support, reformat code
    # Returns
        ridge_idxs: np.ndarray
            Indices for maximum frequency ridge(s).
        fridge: np.ndarray
            Frequencies tracking maximum frequency ridge(s).
        max_energy: np.ndarray [n_timeshifts x n_ridges]
            Energy maxima vectors along time axis.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 037161f... Update ridge_extraction.py
=======

>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======
>>>>>>> 984d549... Add STFT support, reformat code
=======
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

>>>>>>> 0bff8ec... `bw` docstring, styling
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
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
>>>>>>> 037161f... Update ridge_extraction.py
=======
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======
        Arguments:
>>>>>>> 4c0aeee... syntax and grammar changes in  ridge_extract_readme and ride_extrction.py
=======

>>>>>>> 984d549... Add STFT support, reformat code
        `scales`: frequency scale vector from time-freq transform
        `penalty`: user-set penalty for freqency jumps (standard = 1.0)
        """
        # subtract.outer(A, B) = [[A[0] - B[0], A[0] - B[1], ...],
        #                         [A[1] - B[0], A[1] - B[1], ...],]
        dist_matrix = penalty * np.subtract.outer(scales, scales)**2
        return dist_matrix.squeeze()

<<<<<<< HEAD
    def accumulated_penalty_energy_fw(energy_to_track, penalty_matrix):
        """Calculates acummulated penalty in forward direction (t=0...end).

        `energy_to_track`: squared abs time-frequency transform
        `penalty_matrix`: pre-calculated penalty for all potential jumps between
                          two frequencies

<<<<<<< HEAD
        `energy_to_track`: squared abs time-frequency transform
        `penalty_matrix`: pre-calculated penalty for all potential jumps between
                          two frequencies

>>>>>>> 037161f... Update ridge_extraction.py
        # Returns
=======
        # Returns:
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
            `penalized_energy`: new energy with added forward penalty
            `ridge_idxs`: calculated initial ridge with only forward penalty
        """
        penalized_energy = energy_to_track.copy()

        for idx_time in range(1, penalized_energy.shape[1]):
            for idx_freq in range(0, penalized_energy.shape[0]):
                penalized_energy[idx_freq, idx_time
                                 ] += np.amin(penalized_energy[:, idx_time - 1] +
                                              penalty_matrix[idx_freq, :])

        ridge_idxs = np.unravel_index(np.argmin(penalized_energy, axis=0),
<<<<<<< HEAD
<<<<<<< HEAD
                                      penalized_energy.shape)[1]
=======
                                     penalized_energy.shape)[1]
>>>>>>> 037161f... Update ridge_extraction.py
=======
                                      penalized_energy.shape)[1]
>>>>>>> 68c0275... Update ridge_extraction.py

        return penalized_energy, ridge_idxs

    def accumulated_penalty_energy_bw(energy_to_track, penalty_matrix,
                                      penalized_energy_fw, ridge_idxs_fw):
        """Calculates acummulated penalty in backward direction (t=end...0)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 037161f... Update ridge_extraction.py
=======
        Arguments:
<<<<<<< HEAD
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
        `energy_to_track`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies
        `ridge_idxs_fw`: calculated forward ridge

<<<<<<< HEAD
>>>>>>> 037161f... Update ridge_extraction.py
        Returns: `ridge_idxs_fw`: new ridge with added backward penalty, int array
=======
        Returns: 
=======
            `energy_to_track`: squared abs time-frequency transform
            `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies
        `ridge_idxs_fw`: calculated forward ridge

        # Returns:
>>>>>>> 4c0aeee... syntax and grammar changes in  ridge_extract_readme and ride_extrction.py
            `ridge_idxs_fw`: new ridge with added backward penalty, int array
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======

        `energy_to_track`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies
        `ridge_idxs_fw`: calculated forward ridge

        Returns: `ridge_idxs_fw`: new ridge with added backward penalty, int array
>>>>>>> 984d549... Add STFT support, reformat code
        """
        pen_e = penalized_energy_fw
        e = energy_to_track

        for idx_time in range(e.shape[1] - 2, -1, -1):
            val = (pen_e[ridge_idxs_fw[idx_time + 1], idx_time + 1] -
                   e[    ridge_idxs_fw[idx_time + 1], idx_time + 1])
            for idx_freq in range(e.shape[0]):
                new_penalty = penalty_matrix[ridge_idxs_fw[idx_time + 1],
                                             idx_freq]

                if abs(val - (pen_e[idx_freq, idx_time] + new_penalty)) < EPS:
                    ridge_idxs_fw[idx_time] = idx_freq

        ridge_idxs_fw = np.asarray(ridge_idxs_fw).astype(int)
        return ridge_idxs_fw

=======
>>>>>>> 2b676de... Performance optimizations; added author
=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
    def fw_bw_ridge_tracking(energy_to_track, penalty_matrix):
        """Calculates acummulated penalty in forward (t=end...0) followed by
        backward (t=end...0) direction
        `energy`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies
<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        `energy`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies

>>>>>>> 037161f... Update ridge_extraction.py
=======
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
        Returns: `ridge_idxs_fw_bw`: estimated forward backward frequency
=======
        # Returns:
            `ridge_idxs_fw_bw`: estimated forward backward frequency
>>>>>>> 4c0aeee... syntax and grammar changes in  ridge_extract_readme and ride_extrction.py
                                    ridge indices
=======
=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
        Returns: `ridge_idxs_fw_bw`: estimated forward backward frequency
                                     ridge indices
>>>>>>> 984d549... Add STFT support, reformat code
        """
        (penalized_energy_fw, ridge_idxs_fw
         ) = _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix)
        # backward calculation of frequency ridge (min log negative energy)
        ridge_idxs_fw_bw = _accumulated_penalty_energy_bw(
            energy_to_track, penalty_matrix, penalized_energy_fw, ridge_idxs_fw)

        return ridge_idxs_fw_bw

<<<<<<< HEAD
    scales = np.log(scales)
    scales=scales.squeeze()
<<<<<<< HEAD
=======
    scales = scales.squeeze()
>>>>>>> 037161f... Update ridge_extraction.py
=======
>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======
    scales = (np.log(scales) if transform == 'cwt' else
              scales)
    scales = scales.squeeze()
>>>>>>> 984d549... Add STFT support, reformat code
    energy = np.abs(Tf)**2
    n_timeshifts = Tf.shape[1]

    ridge_idxs = np.zeros((n_timeshifts, n_ridges), dtype=int)
    if get_params:
        fridge     = np.zeros((n_timeshifts, n_ridges))
        max_energy = np.zeros((n_timeshifts, n_ridges))

    penalty_matrix = generate_penalty_matrix(scales, penalty)

    for i in range(n_ridges):
        energy_max = energy.max(axis=0)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        energy_neg_log_norm = -np.log(energy / energy_max  + EPS)

        ridge_idxs[:, current_ridge_idxs
                   ] = fw_bw_ridge_tracking(energy_neg_log_norm, penalty_matrix)
=======
=======

>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
        energy_neg_log_norm = -np.log(energy / energy_max + EPS)

        ridge_idxs[:, current_ridge_idxs
<<<<<<< HEAD
                  ] = fw_bw_ridge_tracking(energy_neg_log_norm, penalty_matrix)
>>>>>>> 037161f... Update ridge_extraction.py
=======
                   ] = fw_bw_ridge_tracking(energy_neg_log_norm, penalty_matrix)
>>>>>>> 68c0275... Update ridge_extraction.py
        max_energy[:, current_ridge_idxs
                   ] = energy[ridge_idxs[:, current_ridge_idxs],
                              np.arange(len(ridge_idxs[:, current_ridge_idxs]))]
        fridge[:, current_ridge_idxs] = scales[ridge_idxs[:, current_ridge_idxs]]
=======
        energy_neg_log_norm = -np.log(energy / energy_max + EPS)

        ridge_idxs[:, i] = fw_bw_ridge_tracking(energy_neg_log_norm,
                                                penalty_matrix)
        if get_params:
            max_energy[:, i] = energy[ridge_idxs[:, i], range(n_timeshifts)]
            fridge[:, i] = scales[ridge_idxs[:, i]]
>>>>>>> 984d549... Add STFT support, reformat code

        for time_idx in range(n_timeshifts):
            ridx = ridge_idxs[time_idx, i]
            energy[int(ridx - bw):int(ridx + bw), time_idx] = 0

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> b6bad10... updated ridge_tracking test with appropriate signal padding and test images
=======
    return ridge_idxs, fridge, max_energy
>>>>>>> 66ee952... added example for changing penalty term and rewrote README for example purposes
=======
    return ridge_idxs, fridge, max_energy
>>>>>>> 037161f... Update ridge_extraction.py
=======
    return ridge_idxs, fridge, max_energy

>>>>>>> cf17b5b... merge review changes of initial PR 11.02.2021
=======
    return ((ridge_idxs, fridge, max_energy) if get_params else
            ridge_idxs)
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 984d549... Add STFT support, reformat code
=======
=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba


def _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix):
    """Calculates acummulated penalty in forward direction (t=0...end).
<<<<<<< HEAD

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre-calculated penalty for all potential jumps between
                      two frequencies

=======
    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre-calculated penalty for all potential jumps between
                      two frequencies
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
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
<<<<<<< HEAD

=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre calculated penalty for all potential jumps between
                      two frequencies
    `ridge_idxs_fw`: calculated forward ridge
<<<<<<< HEAD

=======
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
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
<<<<<<< HEAD
    return ridge_idxs_fw
>>>>>>> 2b676de... Performance optimizations; added author
=======
    return ridge_idxs_fw
>>>>>>> b23f128... adapted sugested style changes and optimization with jit from numba
