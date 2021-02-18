# -*- coding: utf-8 -*-
<<<<<<< HEAD
<<<<<<< HEAD
"""Coverage-mainly tests; see examples/extracting_ridges.py for more test cases.
"""
import pytest
import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from ssqueezepy.visuals import plot, imshow

# set to 1 to run tests as functions, showing plots
VIZ = 0
=======
=======
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py
import pytest
import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, extract_ridges
from ssqueezepy.visuals import plot, imshow

# set to 1 to run tests as functions, showing plots
<<<<<<< HEAD
VIZ = 1
=======
VIZ = 0
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py


def viz(signal, Tf, ridge, yticks=None, ssq=False):
    if not VIZ:
        return
    plot(signal, title="Time signal", show=1,
         xlabel="Time axis [A.U.]",
         ylabel="Signal Amplitude [A.U.]")

    ikw = dict(abs=1, cmap='jet', yticks=yticks, show=0)
    pkw = dict(linestyle='--', color='k', show=1,
               xlabel="Time axis [A.U.]")
    if ssq:
        imshow(np.flipud(Tf), title="abs(SSQ_CWT) w/ ridge", **ikw)
        plot(len(Tf) - ridge, **pkw, ylabel="Frequencies [A.U.]")
    else:
        imshow(Tf, title="abs(CWT) w/ ridge", **ikw)
        plot(ridge, **pkw, ylabel="Frequency scales [A.U.]")
<<<<<<< HEAD
>>>>>>> 66ee952... added example for changing penalty term and rewrote README for example purposes
=======
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py


def test_basic():
    """Example ridge from similar example as can be found at MATLAB:
    https://www.mathworks.com/help/wavelet/ref/wsstridge.html#bu6we25-penalty
    """
<<<<<<< HEAD
<<<<<<< HEAD
    test_matrix = np.array([[1, 4, 4], [2, 2, 2], [5, 5, 4]])
    fs_test = np.exp([1, 2, 3])

    ridge_idxs, *_ = extract_ridges(test_matrix, fs_test, penalty=2.0,
                                    get_params=True)
    assert np.allclose(ridge_idxs, np.array([[2, 2, 2]]))


def test_poly():
    """Cubic polynomial frequency variation + pure tone."""
    N, f = 257, 0.5
    padtype = 'wrap'
    penalty = 2.0

    t  = np.linspace(0, 10, N, endpoint=True)
    p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])
    x1 = sig.sweep_poly(t, p1)
    x2 = np.sin(2*np.pi * f * t)
    x = x1 + x2

    tf_transforms(x, t, padtype=padtype, stft_bw=4, penalty=penalty)


def viz(x, Tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=True):
    if not VIZ:
        return
    if show_x:
        plot(x, title="x(t)", show=1,
             xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]")

    if transform == 'cwt' and not ssq:
        Tf = np.flipud(Tf)
        ridge_idxs = len(Tf) - ridge_idxs

    ylabel = ("Frequency scales [1/Hz]" if (transform == 'cwt' and not ssq) else
              "Frequencies [Hz]")
    title = "abs({}{}) w/ ridge_idxs".format("SSQ_" if ssq else "",
                                             transform.upper())

    ikw = dict(abs=1, cmap='jet', yticks=yticks, title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel,
               xlims=(0, Tf.shape[1]), ylims=(0, len(Tf)))

    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)


def tf_transforms(x, t, wavelet='morlet', window=None, padtype='wrap',
                  penalty=.5, n_ridges=2, cwt_bw=15, stft_bw=15,
                  ssq_cwt_bw=4, ssq_stft_bw=4):
    kw_cwt  = dict(t=t, padtype=padtype)
    kw_stft = dict(fs=1/(t[1] - t[0]), padtype=padtype)
    Twx, Wx, ssq_freqs_c, scales, *_ = ssq_cwt(x,  wavelet, **kw_cwt)
    Tsx, Sx, ssq_freqs_s, Sfs, *_    = ssq_stft(x, window,  **kw_stft)

    ckw = dict(penalty=penalty, n_ridges=n_ridges, transform='cwt')
    skw = dict(penalty=penalty, n_ridges=n_ridges, transform='stft')
    cwt_ridges      = extract_ridges(Wx,  scales,      bw=cwt_bw,      **ckw)
    ssq_cwt_ridges  = extract_ridges(Twx, ssq_freqs_c, bw=ssq_cwt_bw,  **ckw)
    stft_ridges     = extract_ridges(Sx,  Sfs,         bw=stft_bw,     **skw)
    ssq_stft_ridges = extract_ridges(Tsx, ssq_freqs_s, bw=ssq_stft_bw, **skw)

    viz(x, Wx,  cwt_ridges,      scales,      ssq=0, transform='cwt',  show_x=1)
    viz(x, Twx, ssq_cwt_ridges,  ssq_freqs_c, ssq=1, transform='cwt',  show_x=0)
    viz(x, Sx,  stft_ridges,     Sfs,         ssq=0, transform='stft', show_x=0)
    viz(x, Tsx, ssq_stft_ridges, ssq_freqs_s, ssq=1, transform='stft', show_x=0)

=======
=======
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py
    test_matrix = np.array([[1, 4, 4],[2, 2, 2],[5,5,4]])
    fs_test = np.exp([1,2,3])

    ridge_idxs, *_ = extract_ridges(test_matrix, fs_test, penalty=2.0)
    print('ridge follows indexes:', ridge_idxs)


def test_sine():
    """Sine + cosine."""
    sig_len, f1, f2 = 500, 0.5, 2.0
    padtype = 'wrap'
<<<<<<< HEAD
    penalty = 2.0
=======
    penalty = 20
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py

    t_vec = np.linspace(0, 10, sig_len, endpoint=True)
    x1 = np.sin(2*np.pi * f1 * t_vec)
    x2 = np.cos(2*np.pi * f2 * t_vec)
    x = x1 + x2

<<<<<<< HEAD
    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    ridge_idxs, _,max_energy = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs, scales)

    # SSQ_CWT example
    ridge_idxs, _,max_energy = extract_ridges(Tx, ssq_freq, penalty, n_ridges=2, BW=4)
    viz(x, Tx, ridge_idxs, ssq_freq, ssq=True)
=======
    Tx, ssq_freqs, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs, scales)

    # SSQ_CWT example
    ridge_idxs, *_ = extract_ridges(Tx, scales, penalty, n_ridges=2, BW=4)
    viz(x, Tx, ridge_idxs, ssq_freqs, ssq=True)
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py


def test_chirp():
    """Linear + quadratic chirp."""
    sig_len = 500
<<<<<<< HEAD
    penalty = 0.1
=======
    penalty = 0.5
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py
    padtype = 'reflect'

    t_vec = np.linspace(0, 10, sig_len, endpoint=True)
    x1 = sig.chirp(t_vec, f0=2,  f1=8, t1=20, method='linear')
    x2 = sig.chirp(t_vec, f0=.4, f1=4, t1=20, method='quadratic')
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)

    # SSQ_CWT example
<<<<<<< HEAD
    ridge_idxs, *_ = extract_ridges(Tx, ssq_freq, penalty, n_ridges=2, BW=2)
=======
    ridge_idxs, *_ = extract_ridges(Tx, scales, penalty, n_ridges=2, BW=2)
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py
    viz(x, Tx, ridge_idxs, ssq=True)


def test_poly():
    """Cubic polynomial frequency variation + pure tone."""
    sig_len, f = 500, 0.5
    padtype = 'wrap'

    t_vec = np.linspace(0, 10, sig_len, endpoint=True)
    p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])
    x1 = sig.sweep_poly(t_vec, p1)
    x2 = np.sin(2*np.pi * f * t_vec)
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    penalty = 2.0
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)

    # SSQ_CWT example
<<<<<<< HEAD
    ridge_idxs, *_ = extract_ridges(Tx, ssq_freq, penalty, n_ridges=2, BW=2)
    viz(x, Tx, ridge_idxs, ssq=True)

def test_failed_chirp_wsst():
    """Linear + quadratic chirp."""
    sig_len = 600
    padtype = 'symmetric'
    t_vec = np.linspace(0, 3, sig_len, endpoint=True)
  
    x1 = sig.chirp(t_vec-1.5, f0=30,  t1=1.1,f1=40,  method='quadratic')
    x2 = sig.chirp(t_vec-1.5,f0=10,  t1=1.1,f1=5, method='quadratic')
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example no penalty
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty=0.0, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)

    # CWT example with penalty
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty=0.5, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)
    
    
>>>>>>> 66ee952... added example for changing penalty term and rewrote README for example purposes
=======
    ridge_idxs, *_ = extract_ridges(Tx, scales, penalty, n_ridges=2, BW=2)
    viz(x, Tx, ridge_idxs, ssq=True)

>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py

if __name__ == '__main__':
    if VIZ:
        test_basic()
<<<<<<< HEAD
<<<<<<< HEAD
        test_poly()
=======
        test_sine()
        test_chirp()
        test_poly()
        test_failed_chirp_wsst()
>>>>>>> 66ee952... added example for changing penalty term and rewrote README for example purposes
    else:
        pytest.main([__file__, "-s"])
=======
        test_sine()
        test_chirp()
        test_poly()
    else:
        pytest.main([__file__, "-s"])
>>>>>>> cfc25f1... Update and rename ridge_extract_test.py to ridge_extraction_test.py
