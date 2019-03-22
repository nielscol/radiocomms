""" Methods for plotting data in Signal objects
    Cole Nielsen 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from lib.tools import *

def plot_td(signal, verbose=True, label="", *args, **kwargs):
    if verbose:
        print("\n* Plotting signal %s in time domain"%signal.name)
    times = np.arange(signal.samples)/float(signal.fs)
    plt.ylabel("Signal")
    plt.xlabel("Time [s]")
    plt.grid()
    plt.plot(times, signal.td, label=signal.name)
    plt.title("Time domain")

def plot_fd(signal, log=True, label="", verbose=True, *args, **kwargs):
    if not any(signal.fd): # freq. domain not calculated
        print("\n* Calculating frequency domain representation of signal. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    if verbose:
        print("\n* Plotting signal %s in frequency domain"%signal.name)
    freqs = (np.arange(signal.samples) - (signal.samples/2)) * signal.fbin
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    if log:
        if verbose:
            print("\tFFT - Log [dB] scale")
        plt.ylabel("FFT(Signal) [dB]")
        plt.plot(freqs, 20*np.log10(np.abs(np.fft.fftshift(signal.fd))), label=label)
    else:
        if verbose:
            print("\tFFT - Magnitude")
        plt.ylabel("FFT(Signal) [magnitude]")
        plt.plot(freqs, np.abs(np.fft.fftshift(signal.fd)), label = label)
    plt.title("Power Spectral Density")
    plt.legend()

def plot_constellation(i, q, verbose=True, label="", *args, **kwargs):
    if verbose:
        print("\n* Plotting IQ signal constellation")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    plt.plot(i.td, q.td, label=label)
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.grid()
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("IQ Constellation")
    plt.legend()

def plot_iq_phase_mag(i, q, verbose=True, label="", *args, **kwargs):
    if verbose:
        print("\n* Plotting IQ signal phase and magnitude")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    times = np.arange(i.samples)/float(i.fs)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.plot(times, np.arctan2(q.td, i.td), label=label+" Phase")
    plt.plot(times, np.hypot(i.td, q.td), label=label+" Magnitude")
    plt.title("IQ Phase and Magnitude")
    plt.legend()

def plot_phase_histogram(i, q, verbose=True, label="", *args, **kwargs):
    if verbose:
        print("\n* Plotting IQ signal phase histrogram")
        print("\tI.name = %s"%i.name)
        print("\tQ.name = %s"%q.name)
    plt.hist(np.arctan2(q.td, i.td), bins=128, density=True, label=label)
    plt.xlabel("IQ Phase")
    plt.ylabel("Density")
    plt.title("IQ Phase Histrogram")
    plt.legend()

def iq_to_coordinate(i, q, ax_dim):
    bin_size = 2.0/float(ax_dim)
    ii = int(round((i + 1.0)/bin_size))
    qq = int(round((q + 1.0)/bin_size))
    ii = ax_dim - 1 if ii >= ax_dim else ii
    qq = ax_dim - 1 if qq >= ax_dim else qq
    ii = 0 if ii < 0 else ii
    qq = 0 if qq < 0 else qq
    return (ii,qq)

def plot_constellation_density(i, q, verbose=True, ax_dim=128, label="", *args, **kwargs):
    im = np.zeros((ax_dim,ax_dim))

    for n, ii in enumerate(i.td):
        im[iq_to_coordinate(ii, q.td[n], ax_dim)] += 1
    plt.imshow(np.log(im+1), cmap="inferno", interpolation="gaussian")
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.title("IQ Constellation")


def crossing_times(signal_td):
    crossings = np.where(np.diff(np.sign(signal_td)))[0] # returns index of values before or at crossings
    n_crossings = len(crossings)
    frac_crossings = []
    td = signal_td
    td_len = len(signal_td)
    for cross_n, td_n in enumerate(crossings):
        if td_n+1 < td_len and td[td_n+1] != 0.0:
            frac_crossings.append(td_n-(td[td_n]/float(td[td_n+1]-td[td_n])))
        elif td_n > 0 and td[td_n] == 0.0 and td[td_n-1] != 0:
            frac_crossings.append(float(td_n))
        elif td_n+1 < td_len and cross_n+1 < n_crossings and td[td_n] != 0.0 and td[td_n+1] == 0 and td_n+1 != crossings[cross_n+1]:
            frac_crossings.append(float(td_n+1))
    return frac_crossings


def line_eye(signal, interp_factor=10, interp_span=128, remove_ends=100):
    td = signal.td[remove_ends:]
    td = td[:-remove_ends]
    interpolated = sinx_x_interp(td, interp_factor, interp_span)
    ui_samples = interp_factor*signal.fs/float(signal.bitrate)
    crossings = crossing_times(interpolated)
    td = interpolated
    td_len = len(interpolated)
    _ui_samples = round(ui_samples)
    for crossing in crossings:
        if round(crossing) > _ui_samples and round(crossing) < td_len - 2*_ui_samples:
            _slice = td[int(round(crossing)-_ui_samples):int(round(crossing)+2*_ui_samples)]
            time = np.arange(len(_slice)) - _ui_samples - (crossing-round(crossing))
            time /= float(ui_samples)
            plt.plot(time, _slice)
    plt.xlim((-0.5,1.5))
