""" Methods for plotting data in Signal objects
    Cole Nielsen 2019
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_td(signal, verbose=True, label="", *args, **kwargs):
    if verbose:
        print("\n* Plotting signal %s in time domain"%signal.name)
    times = np.arange(signal.samples)/float(signal.fs)
    plt.ylabel("Signal")
    plt.xlabel("Time [s]")
    plt.plot(times, signal.td, label=signal.name)

def plot_fd(signal, log=True, label="", verbose=True, *args, **kwargs):
    if not any(signal.fd): # freq. domain not calculated
        print("\n* Calculating frequency domain representation of signal. This may be slow...")
        signal.fd = np.fft.fft(signal.td)
    if verbose:
        print("\n* Plotting signal %s in frequency domain"%signal.name)
    freqs = (np.arange(signal.samples) - (signal.samples/2)) * signal.fbin
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
