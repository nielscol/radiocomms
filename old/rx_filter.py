import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import *
from lib.modulation import *
from lib.plot import *
from lib.transforms import *


BT = 0.3
OVERSAMPLING = 1
pulse_span = 32

def gaussian_to_raised_cosine(bt, oversampling, pulse_span):
    # Make GMSK pulse shape
    pulse_len = int(oversampling*pulse_span)
    t = (np.arange(pulse_len)-pulse_len/2)/float(oversampling)
    gmsk_pulse_shape = 2*v_gmsk_pulse(t, bt, 1.0)
    # add an offset to pulse so sum of samples = 1.0
    gmsk_pulse_error = np.sum(gmsk_pulse_shape)-oversampling
    gmsk_pulse_shape -= gmsk_pulse_error/float(pulse_len)
    rc_pulse_shape = v_raised_cos(t, 1.0, 1.0)
    gmsk_pulse_shape += np.random.normal(0,1e-9,pulse_len)
    fft = np.fft.fft(rc_pulse_shape)/np.fft.fft(gmsk_pulse_shape)
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(gmsk_pulse_shape))))
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(rc_pulse_shape))))
    plt.plot(np.abs(np.fft.fftshift(fft)))
    plt.show()
    fir = np.fft.ifft(np.fft.fft(rc_pulse_shape)/np.fft.fft(gmsk_pulse_shape))
    fir = np.fft.fftshift(fir)
    # add an offset to pulse so sum of samples = 1.0
    #pulse_error = np.sum(pulse_shape)-oversampling
    #pulse_shape -= pulse_error/float(pulse_len)
    return fir

fir = gaussian_to_raised_cosine(BT, OVERSAMPLING, pulse_span)

