import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import *
from lib.modulation import *
from lib.plot import *
from lib.transforms import *

def upsample(signal, factor):
    upsampled = np.zeros(len(signal)*factor)
    upsampled[np.arange(len(signal))*factor] = signal
    return upsampled

def pm_map(bits):
    bits[bits<=0] = -1
    bits[bits>0] = 1
    return bits

N_BITS = 1000
BIT_RATE = 64000 + 2400
OVERSAMPLING = 100

SPECTRAL_EFF = 1.0 # bits/s/Hz
BW = BIT_RATE/(SPECTRAL_EFF*2.0) # theoretical signal bandwidth
CARRIER = 20.0*BW

message = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)

msk_i, msk_q = generate_msk_baseband(message, OVERSAMPLING)
msk_rf = upconvert_baseband(CARRIER, msk_i, msk_q)

gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=0.3, pulse_span=4)
gmsk_rf = upconvert_baseband(CARRIER, gmsk_i, gmsk_q)

plt.figure(0)
plot_constellation_density(gmsk_i, gmsk_q)
plt.figure(1)
plot_iq_phase_mag(gmsk_i, gmsk_q)
plt.figure(2)
plot_phase_histogram(gmsk_i, gmsk_q)

msk_rf = add_noise(msk_rf, rms=0.01)
gmsk_rf = add_noise(gmsk_rf, rms=0.01)

#plt.figure(0)
#plot_td(gmsk_rf)
#plt.show()
#plt.figure(1)
plt.figure(3)
plot_fd(msk_rf)
plot_fd(gmsk_rf)
plt.show()


