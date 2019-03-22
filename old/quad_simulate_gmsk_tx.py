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
OVERSAMPLING = 16

BT = 0.3
PULSE_SPAN = 4

SPECTRAL_EFF = 1.0 # bits/s/Hz
BW = BIT_RATE/(SPECTRAL_EFF*2.0) # theoretical signal bandwidth
CARRIER = 10.0*BW

message = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)
message2 = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)

msk_i, msk_q = generate_msk_baseband(message, OVERSAMPLING)
msk_rf = upconvert_baseband(CARRIER, msk_i, msk_q)

gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT, pulse_span=PULSE_SPAN)
gmsk_i2, gmsk_q2 = generate_gmsk_baseband(message2, OVERSAMPLING, bt=BT, pulse_span=PULSE_SPAN)
gmsk_q2.td *= -1.0
gmsk_rf = upconvert_baseband(CARRIER, sum_signals(gmsk_i, gmsk_q2), sum_signals(gmsk_q, gmsk_i2))

plt.subplot(2,2,1)
plot_constellation_density(gmsk_i, gmsk_q)
plt.subplot(2,2,2)
plot_iq_phase_mag(gmsk_i, gmsk_q)
plt.subplot(2,2,3)
plot_phase_histogram(gmsk_i, gmsk_q)

msk_rf = add_noise(msk_rf, rms=0.05)
gmsk_rf = add_noise(gmsk_rf, rms=0.05)

#plt.figure(0)
#plot_td(gmsk_rf)
#plt.show()
#plt.figure(1)
plt.subplot(2,2,4)
plot_fd(msk_rf)
plot_fd(gmsk_rf)
plt.show()

# gmsk_rf = gaussian_fade(gmsk_rf, f = 1000)

n = int(round(gmsk_rf.fs/float(gmsk_i.fs)))
rx_i, rx_q = downconvert_rf(CARRIER, gmsk_rf)
#downsample to original IQ rate
rx_i = filter_and_downsample(rx_i, n=n)
rx_q = filter_and_downsample(rx_q, n=n)

rx_i = butter_lowpass_filter(rx_i, cutoff = 2.0*BW)
rx_q = butter_lowpass_filter(rx_q, cutoff = 2.0*BW)
plt.subplot(2,2,1)
plot_constellation_density(rx_i, rx_q)
plt.subplot(2,2,2)
plot_iq_phase_mag(rx_i, rx_q)
plt.subplot(2,2,3)
plot_phase_histogram(rx_i, rx_q)

demodulated = demodulate_gmsk(rx_i, rx_q, OVERSAMPLING)
plt.subplot(2,2,4)
plot_td(demodulated)

plt.show()
