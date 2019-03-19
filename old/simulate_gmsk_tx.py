import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import *
from lib.modulation import *
from lib.plot import *

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
SPECTRAL_EFF = 1.0 # bits/s/Hz
OVERSAMPLING = 100
message = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)

BW = BIT_RATE/(SPECTRAL_EFF*2.0) # max deviation from carrier
CARRIER = 20.0*BW

bb_i, bb_q = generate_msk_baseband(message, OVERSAMPLING)
rf = upconvert_baseband(CARRIER, bb_i, bb_q)

plot_fd(rf)
plt.show()
foo()

###############################################################
# OLD ATTEMPTS AT MSK
###############################################################

pm_mapped_message = pm_map(message)

t_bit = 1.0/float(BIT_RATE)
t_sample = 1/float(BIT_RATE*OVERSAMPLING)

#signal = np.zeros(SAMPLES_PER_BIT*len(BITS))

pulse_shape = np.ones(OVERSAMPLING)
bb_message = upsample(pm_mapped_message, OVERSAMPLING)
bb_message = np.convolve(bb_message, pulse_shape, mode="full")
msk_phase = (pi/2.0)*np.cumsum(bb_message)/float(OVERSAMPLING)
bbI = np.cos(msk_phase)
bbQ = np.sin(msk_phase)
plt.plot(bbI)
plt.plot(bbQ)
plt.plot(bbI**2 + bbQ**2)
plt.show()

time = np.arange(len(bbI))*t_sample
rfI = np.cos(2*pi*CARRIER*time)*bbI
rfQ = -np.sin(2*pi*CARRIER*time)*bbQ
rf = rfI + rfQ
plt.plot(rf)
plt.show()

plt.plot(20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(rf)))))
plt.show()

bit_last = 0
phase = 0.0


foo()

bit_last = 0
for n, bit in enumerate(BITS):
    for m in range(SAMPLES_PER_BIT):
        if bit and not bit_last:
            phase += 2*pi*(-np.cos(pi*m/float(SAMPLES_PER_BIT))*F_DEV+CARRIER)*t_sample
            #phase += 2*pi*((2*m/float(SAMPLES_PER_BIT)-1)*F_DEV+CARRIER)*t_sample
        elif bit and bit_last:
            phase += 2*pi*(F_DEV+CARRIER)*t_sample
        elif not bit and bit_last:
            phase += 2*pi*(np.cos(pi*m/float(SAMPLES_PER_BIT))*F_DEV+CARRIER)*t_sample
            #phase += 2*pi*((-2*pi*m/float(SAMPLES_PER_BIT)+1)*F_DEV+CARRIER)*t_sample
        elif not bit and not bit_last:
            phase += 2*pi*(F_DEV-CARRIER)*t_sample
        signal[n*SAMPLES_PER_BIT+m] = phase
    bit_last = bit
b = np.cos(signal)
plt.figure(0)
plt.plot(b)
#plt.plot(signal - pi*CARRIER*t_sample*np.arange(len(signal)))
#plt.plot(b*np.cos(2*pi*CARRIER*t_sample*np.arange(len(signal))))
s = b*np.cos(2*pi*100*CARRIER*t_sample*np.arange(len(signal)))
#plt.plot(s)
plt.figure(1)
plt.plot(20*np.log10(np.fft.fftshift(np.abs(np.fft.fft(b)))))
plt.plot(20*np.log10(np.fft.fftshift(np.abs(np.fft.fft(s)))))
plt.show()

