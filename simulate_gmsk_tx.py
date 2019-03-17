import numpy as np
import matplotlib.pyplot as plt
from math import pi

BIT_RATE = 64000 + 2400
SPECTRAL_EFF = 1.0 # bits/s/Hz

SAMPLES_PER_BIT = 10000
BITS = np.random.choice([0,1], 100)

t_bit = 1.0/float(BIT_RATE)
t_sample = 1/float(BIT_RATE*SAMPLES_PER_BIT)

signal = np.zeros(SAMPLES_PER_BIT*len(BITS))


bit_last = 0
phase = 0.0

F_DEV = BIT_RATE*SPECTRAL_EFF*0.5

CARRIER = 10.0*F_DEV
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
print BITS
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

