import numpy as np
import matplotlib.pyplot as plt
import math
from lib._signal import *
from lib.plot import *
from lib.transforms import *



def butterworth_tf(fc):
    def filter(f):
        return 1.0/(1.0+1.0j*(f/fc))
    return filter

SAMPLES = 80000
SAMPLING_RATE = 8000

fd_dirac_delta = np.ones(SAMPLES, dtype=np.complex)
signal = make_signal(fd=fd_dirac_delta, fs=SAMPLING_RATE)

tf = butterworth_tf(fc=400.0)
filt_signal = simulate_tf_on_signal(signal, tf=tf)
plot_fd(filt_signal)
plt.show()

FREQS = np.linspace(20, 4000, 191)
max_diff_by_freq = []
for freq in FREQS:
    print freq
    tf = butterworth_tf(fc=freq)
    filt_signal = simulate_tf_on_signal(signal, tf=tf)
    max_diff_by_freq.append(np.amax(np.abs(filt_signal.td)))


plt.xlabel("Frequency [Hz]")
plt.ylabel("Max diff(Signal) [LSB]")
plt.plot(FREQS, max_diff_by_freq)
plt.show()


