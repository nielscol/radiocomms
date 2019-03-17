from lib._signal import *
from lib.transforms import *
from lib.analysis import *
from lib.plot import *
import matplotlib.pyplot as plt
import numpy as np
from math import pi

fs = 1e6
n_samples = 200000
IF = 100e3
MU = 0.5
D = 0.5

for mu in [1.0, 0.5 ]:
    time = np.arange(n_samples)/fs

    m = np.zeros(n_samples)

    for freq in range(1000, 4000, 1000):
        m += np.cos(2*pi*freq*time)

    if abs(np.amax(m)) > abs(np.amin(m)):
        mp = abs(np.amax(m))
    else:
        mp = abs(np.amin(m))
    td = D*(1.0 + mu*m/mp)
    td *=  np.cos(2*pi*IF*time)
    td +=  np.random.normal(0, 0.01, n_samples)
    td /= float(n_samples)

    message = make_signal(td=td, fs=fs)
    carrier_i = freq_to_index(message, 100e3)
    sideband_i = freq_to_index(message, 103e3)
    carrier = 20*np.log10(np.abs(message.fd[carrier_i]))
    sideband = 20*np.log10(np.abs(message.fd[sideband_i]))
    print "mu = %f, carrier - sideband = %f dB"%(mu, carrier-sideband)
    plot_fd(message, log=True, label="M = %.2f"%mu)
    ax = plt.gca()
    ax.annotate('(100 kHz, %.2f dB)' %(carrier), xy=(100e3,carrier), )
    ax.annotate('(103 kHz, %.2f dB)' %(sideband), xy=(103e3,sideband), )
plt.legend()
plt.grid()
plt.title("Theoretical IF Spectrum for Cosine tones")
plt.xlim([90e3,110e3])
plt.ylim([-120, 0])
plt.show()
