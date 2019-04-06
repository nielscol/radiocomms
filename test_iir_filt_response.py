from lib.transforms import butter_lpf
from lib._signal import make_signal
import matplotlib.pyplot as plt
import numpy as np

a = np.zeros(40)
a[0] = 1.0

b = make_signal(td=a, fs=83000.0*8, bitrate=83000)
c = butter_lpf(b, cutoff=83000.0, order=5)
plt.plot(c.td)
plt.grid()
plt.xlabel("Samples")
plt.title("5th order FIR delay with fc=BW")
plt.show()

