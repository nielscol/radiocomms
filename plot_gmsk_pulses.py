import matplotlib.pyplot as plt
import numpy as np
from lib.modulation import *

ts = 1.0
BT = [0.1, 0.3, 0.5, 1.0]

time = np.linspace(-5, 5, 1001)

for bt in BT:
    pulse = v_gmsk_pulse(time, bt, ts)
    plt.plot(time, pulse, label="BT=%.2f"%bt)
plt.legend()
plt.xlabel("Time [UI]")
plt.ylabel("Amplitude")
plt.title("GMSK Pulse shape for various BT products")
plt.show()
