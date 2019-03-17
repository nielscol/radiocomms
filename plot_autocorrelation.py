'''Methods for synchronization related things
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from lib.sync import *

a = get_precomputed_codes()
b = np.correlate(a[24],a[24], mode="full")

plt.plot(b)
plt.title("24 Length Code Autocorrelation")
plt.ylabel("Autocorrelation value")
plt.show()


