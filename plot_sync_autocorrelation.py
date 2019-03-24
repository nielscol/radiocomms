''' Plot autocorrelation of sync codes from lib.sync code generation module
    Cole Nielsen 2019
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from lib.sync import get_precomputed_codes

a = get_precomputed_codes()
b = np.correlate(a[24],a[24], mode="full")

plt.plot(b)
plt.title("24 Length Code Autocorrelation")
plt.ylabel("Autocorrelation value")
plt.show()


