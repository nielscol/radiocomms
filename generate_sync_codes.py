""" Given synchronization word length, brute force finds code with lowest correlation sideband power
    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf
import math
from lib.sync import find_code, code_string
import json

MAX_N_BITS = 24

code_dict = {}
for n in range(1,1+MAX_N_BITS):
    code_dict[n] = list(find_code(bits=n))

text = json.dumps(code_dict)
f = open("sync_codes.json", "w")
f.write(text)
f.close()

#code_dict = get_precomputed_codes()
print("\nN bits\t Code")
for n, code in code_dict.items():
    print("%d\t%s"%(n, code_string(code)))

