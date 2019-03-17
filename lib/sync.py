""" Methods for synchronization related things
    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from lib.tools import timer
import json

def get_precomputed_codes():
    '''Precomputed codes for N=1 to N=24
    '''
    print("\n* Reading lib/sync_codes.json to obtain synchronization codes dictionary")
    f = open("./sync_codes.json","r")
    data = f.read()
    f.close()
    return {int(k):v for (k,v) in json.loads(data).items()}

def code_string(code):
    s = ""
    for x in code:
        if x == 0:
            s += "0"
        else:
            s += "+" if x > 0.0 else "-"
    return s


def gen_array_from_num(num, bits):
    arr = np.zeros(bits)
    for n in range(bits):
        arr[n] = 1 if (2**n & num)>>n else -1
    return arr

def lagged_corr_power(autocorrelation,bits):
    n = bits
    return np.sum(autocorrelation[:n-1]**2) + np.sum(autocorrelation[n:]**2)

v_lagged_corr_power = np.vectorize(lagged_corr_power, otypes=[np.int64])


@timer
def find_code(bits, verbose=True, *args, **kwargs):
    if verbose:
        print("Finding sequency of %d bits with minimum sideband autocorrelation power"%bits)
    best_code = None
    best_cost = np.inf
    for n in range(2**bits):
        #code = np.random.choice([-1,1], size=24)
        code = gen_array_from_num(n, bits=bits)
        autocorrelation = np.correlate(code, code, mode="full")
        cost = lagged_corr_power(autocorrelation, bits=bits)
        if cost < best_cost and any(code): # abs(sum(code))<=1:
            best_code = code
            best_cost = cost
            if verbose:
                print("\t├─ NEW BEST CODE, n=%d"%n)
                print("\t│\t├─ Code = %s"%code_string(code))
                #print(ac_code)
                #print(np.argmax(ac_code))
                print("\t│\t└─ Correlation sideband power = %.1f"%lagged_corr_power(autocorrelation, bits=bits))
    if verbose:
        print("\t│")
        print("\t└─ DONE. Optimal %d bits code = %s"%(bits, code_string(best_code)))
    #plt.plot(np.correlate(best_code, best_code, mode="full"))
    #plt.show()
    return best_code
