""" Random helping functions / decorators """
import functools
import time
import numpy as np
from math import pi, sqrt, log, sin
from scipy.special import erfc

SQRT2 = sqrt(2.0)
SQRTLN2 = sqrt(log(2.0))

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print("Finished %r in %.2f seconds"%(func.__name__,run_time))
        return value
    return wrapper_timer

##############################################################################
# Useful mathmatical methods
#############################################################################

def sinx_x(x):
    return 1.0 if x==0 else sin(pi*x)/(pi*x)
v_sinx_x = np.vectorize(sinx_x, otypes=[float])

@timer
def sinx_x_interp(x, factor, span):
    # make sinx_x pulse shape
    fir_len = int(factor*span)
    t = (np.arange(fir_len)-fir_len/2)/float(factor)
    pulse_shape = v_sinx_x(t)
    pulse_error = np.sum(pulse_shape)-factor
    pulse_shape -= pulse_error/float(fir_len)
    # upsample signal and apply pulse shape through convolution
    upsampled = np.zeros(len(x)*factor)
    upsampled[np.arange(len(x))*factor] = x
    interpolated = np.convolve(upsampled, pulse_shape, mode="full")
    return interpolated

def q(x):
    return 0.5*erfc(x/SQRT2)

def raised_cos(t, tbit, rolloff):
    tbit=float(tbit)
    if rolloff != 0.0 and abs(tbit/(2.0*rolloff)) == t:
        return (pi/(4.0*tbit))*sinx_x(1/(2.0*rolloff))
    elif (2*rolloff*t/tbit)**2 == 1.0:
        if raised_cos(t*(1+1e-6), tbit,rolloff) < 0.25:
            return 0.0
        else:
            return 0.5
    else:
        return (1.0/tbit)*sinx_x(t/tbit)*np.cos(pi*rolloff*t/tbit)/(1.0-(2*rolloff*t/tbit)**2)
v_raised_cos = np.vectorize(raised_cos, otypes=[float])

