""" Figure out probability of single bit errors in audio sample, and
    probability for error in MSB of sample. 

    Cole Nielsen 2019
"""
import numpy as np
from scipy.stats import binom

SAMPLING_RATES = [4000,8000]
DEFAULT_FS = 8000
BITS_PER_SAMPLE = 8
bers = 10**((np.arange(17)-18.0)/2.0)

for SAMPLING_RATE in SAMPLING_RATES:
    print("\n* Sampling rate = %d Hz,\tbits per sample =%d"%(SAMPLING_RATE,BITS_PER_SAMPLE))
    print("\nBER prob.\tSample err pr\tMSB err/sec")
    sample_error_rate = {}
    msb_error_rate = {}
    for ber in bers:
        sample_error_rate[ber] = binom.pmf(n=BITS_PER_SAMPLE,k=1,p=ber)
        msb_error_rate[ber] = SAMPLING_RATE*ber
        print("%E\t%E\t%E"%(ber, sample_error_rate[ber], msb_error_rate[ber]))

    print("\n* Required BER for various MSB errors per second")
    print("MSB Error/s\tBER")
    _bers = 10**np.linspace(-9.0, -1, 1000)
    for r in [1e-2, 1e-1, 1e0, 1e1]:
        _ber = _bers[np.argmin((r-SAMPLING_RATE*_bers)**2)]
        print("%E\t%E"%(r,_ber))
