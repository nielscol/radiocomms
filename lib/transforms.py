""" Methods for transforming data in Signal objects
    Cole Nielsen 2019
"""

import numpy as np
import math
from scipy.signal import decimate, resample
from lib._signal import *
from lib.tools import *

##########################################################################################
# FREQ DOMAIN TRANSFORMS
##########################################################################################


def simulate_tf_on_signal(signal, tf, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Simulate effect of transfer function on signal with hermitian transfer function
    '''
    if not any(signal.fd):
        signal.fd = np.fft.fft(signal.td)
    v_tf = np.vectorize(tf, otypes=[np.complex])
    if verbose:
        print("\n* Applying transfer function to signal.")
    new_fd = np.zeros(signal.samples, dtype=np.complex)
    new_fd[:int(signal.samples/2)] = v_tf(np.arange(int(signal.samples/2), dtype=np.complex)*signal.fbin)*signal.fd[:int(signal.samples/2)]
    new_fd[int(signal.samples/2):] = v_tf((np.arange(int(signal.samples/2), dtype=np.complex)-signal.samples/2)*signal.fbin)*signal.fd[int(signal.samples/2):]
    return make_signal(fd=new_fd, fs = signal.fs, bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name, verbose=False, *args, **kwargs)



##########################################################################################
# TIME DOMAIN TRANSFORMS
##########################################################################################

def remove_dc(signal, autocompute_fd=False, verbose=True, *args, **kwargs):
    if verbose:
        print("\n* Removing DC signal component from signal %s"%str(signal.name))
    dc = np.mean(signal.td)
    new_td = signal.td - dc
    if autocompute_fd:
        new_fd = np.fft.fft(new_td)
    else:
        new_fd = np.zeros(len(new_td))
    return Signal(new_td, new_fd, signal.fs, signal.samples, signal.bits, signal.signed, signal.fbin, signal.name)

def convert_to_unsigned(signal, autocompute_fd=False, verbose=True, *args, **kwargs):
    if signal.signed:
        new_td = signal.td + 2**(signal.bits-1)
        return make_signal(td=new_td, fs = signal.fs, bits=signal.bits, signed=False, autocompute_fd=autocompute_fd, name=signal.name, verbose=False, *args, **kwargs)
    else:
        return signal

def convert_to_signed(signal, autocompute_fd=False, verbose=True, *args, **kwargs):
    if not signal.signed:
        new_td = signal.td - 2**(signal.bits-1)
        return make_signal(td=new_td, fs = signal.fs, bits=signal.bits, signed=True, autocompute_fd=autocompute_fd, name=signal.name, verbose=False, *args, **kwargs)
    else:
        return signal


def scale_to_fill_range(signal, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Finds range of signal (number of codes), then DC centers signal and resizes sample bit number
    to just fit the data
    '''
    if not signal.signed:
        raise Exception("Please convert signal to be signed with convert_to_signed")
    average = int(round(np.mean(signal.td)))
    sig_min = average - np.amin(signal.td)
    sig_max = np.amax(signal.td) - average
    sig_peak = abs(sig_min) if abs(sig_min) > abs(sig_max) else abs(sig_max)
    eff_sig_bits = math.log(2.0*sig_peak, 2)
    # fix audio to maximize utilization of range available with eff_bits
    # i.e. remove DC offset and rescale so audio peak hits maximum code for ceil(eff_bits)
    fill_range_gain = (2**(signal.bits-1)-1)/float(sig_peak)
    new_td = np.array((np.rint((signal.td-average)*fill_range_gain)), dtype=np.int32)
    if verbose:
        print("\n* Forced signal to fill range of %d bits"%signal.bits)
        print("\t* Source audio effective bits = %f"%eff_sig_bits)
    return make_signal(td=new_td, fs = signal.fs, bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name, verbose=False, *args, **kwargs)

def rescale_signal(signal, factor, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Rescales signal
    '''
    if not signal.signed:
        raise Exception("Please convert signal to be signed with convert_to_signed")
    if factor > 1.0:
        raise(Exception("Do not use to scale beyond 1.0, always use scale_to_fill_range, and then this method with factor<1.0"))
    new_td = np.array(np.rint(signal.td*factor), dtype=np.int32)
    if verbose:
        print("\n* Rescaled signal by factor %f"%factor)
    return make_signal(td=new_td, fs = signal.fs, bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name, verbose=False, *args, **kwargs)



##########################################################################################
# DOWNSAMPLING AND DECIMATION 
##########################################################################################

def filter_and_downsample(signal, n, order=None, ftype="iir", autocompute_fd=False, verbose=True, *args, **kwargs):
    if verbose:
        print("\n* %s filtering and downsampling %s by factor %d."%(ftype, signal.name, n))
    td = np.array(np.rint(decimate(x=signal.td, q=n, n=order, ftype=ftype)), dtype=np.int32)
    return make_signal(td=td, fs = signal.fs/float(n), bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name +"-filtdecim", verbose=verbose, *args, **kwargs)

def fft_downsample(signal, n, autocompute_fd=False, verbose=True, *args, **kwargs):
    if verbose:
        print("\n* FFT downsampling %s by factor %d."%(signal.name, n))
    td = np.array(np.rint(resample(signal.td, num = int(round(signal.samples/float(n))))) ,dtype=np.int32)
    return make_signal(td=td, fs = signal.fs/float(n), bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name +"-fftdecim", verbose=verbose, *args, **kwargs)

def no_filter_downsample(signal, n, autocompute_fd=False, verbose=True, *args, **kwargs):
    if verbose:
        print("\n* Downsampling %s by factor %d."%(signal.name, n))
        print("\rNo anti-aliasing filter applied")
    indices = np.arange(signal.samples)
    td = signal.td[indices[indices%n == 0]]
    return make_signal(td=td, fs = signal.fs/float(n), bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name + "-decimated", verbose=verbose, *args, **kwargs)


##########################################################################################
# FOR SIMULATED CORRUPTION OF DATA
##########################################################################################

def corrupt_sample(sample, ber, bits):
    '''Takes int type sample, corrupts individual bits of sample with
    probability ber
    '''
    corrupted = 0
    for bit in range(bits):
        corrupted += ((2**bit)&sample)^(np.random.binomial(n=1,p=ber)<<bit)
    return corrupted

v_corrupt_samples = np.vectorize(corrupt_sample, otypes=[np.int32])

def corrupt(signal, ber, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Takes signal, corrupts bits in samples with error rate ber
    '''
    og_signed = signal.signed
    og_type = signal.td.dtype
    signal = convert_to_unsigned(signal)
    new_td = v_corrupt_samples(signal.td, ber=ber, bits=signal.bits)
    signal = make_signal(td=new_td, fs = signal.fs, bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name +"-ber_%E"%ber, verbose=False, *args, **kwargs)
    if og_signed:
        return convert_to_signed(signal)
    elif og_type == np.uint8:
        signal.bits = 8
        signal.signed = False
        signal.td = np.array(signal.td, dtype=np.uint8)
        return signal
    else:
        return signal

def corrupt_sample_nth_bit(sample, n, ber, bits):
    '''Corrupts only the nth bit of sample with probability ber
    '''
    corrupted = 0
    for bit in range(bits):
        if bit == n:
            corrupted += ((2**bit)&sample)^(np.random.binomial(n=1,p=ber)<<bit)
        else:
            corrupted += (2**bit)&sample
    return corrupted


v_corrupt_samples_nth_bit = np.vectorize(corrupt_sample_nth_bit, otypes=[np.int32])

def corrupt_nth_bit_of_samples(signal, n, ber, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Takes signal, corrupts only nth bit in samples with error rate ber
    '''
    og_signed = signal.signed
    og_type = signal.td.dtype
    signal = convert_to_unsigned(signal)
    if not signal.bits:
        raise(Exception("Signal does not have number of bits per sample (.bits) specificed."))
    if n >= signal.bits:
        raise Exception("Can't change bit %d of %d bit sample. Bits start at 0."%(n, signal.bits))
    new_td = v_corrupt_samples_nth_bit(signal.td, n=n, ber=ber, bits=signal.bits)
    signal = make_signal(td=new_td, fs = signal.fs, bits=signal.bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name +"-_bit%d_ber_%E"%(n,ber), verbose=False, *args, **kwargs)
    if og_signed:
        return convert_to_signed(signal)
    elif og_type == np.uint8:
        signal.bits = 8
        signal.signed = False
        signal.td = np.array(signal.td, dtype=np.uint8)
        return signal
    else:
        return signal

##########################################################################################
# REDUCE/INCREASE BITS IN SAMPLES
##########################################################################################

def round_truncate_sample(sample, n_in, n_out):
    '''Takes int type sample of n_in bits, reduces it n_out bits with rounding
    '''
    delta_n = n_in-n_out
    sum_of_truncated = 0
    for n in range(delta_n):
        sum_of_truncated += sample&(2**n)
    if sum_of_truncated < 2**(delta_n-1):
        return sample >> delta_n
    elif (sample >> delta_n) + 1 == 2**(n_out): # if will cause overflow
        return sample >> delta_n
    else:
        return (sample >> delta_n) + 1

v_round_truncate_samples = np.vectorize(round_truncate_sample, otypes=[np.int32])

def truncate_with_rounding(signal, n_bits, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Takes signal, reduced number of bits per sample to n_bits
    '''
    og_signed = signal.signed
    signal = convert_to_unsigned(signal)
    if not signal.bits:
        raise(Exception("Signal does not have number of bits per sample (.bits) specificed."))
    if n_bits > signal.bits:
        raise(Exception("Can't truncate signal (%d bits) to smaller number of bits (%d bits)."%(signal.bits, n_bits)))
    new_td = v_round_truncate_samples(signal.td, n_in=signal.bits, n_out=n_bits)
    signal = make_signal(td=new_td, fs = signal.fs, bits=n_bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name +"-truncated_to_%d_bits"%(n_bits), verbose=False, *args, **kwargs)
    if og_signed:
        return convert_to_signed(signal)
    else:
        return signal

def increase_bits(signal, n_bits, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Takes signal, increases number of bits to be n_bits in total
    '''
    og_signed = signal.signed
    signal = convert_to_unsigned(signal)
    if not signal.bits:
        raise(Exception("Signal does not have number of bits per sample (.bits) specificed."))
    if n_bits < signal.bits:
        raise(Exception("Can't increase signal (%d bits) to smaller number of bits (%d bits)."%(signal.bits, n_bits)))
    new_td = (signal.td << (n_bits - signal.bits))
    signal = make_signal(td=new_td, fs = signal.fs, bits=n_bits, signed=signal.signed, autocompute_fd=autocompute_fd, name=signal.name, verbose=False, *args, **kwargs)
    if og_signed:
        return convert_to_signed(signal)
    else:
        return signal

##########################################################################################
# CODECS
##########################################################################################

# # # u-Law implementation from http://www.speech.cs.cmu.edu/comp.speech/Section2/Q2.7.html

'''
    This routine converts from linear to ulaw

    Craig Reese: IDA/Supercomputing Research Center
    Joe Campbell: Department of Defense
    29 September 1989

    References:
    1) CCITT Recommendation G.711  (very difficult to follow)
    2) "A New Digital Technique for Implementation of Any
        Continuous PCM Companding Law," Villeret, Michel,
        et al. 1973 IEEE Int. Conf. on Communications, Vol 1,
        1973, pg. 11.12-11.17
    3) MIL-STD-188-113,"Interoperability and Performance Standards
        for Analog-to_Digital Conversion Techniques,"
        17 February 1987

    Input: Signed 16 bit linear sample
    Output: 8 bit ulaw sample
'''

BIAS = 0x84   # define the add-in bias for 16 bit samples */
CLIP = 32635

exp_lut = [
    0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
]

def linear2ulaw(sample):
    # Get the sample into sign-magnitude. 
    sign = (sample >> 8) & 0x80        #set aside the sign
    if (sign != 0): sample = -sample    #get magnitude
    if (sample > CLIP): sample = CLIP   #clip the magnitude
    # Convert from 16 bit linear to ulaw.
    sample = sample + BIAS
    exponent = exp_lut[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa)
    return ulawbyte

'''
    This routine converts from ulaw to 16 bit linear.

    Craig Reese: IDA/Supercomputing Research Center
    29 September 1989

    References:
    1) CCITT Recommendation G.711  (very difficult to follow)
    2) MIL-STD-188-113,"Interoperability and Performance Standards
        for Analog-to_Digital Conversion Techniques,"
        17 February 1987

    Input: 8 bit ulaw sample
        Output: signed 16 bit linear sample
'''

def ulaw2linear(ulawbyte):
    exp_lut = [0,132,396,924,1980,4092,8316,16764]
    ulawbyte = ~ulawbyte
    sign = (ulawbyte & 0x80)
    exponent = (ulawbyte >> 4) & 0x07
    mantissa = ulawbyte & 0x0F
    sample = exp_lut[exponent] + (mantissa << (exponent + 3))
    if (sign != 0): sample = -sample
    return sample

# # # End of u-Law implementation from cmu.edu

v_linear2ulaw = np.vectorize(linear2ulaw, otypes=[np.uint8])
v_ulaw2linear = np.vectorize(ulaw2linear, otypes=[np.int16])

def ulaw_compress(signal, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Notes: implements u law compression for 16 bit signed input and 8 bit output
    '''
    if verbose:
        print("\n* Performing u-law compression on signal.")
    #Force signal to 16 bits and to be signedhash
    if signal.bits > 16:
        signal = truncate_with_rounding(signal, n_bits=16, autocompute_fd=autocompute_fd, verbose=False)
    elif signal.bits < 16:
        signal = increase_bits(signal, n_bits=16, autocompute_fd=autocompute_fd, verbose=False)
    signal = convert_to_signed(signal)
    td = np.array(signal.td, dtype=np.int16) # force to signed int16 type for ulaw conversion
    comp_td = v_linear2ulaw(td)
    return make_signal(td=comp_td, fs = signal.fs, bits=8, signed=False, autocompute_fd=autocompute_fd, name=signal.name+"-ulaw_comp", verbose=False, *args, **kwargs)

def ulaw_expand(signal, autocompute_fd=False, verbose=True, *args, **kwargs):
    '''Notes: implements u law compression for 16 bit signed input and 8 bit output
    '''
    if verbose:
        print("\n* Performing u-law expanding on signal")
    if signal.bits!=8 or signal.signed or signal.td.dtype != np.uint8:
        raise Exception("ulaw_expand only takes signals with uint8 type")
    expanded_td = v_ulaw2linear(signal.td)
    return make_signal(td=expanded_td, fs = signal.fs, bits=16, signed=True, autocompute_fd=autocompute_fd, name=signal.name+"-ulaw_expanded", verbose=False, *args, **kwargs)

