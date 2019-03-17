import matplotlib.pyplot as plt
import numpy as np
import math
from _signal import *
from scipy.io import wavfile
from scipy.signal import decimate, resample

BITS_PER_SAMPLE = 5
SAMPLING_RATE = 8000
BER = 1e-5
SIMULATION_SAMPLES = 80000
TEST_TONE = 1e2 # Hz
WAV_BITS = 16
FILE = "./speech.wav"

def round_truncate_sample(sample, n_in, n_out):
    '''Takes int type sample of n_in bits, reduces it n_out bits with rounding
    '''
    delta_n = n_in-n_out
    sum_of_truncated = 0
    for n in range(delta_n):
        sum_of_truncated += sample&(2**n)
    if sum_of_truncated < 2**(delta_n-1):
        return sample >> delta_n
    else:
        return (sample >> delta_n) + 1

v_round_truncate_sample = np.vectorize(round_truncate_sample, otypes=[np.int16])

def corrupt(sample, ber=BER, bits=BITS_PER_SAMPLE):
    '''Takes int type sample, corrupts individual bits of sample with
    probability ber
    '''
    corrupted = 0
    for bit in range(BITS_PER_SAMPLE):
        corrupted += ((2**bit)&sample)^(np.random.binomial(n=1,p=ber)<<bit)
    return corrupted

v_corrupt = np.vectorize(corrupt, otypes=[np.int16])

def corrupt_nth_bit(sample, n, ber=BER):
    '''Corrupts only the nth bit of sample with probability ber
    '''
    corrupted = 0
    for bit in range(BITS_PER_SAMPLE):
        if bit == n:
            corrupted += ((2**bit)&sample)^(np.random.binomial(n=1,p=ber)<<bit)
        else:
            corrupted += (2**bit)&sample
    return corrupted


v_corrupt_nth_bit = np.vectorize(corrupt_nth_bit, otypes=[np.int16])

# read source audio file
fs, audio = wavfile.read(FILE)
print("* Source audio sample rate : %d Hz"%fs)

# detect input audio parameters
average = int(round(np.mean(audio)))
audio_min = average - np.amin(audio)
audio_max = np.amax(audio) - average
audio_peak = abs(audio_min) if abs(audio_min) > abs(audio_max) else abs(audio_max)
audio_bits = math.log(2.0*audio_peak, 2)
eff_bits = int(math.ceil(audio_bits))
# fix audio to maximize utilization of range available with eff_bits
# i.e. remove DC offset and rescale so audio peak hits maximum code for ceil(eff_bits)
fill_range_gain = (2**(eff_bits-1)-1)/float(audio_peak)
audio = np.array((np.rint((audio-average)*fill_range_gain) + 2**(eff_bits-1)), dtype=np.int16)

print("* Source audio effective bits = %f,\tceil(eff_bits) = %d"%(audio_bits, eff_bits))


#RATE = 10
#N = 0
#ber = RATE/float(SAMPLING_RATE)

#for n in range(BITS_PER_SAMPLE):
#    print("* Generating autio with %d corruptions/s on bit %d of sample"%(RATE, n))
#    reduced_audio = (audio-average+2**(eff_bits-1))>>(eff_bits-BITS_PER_SAMPLE)
#    reduced_audio = np.array(reduced_audio, dtype=np.int16)
#    reduced_audio = v_corrupt_nth_bit(reduced_audio, n=n, ber=ber)
#    reduced_audio = (reduced_audio << (WAV_BITS - BITS_PER_SAMPLE-1)) - 2**(WAV_BITS-2)
#
#    wavfile.write("./speech_%dbit_bit%d_%d_per_sec.wav"%(BITS_PER_SAMPLE,n,RATE), rate=fs, data=reduced_audio)

#for rate_exp in range(-3,2,1):
#    N = 7
#    rate = (10.0**rate_exp)
#    ber = rate/float(SAMPLING_RATE)
#    print("* Generating autio with %.3f corruptions/s on bit %d of sample"%(rate, N))
#    reduced_audio = (audio-average+2**(eff_bits-1))>>(eff_bits-BITS_PER_SAMPLE)
#    reduced_audio = np.array(reduced_audio, dtype=np.int16)
#    reduced_audio = v_corrupt_nth_bit(reduced_audio, n=N, ber=ber)
#    reduced_audio = (reduced_audio << (WAV_BITS - BITS_PER_SAMPLE-1)) - 2**(WAV_BITS-2)
#
#    wavfile.write("./speech_%dbit_bit%d_%.3f_per_sec.wav"%(BITS_PER_SAMPLE,N,rate), rate=fs, data=reduced_audio)

print("Generating audio with truncated sample sizes and decimated sampling rate")
for decim_factor in [1,2]:
    if decim_factor != 1:
        _audio = decimate(x=audio, q=decim_factor)
        _audio = np.array(np.rint(_audio), dtype=np.int16)
    else:
        _audio = audio
    for n_bits in range(4,11,1):
        reduced_audio = v_round_truncate_sample(_audio, n_in=eff_bits, n_out=n_bits)
        reduced_audio = np.array(reduced_audio, dtype=np.int16)
        #reduced_audio = v_corrupt(reduced_audio)
        reduced_audio = (reduced_audio << (WAV_BITS - n_bits - 1)) - 2**(WAV_BITS-2)
        print("* sampling rate = %d Hz,\tnumber of bits = %d"%(fs/decim_factor,n_bits))
        wavfile.write("./speech_%d_Hz_%d_bits.wav"%(fs/decim_factor,n_bits), rate=fs/decim_factor, data=reduced_audio)


time = np.arange(len(audio))/float(fs)

#plt.plot(time, audio)
plt.plot(time, reduced_audio)
plt.show()


