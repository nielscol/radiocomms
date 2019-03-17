import matplotlib.pyplot as plt
import numpy as np
import math
from _signal import *
from scipy.io import wavfile

BITS_PER_SAMPLE = 8
SAMPLING_RATE = 8000
BER = 1e-5
SIMULATION_SAMPLES = 80000
TEST_TONE = 1e2 # Hz
WAV_BITS = 16
FILE = "./speech.wav"

def corrupt(sample, ber=BER, bits=BITS_PER_SAMPLE):
    corrupted = 0
    for bit in range(BITS_PER_SAMPLE):
        corrupted += ((2**bit)&sample)^(np.random.binomial(n=1,p=ber)<<bit)
    return corrupted

v_corrupt = np.vectorize(corrupt, otypes=[np.int16])

fs, audio = wavfile.read(FILE)

average = int(round(np.mean(audio)))

audio_min = average - np.amin(audio)
audio_max = np.amax(audio) - average

audio_peak = abs(audio_min) if abs(audio_min) > abs(audio_max) else abs(audio_max)

audio_bits = math.log(2.0*audio_peak, 2)
eff_bits = int(math.ceil(audio_bits))
print("Audio effective bits = %f,\tTotal = %d"%(audio_bits, eff_bits))


reduced_audio = (audio-average+2**(eff_bits-1))>>(eff_bits-BITS_PER_SAMPLE)
reduced_audio = np.array(reduced_audio, dtype=np.int16)
reduced_audio = (reduced_audio << (WAV_BITS - BITS_PER_SAMPLE-1)) - 2**(WAV_BITS-2)

wavfile.write("./speech_%dbit.wav"%BITS_PER_SAMPLE, rate=fs, data=reduced_audio)

reduced_audio = (audio-average+2**(eff_bits-1))>>(eff_bits-BITS_PER_SAMPLE)
reduced_audio = np.array(reduced_audio, dtype=np.int16)
reduced_audio = v_corrupt(reduced_audio)
reduced_audio = (reduced_audio << (WAV_BITS - BITS_PER_SAMPLE-1)) - 2**(WAV_BITS-2)

wavfile.write("./speech_%dbit_ber_%E.wav"%(BITS_PER_SAMPLE,BER), rate=fs, data=reduced_audio)


plt.plot(audio)
plt.plot(reduced_audio)
plt.show()


