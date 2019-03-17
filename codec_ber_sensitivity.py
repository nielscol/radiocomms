'''Simulates power spectrum of single tone audio with quantization 
and injected bit errors
'''
import matplotlib.pyplot as plt
import numpy as np
import math
from lib._signal import *
from lib.plot import *
from lib.analysis import *
from lib.transforms import *

BITS_PER_SAMPLE = 16
SAMPLING_RATE = 8000
BER = 0.0
SIMULATION_SAMPLES = 800000
TEST_TONE = 440 # Hz
NOISE_IN_LSB = 0 # standard deviation

LEVEL = 0.131899 # rms level of test audio

_signal_og_16b = generate_quantized_tone(tone_freq=TEST_TONE, fs=SAMPLING_RATE, samples=SIMULATION_SAMPLES, bits=BITS_PER_SAMPLE, noise_lsbs=NOISE_IN_LSB)


sfdr = {"og":[], "pcm":[], "ulaw":[]}
sndr = {"og":[], "pcm":[], "ulaw":[]}
enob = {"og":[], "pcm":[], "ulaw":[]}

LOG_BER = np.linspace(-2, -7, 11)

BERS = 10**LOG_BER

for BER in BERS:
    signal_og_16b = rescale_signal(_signal_og_16b, factor = LEVEL)

    signal_ulaw_comp_8b = ulaw_compress(signal_og_16b)
    signal_ulaw_comp_8b = corrupt(signal_ulaw_comp_8b, ber=BER)
    signal_ulaw_exp_16b = ulaw_expand(signal_ulaw_comp_8b)
    signal_ulaw_exp_16b.name = "corrupted_ulaw"

    signal_pcm_8b = truncate_with_rounding(signal_og_16b, n_bits=8)
    signal_corrupted_pcm = corrupt(signal_pcm_8b, ber=BER)
    signal_corrupted_pcm.name = "corrupted_pcm"
    signal_corrupted_pcm_16b = increase_bits(signal_corrupted_pcm, n_bits=16)

    sfdr["og"].append(measure_sfdr(remove_dc(signal_og_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5))
    sndr["og"].append(measure_sndr(remove_dc(signal_og_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5))
    enob["og"].append((sndr["og"][-1]-1.76)/6.02)
    sfdr["pcm"].append(measure_sfdr(remove_dc(signal_corrupted_pcm_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5))
    sndr["pcm"].append(measure_sndr(remove_dc(signal_corrupted_pcm_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5))
    enob["pcm"].append((sndr["pcm"][-1]-1.76)/6.02)
    sfdr["ulaw"].append(measure_sfdr(remove_dc(signal_ulaw_exp_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5))
    sndr["ulaw"].append(measure_sndr(remove_dc(signal_ulaw_exp_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5))
    enob["ulaw"].append((sndr["ulaw"][-1]-1.76)/6.02)

plt.figure(0)
plt.plot(LOG_BER, sfdr["og"], label="original")
plt.plot(LOG_BER, sfdr["pcm"], label="pcm corrupted")
plt.plot(LOG_BER, sfdr["ulaw"], label="ulaw corrupted")
plt.legend()
plt.ylabel("[dB]")
plt.xlabel("Log(BER)")
plt.title("SFDR")

plt.figure(1)
plt.plot(LOG_BER, sndr["og"], label="original")
plt.plot(LOG_BER, sndr["pcm"], label="pcm corrupted")
plt.plot(LOG_BER, sndr["ulaw"], label="ulaw corrupted")
plt.legend()
plt.ylabel("[dB]")
plt.xlabel("Log(BER)")
plt.title("SNDR")

plt.figure(2)
plt.plot(LOG_BER, enob["og"], label="original")
plt.plot(LOG_BER, enob["pcm"], label="pcm corrupted")
plt.plot(LOG_BER, enob["ulaw"], label="ulaw corrupted")
plt.legend()
plt.ylabel("[bits]")
plt.xlabel("Log(BER)")
plt.title("ENOB")
plt.show()

