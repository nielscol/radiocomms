""" Simulates power spectrum of single tone audio with quantization
    and injected bit errors with the average signal level found from
    the "./speech.wav" file, to determine the PSNR and ENOB for that
    audio file with various codecs.

    Cole Nielsen 2019
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from lib._signal import generate_quantized_tone
from lib.plot import plot_td, plot_fd
from lib.analysis import measure_sndr, measure_sfdr
from lib.transforms import * # everything else is from here

BITS_PER_SAMPLE = 16
SAMPLING_RATE = 8000
BER = 1e-4
SIMULATION_SAMPLES = 80000
TEST_TONE = 100 # Hz
NOISE_IN_LSB = 0.0 # standard deviation

signal_og_16b = generate_quantized_tone(tone_freq=TEST_TONE, fs=SAMPLING_RATE, samples=SIMULATION_SAMPLES, bits=BITS_PER_SAMPLE, noise_lsbs=NOISE_IN_LSB)
signal_og_16b = rescale_signal(signal_og_16b, factor = 0.131899)

signal_ulaw_comp_8b = ulaw_compress(signal_og_16b)
signal_ulaw_comp_8b = corrupt(signal_ulaw_comp_8b, ber=BER)
signal_ulaw_exp_16b = ulaw_expand(signal_ulaw_comp_8b)
signal_ulaw_exp_16b.name = "corrupted_ulaw"

signal_pcm_8b = truncate_with_rounding(signal_og_16b, n_bits=8)
signal_corrupted_pcm = corrupt(signal_pcm_8b, ber=BER)
signal_corrupted_pcm.name = "corrupted_pcm"
signal_corrupted_pcm_16b = increase_bits(signal_corrupted_pcm, n_bits=16)

measure_sfdr(remove_dc(signal_og_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5)
measure_sndr(remove_dc(signal_og_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5)
measure_sfdr(remove_dc(signal_corrupted_pcm_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5)
measure_sndr(remove_dc(signal_corrupted_pcm_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5)
measure_sfdr(remove_dc(signal_ulaw_exp_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5)
measure_sndr(remove_dc(signal_ulaw_exp_16b), tone_freq=TEST_TONE, tone_bw=signal_og_16b.fbin*5)

plot_td(signal_og_16b, log=True, label="original")
plot_td(signal_corrupted_pcm_16b, log=True, label="pcm corrupted")
plot_td(signal_ulaw_exp_16b, log=True, label="ulaw corrupted")
plt.show()

plot_fd(signal_og_16b, log=True, label="original")
plot_fd(signal_corrupted_pcm_16b, log=True, label="pcm corrupted")
plot_fd(signal_ulaw_exp_16b, log=True, label="ulaw corrupted")
plt.legend()
plt.ylabel("[dB]")
plt.xlabel("Freq [Hz]")
plt.show()

