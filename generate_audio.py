import matplotlib.pyplot as plt
import numpy as np
import math
from lib._signal import *
from lib.transforms import *
from lib.plot import *

BITS_PER_SAMPLE = [8, 16]
SAMPLING_RATE = 8000
BERS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
WAV_BITS = 16
FILE = "./speech.wav"

# audio generation loops to run
GENERATE_SAMPLING_SWEEP = False
GENERATE_BER_SWEEP = False
GENERATE_ERRORS_AT_NTH_BIT = False
GENERATE_MSB_ERRORS_AT_VARIOUS_RATES = False
GENERATE_DYNAMIC_RANGE_TEST_AUDIO = True


# Read source audio file and scale to fit range of sample size
audio = wav_to_signal(FILE)
audio = scale_to_fill_range(audio)

#############################################################################
# RUN audio generation loops
##############################################################################

# GENERATE audio with various sampling frequencies and bits per sample
if GENERATE_SAMPLING_SWEEP:
    print("Simulating audio at various samling rates and sample sizes")
    for freq in [2000, 4000, 8000]:
        if freq == audio.fs:
            audio_filtered = audio
        else:
            audio_filtered = filter_and_downsample(audio, n = audio.fs/freq)
        for n_bits in range(4,13,1):
            _audio = truncate_with_rounding(audio_filtered, n_bits=n_bits) # reduce to n_bits per sample
            _audio = increase_bits(_audio, n_bits=WAV_BITS) # increase back to 16 bits for saving as .wav audio
            save_signal_to_wav(_audio, file_name="./speech_%dHz_%dbits.wav"%(freq,n_bits))

# Generate audio at various BER with the BITS_PER_SAMPLE specified at file
# header
if GENERATE_BER_SWEEP:
    print("Simulating audio with various BER levels")
    for n_bits in BITS_PER_SAMPLE:
        _reduced_audio = truncate_with_rounding(audio, n_bits=n_bits)
        for ber in BERS:
            _audio = corrupt(_reduced_audio, ber=ber)
            _audio = increase_bits(_audio, n_bits=WAV_BITS)
            save_signal_to_wav(_audio, file_name="./speech_%dbits_ber_%E.wav"%(n_bits,ber))

# Generate audio with errors only at the nth bit in the sample, at RATE/second
RATE = 10 # per second
probability = RATE/float(audio.fs)
if GENERATE_ERRORS_AT_NTH_BIT:
    print("Simulating audio with bit errors only at nth bit in the sample")
    for bit in range(audio.bits):
        _audio = corrupt_nth_bit_of_samples(audio, n=bit, ber=probability)
        save_signal_to_wav(_audio, file_name="./speech_bit%d_errors_%s_per_sec.wav"%(bit,RATE))

# Generate audio with errors only at the MSB bit in the sample, at RATE/second
RATES = [0.1, 0.3, 1, 3, 10] # per second
if GENERATE_MSB_ERRORS_AT_VARIOUS_RATES:
    print("Simulating audio with bit errors only at nth bit in the sample")
    for rate in RATES:
        probability = rate/float(audio.fs)
        _audio = corrupt_nth_bit_of_samples(audio, n=audio.bits-1, ber=probability)
        save_signal_to_wav(_audio, file_name="./speech_msb_errors_%.2f_per_sec.wav"%(rate))


# Generate audio at various audio levels
LEVELS = [0.1, 0.3, 0.5, 1.0] # relative to full scale
if GENERATE_DYNAMIC_RANGE_TEST_AUDIO:
    print("Generating audio with various levels relative to full range to test dynamic range")
    for level in LEVELS:
        _audio = rescale_signal(audio, factor=level)
        _audio = truncate_with_rounding(_audio, n_bits=8) # reduce to n_bits per sample
        _audio = increase_bits(_audio, n_bits=WAV_BITS) # increase back to 16 bits for saving as .wav audio
        save_signal_to_wav(_audio, file_name="./speech_8bits_%.2f_full_range.wav"%(level))
