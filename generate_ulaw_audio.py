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
GENERATE_BER_SWEEP = False
GENERATE_ERRORS_AT_NTH_BIT = False
GENERATE_MSB_ERRORS_AT_VARIOUS_RATES = False
GENERATE_DYNAMIC_RANGE_TEST_AUDIO = True


# Read source audio file and scale to fit range of sample size
audio = wav_to_signal(FILE)
audio = scale_to_fill_range(audio)

#############################################################################
# Make mu-law reference audio
##############################################################################

ulaw_audio = ulaw_compress(audio)
ulaw_audio = ulaw_expand(ulaw_audio)
save_signal_to_wav(ulaw_audio, file_name="./speech_ulaw.wav")
plt.show()

# Generate audio at various BER 
# header
if GENERATE_BER_SWEEP:
    print("Simulating audio with various BER levels")
    for ber in BERS:
        ulaw_audio = ulaw_compress(audio)
        ulaw_audio = corrupt(ulaw_audio, ber=ber)
        ulaw_audio = ulaw_expand(ulaw_audio)
        save_signal_to_wav(ulaw_audio, file_name="./speech_ulaw_ber_%E.wav"%(ber))

# Generate audio with errors only at the nth bit in the sample, at RATE/second
RATE = 10 # per second
probability = RATE/float(audio.fs)
if GENERATE_ERRORS_AT_NTH_BIT:
    print("Simulating audio with bit errors only at nth bit in the sample")
    for bit in range(8):
        ulaw_audio = ulaw_compress(audio)
        ulaw_audio = corrupt_nth_bit_of_samples(ulaw_audio, n=bit, ber=probability)
        ulaw_audio = ulaw_expand(ulaw_audio)
        save_signal_to_wav(ulaw_audio, file_name="./speech_ulaw_bit%d_errors_%s_per_sec.wav"%(bit,RATE))

# Generate audio with errors only at the MSB bit in the sample, at RATE/second
RATES = [0.1, 0.3, 1, 3, 10] # per second
if GENERATE_MSB_ERRORS_AT_VARIOUS_RATES:
    print("Simulating audio with bit errors only at nth bit in the sample")
    for rate in RATES:
        probability = rate/float(audio.fs)
        ulaw_audio = ulaw_compress(audio)
        ulaw_audio = corrupt_nth_bit_of_samples(ulaw_audio, n=7, ber=probability)
        ulaw_audio = ulaw_expand(ulaw_audio)
        save_signal_to_wav(ulaw_audio, file_name="./speech_ulaw_msb_errors_%.2f_per_sec.wav"%(rate))

# Generate audio at various audio levels
LEVELS = [0.1, 0.3, 0.5, 1.0] # relative to full scale
if GENERATE_DYNAMIC_RANGE_TEST_AUDIO:
    print("Generating audio with various levels relative to full range to test dynamic range")
    for level in LEVELS:
        _audio = rescale_signal(audio, factor=level)
        ulaw_audio = ulaw_compress(_audio)
        ulaw_audio = ulaw_expand(ulaw_audio)
        save_signal_to_wav(ulaw_audio, file_name="./speech_ulaw_%.2f_full_range.wav"%(level))

