""" Takes audio file and calcuates relative RMS signal level of the audio versus full range
    for the audio sample size
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from lib._signal import wav_to_signal
from lib.transforms import scale_to_fill_range

FILE = "./speech.wav"

# Read source audio file and scale to fit range of sample size
audio = wav_to_signal(FILE)
audio = scale_to_fill_range(audio)

audio_rms = np.std(audio.td-np.mean(audio.td))
normed_rms = audio_rms/(2**(audio.bits-1))

eff_rms_bits = np.log2(2*audio_rms)/audio.bits

print("\nAudio normalized rms level = %f"%normed_rms)
print("Effective RMS bits normed = %f"%eff_rms_bits)
