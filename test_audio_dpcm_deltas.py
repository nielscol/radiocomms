""" Under differential-PCM, determine the maximum linear code deltas for
    for several test audio files.

    Cole Nielsen 2019
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from lib._signal import *
from lib.plot import *
from lib.analysis import *
from lib.transforms import *

audio_files = [
    "sample_audio/all.wav",
    #"sample_audio/cross.wav",
    "sample_audio/forig.wav",
    "sample_audio/hts1a.wav",
    "sample_audio/hts2a.wav",
    "sample_audio/mmt1.wav",
    "sample_audio/morig.wav",
    "sample_audio/speech.wav",
    "sample_audio/ve9qrp.wav",
    "sample_audio/vk5qi.wav",
]

BITS_PER_SAMPLE = 16
SAMPLING_RATE = 8000
BER = 0.0
SIMULATION_SAMPLES = 8000
TEST_TONE = 440 # Hz
NOISE_IN_LSB = 0 # standard deviation

audio = {FILE:wav_to_signal(FILE) for FILE in audio_files}
print("")
for name, a in audio.items():
    print("%s\t%d"%(name, np.max(np.diff(a.td))))
