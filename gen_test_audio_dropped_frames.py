import matplotlib.pyplot as plt
import numpy as np
import math
from lib._signal import wav_to_signal, save_signal_to_wav
from lib.transforms import * # all signal transforms are from here
from lib.plot import plot_td

WAV_BITS = 16
FILE = "./speech.wav"

FRAME_SA = 160
DROP_RATE = 5 # per sec
SA_RATE = 8000

period = int(SA_RATE/DROP_RATE)

audio = wav_to_signal(FILE)

len_audio = len(audio.td)
n = 0
added = 0
while n<len_audio:
    for m in range(FRAME_SA):
        if n+m > len_audio:
            break
        else:
            audio.td[n+m] = 0
            # audio.td[n+m] = np.random.choice(np.arange(2**16)-2**15)
    n += int(round(np.random.exponential(period)))
    added += 1

print(added)

save_signal_to_wav(audio, file_name="./speech_20ms_drops_5_per_sec.wav")
