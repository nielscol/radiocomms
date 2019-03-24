''' Simulates correlation of data with synchronization codes
    Cole Nielsen 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf
import math
from lib._signal import wav_to_signal
from lib.sync import get_precomputed_codes
import json

FILE = "./speech.wav"
SYNC_LENGTH=18 # bits
DATA_FRAME_LENGTH=640 # bits
SIMULATION_FRAMES = 200
SYNC_ERROR_TOLERANCE = 0 # bits
BITS_PER_SAMPLE = 8
SAMPLE_RATE = 8000
BIT_RATE = BITS_PER_SAMPLE*SAMPLE_RATE

def signal_to_bitstream(signal):
    N = signal.bits
    total_bits = len(signal.td) * N
    arr = np.zeros(total_bits)
    for i, sample in enumerate(signal.td):
        for j in range(N):
            arr[i*N + j] = 1 if (2**j & sample)>>j else -1
    return arr


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

#data = np.random.choice([-1,1], size=DATA_FRAME_LENGTH*SIMULATION_FRAMES)

sync_codes = get_precomputed_codes()
for FILE in audio_files:
    print("\n* FILE NAME : %s"%FILE)
    audio = wav_to_signal(FILE)
    data = signal_to_bitstream(audio)

    bits_faulty_sync = {}
    for SYNC_LENGTH, code in sync_codes.items():
        correlated_data = np.correlate(data, code, mode="full")
        corr_val_count = {k:0 for k in np.linspace(-SYNC_LENGTH,SYNC_LENGTH,SYNC_LENGTH*2+1)}
        for val in correlated_data:
            corr_val_count[val] += 1
        print("\n* Correlation value counts")
        faulty_sync_prob = 0
        for corr_val, count in corr_val_count.items():
            print("\t├─  Val = %.0f --> count = %d, p = %E"%(corr_val, count, count/float(len(correlated_data))))
            if abs(corr_val) >= SYNC_LENGTH-SYNC_ERROR_TOLERANCE:
                faulty_sync_prob += count/float(len(correlated_data))
        print("\t│")
        print("\t└─ Probability abs(correlation value) >= %d = %E"%(SYNC_LENGTH-SYNC_ERROR_TOLERANCE, faulty_sync_prob))

        bits_faulty_sync[SYNC_LENGTH] =faulty_sync_prob

    print("\n* SYNC BITS SWEEP")
    for bits, prob_faulty in bits_faulty_sync.items():
        print("\t├─ %d bits --> p(faulty correlation) = %E"%(bits, prob_faulty))
        print("\t│\t└─ p(faulty sync per second) = %E"%(prob_faulty*BIT_RATE))
        #print("\t└─ END")

    sync_length = []
    p_faulty_correlation = []
    faulty_sync_per_second = []
    for n_bits, prob_faulty in bits_faulty_sync.items():
        sync_length.append(n_bits)
        p_faulty_correlation.append(prob_faulty)
        faulty_sync_per_second.append(prob_faulty*BIT_RATE)

    plt.figure(0)
    plt.grid(True)
    plt.plot(sync_length, p_faulty_correlation, label=FILE)
    ax = plt.gca()
    ax.set_yscale("log")
    plt.xlabel("Synchronization code length [bits]")
    plt.ylabel("Probability")
    plt.title("Probability of faulty sync correlation on random data")
    plt.legend()
    plt.figure(1)
    plt.grid(True)
    ax = plt.gca()
    ax.set_yscale("log")
    plt.plot(sync_length, faulty_sync_per_second, label=FILE)
    plt.xlabel("Synchronization code length [bits]")
    plt.ylabel("Faulty sync/s")
    plt.title("Faulty sync rate on random data, %d bits/s"%(BIT_RATE))
    plt.legend()
plt.show()

mean = np.mean(correlated_data)
stdev = np.std(correlated_data)

# probability of single correlation on random data being within acceptance range
# for synchronization
prob_over_corr_range = 1 - erf((float(SYNC_LENGTH)-mean)/(math.sqrt(2)*stdev))
prob_faulty_correlation = 1 - prob_over_corr_range - erf((float(SYNC_LENGTH-SYNC_ERROR_TOLERANCE)-mean)/(math.sqrt(2)*stdev))
# probability that synchronization anywhere in a frame of random data
prob_faulty_corr_in_fram = DATA_FRAME_LENGTH*prob_faulty_correlation
print("\n* Probability of faulty sync in data fitted to normal: %E"%prob_faulty_corr_in_fram)
print("\t└─ Random data correlation stdev = %.3f"%stdev)
x = np.linspace(-SYNC_LENGTH, SYNC_LENGTH, 100)
hist_data = plt.hist(correlated_data, bins=SYNC_LENGTH, density=True)
bellcurve = norm.pdf(x-1.0, mean, stdev)
plt.plot(x, bellcurve)
plt.show()

