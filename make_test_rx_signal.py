import numpy as np
import matplotlib.pyplot as plt
from lib._signal import wav_to_signal, save_signal_to_wav, make_signal
from lib.sync import frame_data_bursts, get_precomputed_codes
from lib.transforms import convert_to_bitstream, ulaw_compress, ulaw_expand
from lib.modulation import generate_gmsk_baseband, upconvert_baseband, demodulate_gmsk
from lib.plot import plot_td, plot_fd

DELAY = 0.15 # second
DURATION = 1.0 # second
FILE = "./sample_audio/all.wav"
SYNC_CODE_LEN = 24
FRAME_PAYLOAD = 640
AUDIO_RATE = 64000
BITRATE = AUDIO_RATE + 2400
TX_RATE = 79680
SYNC_POS = "center"
OVERSAMPLING = 8
BT_TX = 0.3
TX_FIR_SPAN = 4
BLOCKS_PER_S = 10
# Read source audio file and scale to fit range of sample size
audio = wav_to_signal(FILE)
samples = int(audio.fs*DURATION)
delay = int(audio.fs*DELAY)
audio.td = audio.td[delay:delay+samples]

ramp = np.arange(2**13)*2**3 - 2**15
ramp = ramp.astype(np.int16)
ramp = make_signal(td=ramp, fs=AUDIO_RATE, signed=True, bits=16)
#plt.plot(ramp.td)
#plt.show()

block_audio_sa = int(AUDIO_RATE/(BLOCKS_PER_S*OVERSAMPLING))
num_blocks = int(len(ramp.td)/block_audio_sa)


coded = ulaw_compress(ramp)
recovered = ulaw_expand(coded)
save_signal_to_wav(recovered, "ramp.wav")
bitstream = convert_to_bitstream(coded)


for n in range(num_blocks):
    print(n, num_blocks)
    plt.subplot(2,5,n+1)
    plt.plot(recovered.td[n*block_audio_sa:(n+1)*block_audio_sa])
    plt.title("Waveform %d"%(n+1))

plt.show()

sync_codes = get_precomputed_codes()
sync_code = sync_codes[SYNC_CODE_LEN]
message = frame_data_bursts(bitstream, sync_code, FRAME_PAYLOAD, TX_RATE, BITRATE, sync_pos=SYNC_POS)


gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX,
                                        pulse_span=TX_FIR_SPAN, binary_message=False)

demod = demodulate_gmsk(gmsk_i, gmsk_q, OVERSAMPLING)

#np.savetxt("demod_burst.csv", demod.td, delimiter=",")

gmsk = np.asarray([gmsk_i.td, gmsk_q.td])
np.savetxt("ramp_gmsk_iq_burst.csv", gmsk.T, delimiter=",")


#plt.figure(1)
#plot_td(demod)

#TX_LO = TX_RATE*3.0
#rf = upconvert_baseband(TX_LO, gmsk_i, gmsk_q)
#plt.figure(2)
#plot_fd(rf)

plt.show()
