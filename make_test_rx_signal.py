import numpy as np
import matplotlib.pyplot as plt
from lib._signal import wav_to_signal
from lib.sync import frame_data_bursts, get_precomputed_codes
from lib.transforms import convert_to_bitstream, ulaw_compress, ulaw_expand
from lib.modulation import generate_gmsk_baseband, upconvert_baseband, demodulate_gmsk
from lib.plot import plot_td, plot_fd

DELAY = 0.15 # second
DURATION = 1.0 # second
FILE = "./sample_audio/all.wav"
SYNC_CODE_LEN = 24
FRAME_PAYLOAD = 640
BITRATE = 64000 + 2400
TX_RATE = 79680
SYNC_POS = "center"
OVERSAMPLING = 8
BT_TX = 0.3
TX_FIR_SPAN = 4
# Read source audio file and scale to fit range of sample size
audio = wav_to_signal(FILE)
samples = int(audio.fs*DURATION)
delay = int(audio.fs*DELAY)
audio.td = audio.td[delay:delay+samples]

coded = ulaw_compress(audio)
bitstream = convert_to_bitstream(coded)


sync_codes = get_precomputed_codes()
sync_code = sync_codes[SYNC_CODE_LEN]
message = frame_data_bursts(bitstream, sync_code, FRAME_PAYLOAD, TX_RATE, BITRATE, sync_pos=SYNC_POS)


gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX,
                                        pulse_span=TX_FIR_SPAN, binary_message=False)

demod = demodulate_gmsk(gmsk_i, gmsk_q, OVERSAMPLING)

np.savetxt("demod_burst.csv", demod.td, delimiter=",")

gmsk = np.asarray([gmsk_i.td, gmsk_q.td])
np.savetxt("gmsk_iq_burst.csv", gmsk.T, delimiter=",")


plt.figure(1)
plot_td(demod)

TX_LO = TX_RATE*3.0
rf = upconvert_baseband(TX_LO, gmsk_i, gmsk_q)
plt.figure(2)
plot_fd(rf)

plt.show()
