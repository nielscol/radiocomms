import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
from lib._signal import make_signal
from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter, kaiser_composite_tx_rx_filter, rcos_composite_tx_rx_filter, gmsk_tx_filter
from lib.sync import get_precomputed_codes, make_sync_fir, code_string, frame_data, detect_sync
from lib.plot import *  # Assuming all plotting methods come from here


if __name__ == "__main__":

	_3D_EYE = False                 # Plot eye diagrams in 3D!
	EYE_VPP = 3.0                   # Set if you want to force a p-p eye value. Range of plot will be 1.25*EYE_VPP
	CMAP = "nipy_spectral"          # Color map of plot, best options are (1) "nipy_spectral" and (2) "inferno"
	SAMPLE_LINES = False            # Draw lines corresponding to where samples occur
	POOLS = 8                       # For using multiple processes to compute eyes. POOLS = # logical processors on your computer

	BT_TX = 0.3                     # GMSK BT
	BT_COMPOSITE = 1.0              # response of combined Rx+Tx filter
	FRAME_PAYLOAD = 640             # Number of data bits per fram
	SYNC_CODE_LEN = 24              # Length of sync field in bits, from precomputed values up to N=24
	SYNC_POS = "center"             # Where to place sync field in frame, {"center", "start"}
	SYNC_PULSE_SPAN = 16             # FIR span in symbols of pulse shape used to filter sync code for 
                                        # detection of frame synchronization

	RECOVERY_METHOD = "frame_sync"  # {"frame_sync", "constant_f", "edge"}, use "frame_sync" if data if framed, else "constant_f"

	TX_RATE = 79680
	OVERSAMPLING = 8                # Rate beyond bit rate which baseband will sample
	IQ_RATE = TX_RATE*OVERSAMPLING  # Baseband sampling rate
	x = np.genfromtxt('./eye_data/span_16.csv',delimiter=',')
	demodulated = make_signal(td=x, fs=TX_RATE*OVERSAMPLING, bitrate=TX_RATE)
	plot_td(demodulated)
	plt.show()

	sync_codes = get_precomputed_codes()
	sync_code = sync_codes[SYNC_CODE_LEN]
	kaiser_fir = kaiser_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                               fs=IQ_RATE, norm=True)
	plt.figure(1)
	plot_eye_density(demodulated, _3d=_3D_EYE, pools=POOLS, title="BT Tx=%.2f, Oversampling=%d,\nFIR Symbol span=%d"%(BT_TX, OVERSAMPLING, SYNC_PULSE_SPAN),
                     cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING,
                     sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS,
                     recovery=RECOVERY_METHOD, thresh=2.0, fir_span=SYNC_PULSE_SPAN)

	# plt.figure(2)
	# plt.subplot(1,3,1)
	# plot_tie(demodulated, alpha=0.8, sync_code=sync_code, pulse_fir=kaiser_fir,
	#          payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD, oversampling=OVERSAMPLING)

	# plt.subplot(1,3,2)
	# plot_jitter_histogram(demodulated, alpha=0.8, title="BT Tx=%.2f, Oversampling=%d,\nFIR Symbol span=%d"%(BT_TX, OVERSAMPLING, SYNC_PULSE_SPAN), oversampling=OVERSAMPLING,
	#                       sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
	# plt.subplot(1,3,3)
	# plot_jitter_spectrum(demodulated, alpha=0.8, title="BT Tx=%.2f, Oversampling=%d,\nFIR Symbol span=%d"%(BT_TX, OVERSAMPLING, SYNC_PULSE_SPAN), oversampling=OVERSAMPLING,
	#                      sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
plt.show()
