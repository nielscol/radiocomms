import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import *
from lib.modulation import *
from lib.plot import *
from lib.transforms import *
from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter

if __name__ == "__main__":
    #gaussian_to_raised_cosine(0.3, 16, 4)

    N_BITS = 5000
    BIT_RATE = 64000 + 2400
    OVERSAMPLING = 16

    BT_TX = 0.3
    BT_COMPOSITE = 1.0
    PULSE_SPAN = 8

    SPECTRAL_EFF = 1.0 # bits/s/Hz
    BW = BIT_RATE/(SPECTRAL_EFF*2.0) # theoretical signal bandwidth
    CARRIER = 10.0*BW

    message1 = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)
    message2 = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)
    message = sum_signals(rescale_signal(sign(message1), 2.0/3.0), rescale_signal(sign(message2), 1.0/3.0))
    #plot_td(message)
    #plt.show()


    gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX, pulse_span=PULSE_SPAN, binary_message=False)
    gmsk_rf = upconvert_baseband(CARRIER, gmsk_i, gmsk_q)

    plt.subplot(2,2,1)
    plot_constellation_density(gmsk_i, gmsk_q, title="- Tx")
    plt.subplot(2,2,2)
    plot_iq_phase_mag(gmsk_i, gmsk_q, title="- Tx")
    plt.subplot(2,2,3)
    plot_phase_histogram(gmsk_i, gmsk_q, title="- Tx")

    gmsk_rf = add_noise(gmsk_rf, rms=0.01)

    #plt.figure(0)
    #plot_td(gmsk_rf)
    #plt.show()
    #plt.figure(1)
    plt.subplot(2,2,4)
    plot_fd(gmsk_rf, label="GMSK", title="- Tx")
    plt.show()

    gmsk_rf = gaussian_fade(gmsk_rf, f = 1000)

    n = int(round(gmsk_rf.fs/float(gmsk_i.fs)))
    rx_i, rx_q = downconvert_rf(CARRIER, gmsk_rf)
    #downsample to original IQ rate
    rx_i = filter_and_downsample(rx_i, n=n)
    rx_q = filter_and_downsample(rx_q, n=n)

    rx_i = butter_lowpass_filter(rx_i, cutoff = 2.0*BW)
    rx_q = butter_lowpass_filter(rx_q, cutoff = 2.0*BW)
    plt.subplot(2,2,1)
    plot_constellation_density(rx_i, rx_q, title="- Rx")
    plt.subplot(2,2,2)
    plot_iq_phase_mag(rx_i, rx_q, title="- Rx")
    plt.subplot(2,2,3)
    plot_phase_histogram(rx_i, rx_q, title="- Rx")

    demodulated = demodulate_gmsk(rx_i, rx_q, OVERSAMPLING)
    plt.subplot(2,2,4)
    plot_td(demodulated, title="- Rx")

    plt.show()


    fir_matched_kaiser = gmsk_matched_kaiser_rx_filter(OVERSAMPLING, int(2*PULSE_SPAN), BT_TX, BT_COMPOSITE, 0.0)
    fir_matched_rcos = gmsk_matched_rcos_rx_filter(OVERSAMPLING, int(2*PULSE_SPAN), BT_TX, BT_COMPOSITE, 0.0)
    demod_kaiser = rx_filter(demodulated, fir_matched_kaiser, OVERSAMPLING)
    demod_rcos = rx_filter(demodulated, fir_matched_rcos, OVERSAMPLING)
    #demod_mf = matched_filter(demodulated, BT, OVERSAMPLING, PULSE_SPAN)
    #plot_td(demodulated)
    #plt.plot(_fir_taps)
    #fir = make_signal(td=fir_taps_cos, fs=demodulated.fs)
    #plot_fd(fir)
    #plt.show()
    plt.figure(1)
    #plot_fd(demod_filt)
    plot_fd(demod_rcos, label="RCOS beta =%.2f"%BT_COMPOSITE)
    plot_fd(demod_kaiser, label="Kaiser beta=%.2f"%BT_COMPOSITE, title="- Rx")
    #plt.show()
    plt.figure(2)
    plt.subplot(3,1,1)
    plot_eye_density(demodulated, _3d=False, pools=8, title="- NO MATCHED FILTER")
    plt.subplot(3,1,2)
    #plot_eye_lines(demod_filt, recovery="constant_f")
    plot_eye_density(demod_kaiser, _3d=False, pools=8, title="- Kasier beta =%.2f"%BT_COMPOSITE)
    plt.subplot(3,1,3)
    #plot_eye_lines(demodulated, recovery="constant_f")
    plot_eye_density(demod_rcos, _3d=False, pools=None, title=" - RCOS beta =%.2f"%BT_COMPOSITE)
    #plt.show()
    #plt.figure(3)
    #plot_tie(demod_filt)
    #plot_tie(demodulated)
    #plot_jitter_histogram(demod_filt)
    #plot_jitter_histogram(demodulated)
    plt.show()
