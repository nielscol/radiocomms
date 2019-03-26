""" Simulates Tx+Rx for GMSK
    Cole Nielsen 2019

    NOTE: THIS HAS NOT BEEN KEPT UP TO DATE WITH THE STANDARD OF THE GMSK SIMULATION
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import generate_random_bitstream, make_signal
from lib.modulation import generate_msk_baseband, generate_gmsk_baseband, upconvert_baseband, downconvert_rf, demodulate_gmsk
from lib.plot import *  # Assuming all plotting methods come from here
from lib.transforms import butter_lowpass_filter, fir_filter, add_noise, sum_signals, filter_and_downsample, gaussian_fade
from lib.analysis import measure_rms
from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter

if __name__ == "__main__":
    #gaussian_to_raised_cosine(0.3, 16, 4)

    N_BITS = 5000
    BIT_RATE = 64000 + 2400
    OVERSAMPLING = 16

    PULSE_SPAN = 8
    AGWN_RMS = 0.00707

    SPECTRAL_EFF = 1.0 # bits/s/Hz
    BW = BIT_RATE/(SPECTRAL_EFF) # theoretical signal bandwidth
    CARRIER = 5.0*BW

    RX_LPF = None # AA filter
    CMAP = "nipy_spectral"
    _3D_EYE = False
    POOLS = 8

    #
    # Begin simulation
    #

    # make message
    message = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)

    # generate baseband and upconvert to carrier
    msk_i, msk_q = generate_msk_baseband(message, OVERSAMPLING)
    msk_rf = upconvert_baseband(CARRIER, msk_i, msk_q)

    plt.figure(1)
    plt.subplot(2,2,1)
    plot_constellation_density(msk_i, msk_q, title="- Tx")
    plt.subplot(2,2,2)
    plot_iq_phase_mag(msk_i, msk_q, title="- Tx")
    plt.subplot(2,2,3)
    plot_phase_histogram(msk_i, msk_q, title="- Tx")

    #
    # Simulate channel (AWGN and fading)
    # 

    # AWGN
    msk_rf = add_noise(msk_rf, rms=AGWN_RMS)

    msk_rf_rms = measure_rms(msk_rf)
    print("\n* MSK signal SNR = %.2f dB"%(20*np.log10(msk_rf_rms/AGWN_RMS)))

    plt.subplot(2,2,4)
    plot_fd(msk_rf, label="MSK", title="- Tx")

    # simulate fading
    msk_rf = gaussian_fade(msk_rf, f = 1000)

    #
    # Downconvert and demodulate
    #

    rf_oversampling = int(round(msk_rf.fs/float(msk_i.fs))) # oversampling factor from baseband -> RF
    rx_i, rx_q = downconvert_rf(CARRIER, msk_rf)
    # downsample to original IQ rate
    rx_i = filter_and_downsample(rx_i, n=rf_oversampling)
    rx_q = filter_and_downsample(rx_q, n=rf_oversampling)

    # Rx AA filter?
    if RX_LPF:
        rx_i = butter_lowpass_filter(rx_i, cutoff = RX_LPF)
        rx_q = butter_lowpass_filter(rx_q, cutoff = RX_LPF)

    plt.figure(2)
    plt.subplot(2,2,1)
    plot_constellation_density(rx_i, rx_q, title="- Rx")
    plt.subplot(2,2,2)
    plot_iq_phase_mag(rx_i, rx_q, title="- Rx")
    plt.subplot(2,2,3)
    plot_phase_histogram(rx_i, rx_q, title="- Rx")

    demodulated = demodulate_gmsk(rx_i, rx_q, OVERSAMPLING)
    plt.subplot(2,2,4)
    plot_td(demodulated, title="- Rx")


    #
    # Generate eye diagram
    #

    plt.figure(3)
    plot_eye_density(demodulated, _3d=_3D_EYE, pools=POOLS, title="- NO MATCHED FILTER", cmap=CMAP)
    #plot_tie(demodulated)
    #plot_jitter_histogram(demodulated)
    plt.show()
