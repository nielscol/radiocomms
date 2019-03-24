""" Simulates Tx+Rx for GMSK
    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import generate_random_bitstream, make_signal
from lib.modulation import generate_msk_baseband, generate_gmsk_baseband, upconvert_baseband, downconvert_rf, demodulate_gmsk
from lib.plot import *  # Assuming all plotting methods come from here
from lib.transforms import butter_lpf, cheby2_lpf, fir_filter, add_noise, sum_signals, filter_and_downsample, gaussian_fade
from lib.analysis import measure_rms
from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter

if __name__ == "__main__":
    #gaussian_to_raised_cosine(0.3, 16, 4)

    N_BITS = 5000
    BIT_RATE = 64000 + 2400 # Audio rate + sync overhead
    OVERSAMPLING = 8

    BT_TX = 0.3         # GMSK BT
    BT_COMPOSITE = 1.0  # response of combined Rx+Tx filter
    PULSE_SPAN = 3      # extent of FIR filters in # of symbols

    AGWN_RMS = 0.00707 # 0.00707 -> 40 dB SNR, 0.0707 -> 20 dB, 0.223 -> 10 dB, 0.4 -> 6 dB

    SPECTRAL_EFF = 1.0 # bits/s/Hz
    BW = BIT_RATE/(SPECTRAL_EFF) # theoretical signal bandwidth
    CARRIER = 5.0*BW

    FADING = True

    RX_LPF = 1.0*BW # Corner, AA filter/Interference rejection
    RX_LPF_TYPE = "butter" # "butter" or "cheby2"
    ORDER = 5
    CHEBY_STOP_ATTEN = 40.0 # dB of stopband attenuation

    _3D_EYE = False
    EYE_VPP = 3.0
    CMAP = "nipy_spectral"
    POOLS = 8

    #
    # Begin simulation
    #

    # make random message bitstream
    message = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)

    # make GMSK baseband + RF
    gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX, pulse_span=PULSE_SPAN)
    gmsk_rf = upconvert_baseband(CARRIER, gmsk_i, gmsk_q)

    # make MSK baseband + RF for comparison
    msk_i, msk_q = generate_msk_baseband(message, OVERSAMPLING)
    msk_rf = upconvert_baseband(CARRIER, msk_i, msk_q)

    plt.figure(1)
    plt.subplot(2,2,1)
    plot_constellation_density(gmsk_i, gmsk_q, title="- Tx")
    plt.subplot(2,2,2)
    plot_iq_phase_mag(gmsk_i, gmsk_q, title="- Tx")
    plt.subplot(2,2,3)
    plot_phase_histogram(gmsk_i, gmsk_q, title="- Tx")

    #
    # Simulate channel (AWGN + fading)
    #

    # Simulate AWGN
    msk_rf = add_noise(msk_rf, rms=AGWN_RMS)
    gmsk_rf = add_noise(gmsk_rf, rms=AGWN_RMS)

    msk_rf_rms = measure_rms(msk_rf)
    gmsk_rf_rms = measure_rms(gmsk_rf)
    print("\n* MSK signal SNR = %.2f dB"%(20*np.log10(msk_rf_rms/AGWN_RMS)))
    print("\n* GMSK signal SNR = %.2f dB"%(20*np.log10(gmsk_rf_rms/AGWN_RMS)))

    # Plot PSD of signals mixed to carrier
    plt.subplot(2,2,4)
    plot_fd(msk_rf, label="MSK", alpha=0.8)
    plot_fd(gmsk_rf, label="GMSK", title="- Tx", alpha=0.8)

    # Simulate fading
    if FADING:
        gmsk_rf = gaussian_fade(gmsk_rf, f = 1000) # this is really arbitrary...

    #
    # Downconvert RF and demodulate 
    #

    rf_oversampling = int(round(gmsk_rf.fs/float(gmsk_i.fs))) # oversampling factor from baseband -> RF
    rx_i, rx_q = downconvert_rf(CARRIER, gmsk_rf)
    # downsample to original IQ rate
    rx_i = filter_and_downsample(rx_i, n=rf_oversampling)
    rx_q = filter_and_downsample(rx_q, n=rf_oversampling)

    # Interferer/IF/AA filter?
    if RX_LPF:
        if RX_LPF_TYPE == "butter":
            rx_i = butter_lpf(rx_i, cutoff = RX_LPF, order=ORDER)
            rx_q = butter_lpf(rx_q, cutoff = RX_LPF, order=ORDER)
        elif RX_LPF_TYPE == "cheby2":
            rx_i = cheby2_lpf(rx_i, cutoff = RX_LPF, stop_atten=CHEBY_STOP_ATTEN, order=ORDER)
            rx_q = cheby2_lpf(rx_q, cutoff = RX_LPF, stop_atten=CHEBY_STOP_ATTEN, order=ORDER)

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
    # Simulte Rx matched filter
    #

    fir_matched_kaiser = gmsk_matched_kaiser_rx_filter(OVERSAMPLING, int(2*PULSE_SPAN), BT_TX, BT_COMPOSITE, 0.0)
    fir_matched_rcos = gmsk_matched_rcos_rx_filter(OVERSAMPLING, int(2*PULSE_SPAN), BT_TX, BT_COMPOSITE, 0.0)
    demod_kaiser = fir_filter(demodulated, fir_matched_kaiser, OVERSAMPLING)
    demod_rcos = fir_filter(demodulated, fir_matched_rcos, OVERSAMPLING)
    #plot_td(demodulated)
    #plt.plot(_fir_taps)
    sig_fir_kaiser = make_signal(td=fir_matched_kaiser, fs=demodulated.fs)
    sig_fir_rcos = make_signal(td=fir_matched_rcos, fs=demodulated.fs)

    plt.figure(3)
    plt.subplot(1,2,1)
    plot_fd(sig_fir_kaiser, label="Rx Matched Filter - KAISER", alpha=0.9)
    plot_fd(sig_fir_rcos, label="Rx Matched Filter - RCOS", title="- Rx filter response", alpha=0.9)
    plt.subplot(1,2,2)
    plot_td(sig_fir_kaiser, label="Rx Matched Filter - KAISER", alpha=0.9)
    plot_td(sig_fir_rcos, label="Rx Matched Filter - RCOS", title="- Rx FIR", alpha=0.9)

    #
    # Make eye diagrams
    #

    plt.figure(4)
    #plot_fd(demod_filt)
    plot_fd(demod_kaiser, label="Kaiser beta=%.2f"%BT_COMPOSITE, alpha=0.8)
    plot_fd(demod_rcos, label="RCOS beta =%.2f"%BT_COMPOSITE, title="- Rx Data", alpha=0.8)
    #plt.show()
    plt.figure(5)
    plt.subplot(1,3,1)
    plot_eye_density(demodulated, _3d=_3D_EYE, pools=POOLS, title="- NO MATCHED FILTER", cmap=CMAP, eye_vpp=EYE_VPP)
    plt.subplot(1,3,2)
    plot_eye_density(demod_kaiser, _3d=_3D_EYE, pools=POOLS, title="- Kasier beta =%.2f"%BT_COMPOSITE, cmap=CMAP, eye_vpp=EYE_VPP)
    plt.subplot(1,3,3)
    plot_eye_density(demod_rcos, _3d=_3D_EYE, pools=POOLS, title=" - RCOS beta =%.2f"%BT_COMPOSITE, cmap=CMAP, eye_vpp=EYE_VPP)
    #plot_tie(demod_filt)
    #plot_tie(demodulated)
    #plt.subplot(2,3,4)
    #plot_jitter_histogram(demodulated)
    #plt.subplot(2,3,5)
    #plot_jitter_histogram(demod_kaiser)
    #plt.subplot(2,3,6)
    #plot_jitter_histogram(demod_rcos)
    plt.show()
