""" Simulates Tx+Rx for GMSK
    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from lib._signal import generate_random_bitstream, make_signal
from lib.modulation import generate_msk_baseband, generate_gmsk_baseband, upconvert_baseband, downconvert_rf, demodulate_gmsk
from lib.plot import *  # Assuming all plotting methods come from here
from lib.transforms import butter_lpf, cheby2_lpf, fir_filter, fir_correlate, add_noise, sum_signals, filter_and_downsample, gaussian_fade, fft_resample
from lib.analysis import measure_rms
from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter, kaiser_composite_tx_rx_filter, rcos_composite_tx_rx_filter
from lib.sync import get_precomputed_codes, make_sync_fir, code_string, frame_data, detect_sync

if __name__ == "__main__":
    #gaussian_to_raised_cosine(0.3, 16, 4)

    N_BITS = 5000
    BIT_RATE = 64000 + 2400 # Audio rate + sync overhead
    OVERSAMPLING = 8
    IQ_RATE = BIT_RATE*OVERSAMPLING

    FRAME_PAYLOAD = 640 # bits
    SYNC_PULSE_SPAN = 8
    SYNC_CODE_LEN = 24 # from precomputed values up to N=24
    SYNC_POS = "center"

    BT_TX = 0.3         # GMSK BT
    BT_COMPOSITE = 1.0  # response of combined Rx+Tx filter
    TX_FIR_SPAN = 4.0      # extent of FIR filters in # of symbols
    RX_FIR_SPAN = 8.0

    AGWN_RMS = 0.00707 # 0.00707 -> 40 dB SNR, 0.0707 -> 20 dB, 0.223 -> 10 dB, 0.4 -> 6 dB

    SPECTRAL_EFF = 1.0 # bits/s/Hz
    BW = BIT_RATE/(SPECTRAL_EFF) # theoretical signal bandwidth
    TX_LO = 5.0*BW
    RX_LO = 5.0*BW
    LO_PHASE_NOISE = 0.0 # rms in radians, applied to both Rx/Tx identically
    INTERFERER_CARRIER = 3.0*BW

    FADING = True
    FADE_FREQ = 500      # Hz
    FADE_SIGMA = 2.0     # Standard deviations from mean corresponding to peak-peak value
    FADE_PEAK_PEAK = 40  # dB - depends alot on simulation fs, fade freq, and fade sigma

    RX_LPF = 1.0*BW # Corner, AA filter/Interference rejection
    RX_LPF_TYPE = "butter" # "butter" or "cheby2"
    ORDER = 5
    CHEBY_STOP_ATTEN = 40.0 # dB of stopband attenuation

    _3D_EYE = False
    EYE_VPP = 3.0
    CMAP = "nipy_spectral" # "inferno" 
    SAMPLE_LINES = True # Draw lines corresponding to where samples occur
    POOLS = 8 # For running on multiple processes (parallelization)

    PLOT_TX = True
    PLOT_RX = True
    PLOT_RX_EYES = True
    PLOT_RX_EYE_SYNC_RECOVERY = True
    PLOT_RX_JITTER = True
    PLOT_RX_FILTER = True
    PLOT_RX_DATA_PSD = True
    PLOT_SYNC = True
    #
    # Begin simulation
    #
    fig_num = 0

    # make random message bitstream
    sync_codes = get_precomputed_codes()
    sync_code = sync_codes[SYNC_CODE_LEN]

    message = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)
    message = frame_data(message, sync_code, FRAME_PAYLOAD, BIT_RATE, BIT_RATE, sync_pos=SYNC_POS)

    message_interferer = generate_random_bitstream(length=N_BITS, bitrate=BIT_RATE)
    message_interferer = frame_data(message_interferer, sync_code, FRAME_PAYLOAD, BIT_RATE, BIT_RATE, sync_pos=SYNC_POS)

    # make GMSK baseband + RF
    gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX,
                                            pulse_span=TX_FIR_SPAN)
    gmsk_rf = upconvert_baseband(TX_LO, gmsk_i, gmsk_q, rms_phase_noise=LO_PHASE_NOISE)

    # make MSK baseband + RF as interferer
    msk_i, msk_q = generate_msk_baseband(message_interferer, OVERSAMPLING)
    msk_interferer_rf = upconvert_baseband(INTERFERER_CARRIER, msk_i, msk_q, rms_phase_noise=LO_PHASE_NOISE)
    msk_interferer_rf = fft_resample(msk_interferer_rf, n_samples=len(gmsk_rf.td))
    # add GMSK and interferer
    rf = sum_signals(gmsk_rf, msk_interferer_rf, bitrate="signal_a")

    #
    # Simulate channel (AWGN + fading)
    #

    # Simulate AWGN
    rf = add_noise(rf, rms=AGWN_RMS)
    #gmsk_rf = add_noise(gmsk_rf, rms=AGWN_RMS)
    #msk_interferer_rf = add_noise(msk_interferer_rf, rms=AGWN_RMS)

    gmsk_rf_rms = measure_rms(gmsk_rf)
    print("\n* GMSK signal SNR = %.2f dB"%(20*np.log10(gmsk_rf_rms/AGWN_RMS)))

    # Plot PSD of signals mixed to carrier
    if PLOT_TX:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(2,2,1)
        plot_constellation_density(gmsk_i, gmsk_q, title="- Tx")
        plt.subplot(2,2,2)
        plot_iq_phase_mag(gmsk_i, gmsk_q, title="- Tx")
        plt.subplot(2,2,3)
        plot_phase_histogram(gmsk_i, gmsk_q, title="- Tx")
        plt.subplot(2,2,4)
        #plot_fd(gmsk_rf, label="GMSK", title="- Tx", alpha=0.8)
        #plot_fd(msk_interferer_rf, label="MSK Interferer", title="- Tx", alpha=0.8)
        plot_fd(rf, label="GMSK+MSK Interferer", title="- Tx", alpha=0.8)

    # Simulate fading
    if FADING:
        rf = gaussian_fade(gmsk_rf, f=FADE_FREQ, peak_peak=FADE_PEAK_PEAK, n_sigma=FADE_SIGMA) # this is really arbitrary...

    #
    # Downconvert RF and demodulate 
    #

    rf_oversampling = int(round(gmsk_rf.fs/float(IQ_RATE))) # oversampling factor from baseband -> RF
    rx_i, rx_q = downconvert_rf(RX_LO, rf, rms_phase_noise=LO_PHASE_NOISE)
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

    demodulated = demodulate_gmsk(rx_i, rx_q, OVERSAMPLING)

    if PLOT_RX:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(2,2,1)
        plot_constellation_density(rx_i, rx_q, title="- Rx")
        plt.subplot(2,2,2)
        plot_iq_phase_mag(rx_i, rx_q, title="- Rx")
        plt.subplot(2,2,3)
        plot_phase_histogram(rx_i, rx_q, title="- Rx")
        plt.subplot(2,2,4)
        plot_td(demodulated, title="- Rx")

    #
    # Simulte Rx matched filter
    #

    fir_matched_kaiser = gmsk_matched_kaiser_rx_filter(OVERSAMPLING, RX_FIR_SPAN, BT_TX,
                                                       BT_COMPOSITE, fs=IQ_RATE)
    fir_matched_rcos = gmsk_matched_rcos_rx_filter(OVERSAMPLING, RX_FIR_SPAN, BT_TX,
                                                   BT_COMPOSITE, fs=IQ_RATE)
    demod_kaiser = fir_filter(demodulated, fir_matched_kaiser, OVERSAMPLING)
    demod_rcos = fir_filter(demodulated, fir_matched_rcos, OVERSAMPLING)
    #plot_td(demodulated)
    #plt.plot(_fir_taps)

    #
    # Plot Rx data
    #

    if PLOT_RX_FILTER:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,2,1)
        plot_fd(fir_matched_kaiser, label="Rx Matched Filter - KAISER", alpha=0.9)
        plot_fd(fir_matched_rcos, label="Rx Matched Filter - RCOS", title="- Rx filter response", alpha=0.9)
        plt.subplot(1,2,2)
        plot_td(fir_matched_kaiser, label="Rx Matched Filter - KAISER", alpha=0.9)
        plot_td(fir_matched_rcos, label="Rx Matched Filter - RCOS", title="- Rx FIR", alpha=0.9)

    if PLOT_RX_DATA_PSD:
        fig_num += 1
        plt.figure(fig_num)
        #plot_fd(demod_filt)
        plot_fd(demod_kaiser, label="Kaiser beta=%.2f"%BT_COMPOSITE, alpha=0.8)
        plot_fd(demod_rcos, label="RCOS beta =%.2f"%BT_COMPOSITE, title="- Rx Data", alpha=0.8)

    #
    # Make eye diagrams
    #

    if PLOT_RX_EYES:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,3,1)
        plot_eye_density(demodulated, _3d=_3D_EYE, pools=POOLS, title="- NO MATCHED FILTER",
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING)
        plt.subplot(1,3,2)
        plot_eye_density(demod_kaiser, _3d=_3D_EYE, pools=POOLS, title="- Kasier beta =%.2f"%BT_COMPOSITE,
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING)
        plt.subplot(1,3,3)
        plot_eye_density(demod_rcos, _3d=_3D_EYE, pools=POOLS, title=" - RCOS beta =%.2f"%BT_COMPOSITE,
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING)

    if PLOT_RX_JITTER:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,2,1)
        plot_tie(demodulated, alpha=0.8, label="NO MATCHED FILTER")
        plot_tie(demod_kaiser, alpha=0.8, label="Kaiser BT=%.2f"%BT_COMPOSITE)
        plot_tie(demod_rcos, alpha=0.8, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- Rx")

        plt.subplot(1,2,2)
        plot_jitter_histogram(demodulated, alpha=0.8, label="NO MATCHED FILTER")
        plot_jitter_histogram(demod_kaiser, alpha=0.8, label="Kaiser BT=%.2f"%BT_COMPOSITE)
        plot_jitter_histogram(demod_rcos, alpha=0.8, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- Rx")


    #
    # SYNCHRONIZATION
    #

    kaiser_fir = kaiser_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE, fs=IQ_RATE)
    rcos_fir = rcos_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE, fs=IQ_RATE)
    sync_codes = get_precomputed_codes()
    sync_code = sync_codes[SYNC_CODE_LEN]
    print("\n* Applying synchronization FIR sequence to demodulated Rx signal")
    print("\t%s"%code_string(sync_code))
    sync_fir_kaiser = make_sync_fir(sync_code, kaiser_fir, OVERSAMPLING)
    sync_fir_rcos = make_sync_fir(sync_code, rcos_fir, OVERSAMPLING)
    sync_correl_kaiser = fir_correlate(demod_kaiser, sync_fir_kaiser, OVERSAMPLING)
    sync_correl_rcos = fir_correlate(demod_rcos, sync_fir_rcos, OVERSAMPLING)

    if PLOT_RX_EYE_SYNC_RECOVERY:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,2,1)
        plot_eye_density(demod_kaiser, _3d=_3D_EYE, pools=POOLS, title="SYNC RECOVERY - Kasier beta =%.2f"%BT_COMPOSITE,
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING,
                         sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS,
                         recovery="frame_sync")
        plt.subplot(1,2,2)
        plot_eye_density(demod_rcos, _3d=_3D_EYE, pools=POOLS, title="SYNC RECOVERY - RCOS beta =%.2f"%BT_COMPOSITE,
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING,
                         sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS,
                         recovery="frame_sync")

    if PLOT_SYNC:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,2,1)
        plot_td(sync_correl_kaiser, label="Kaiser BT=%.2f"%BT_COMPOSITE, alpha=0.8)
        plot_td(sync_correl_rcos, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- Sync Correlation", alpha=0.8)
        plt.subplot(1,2,2)
        plot_histogram(sync_correl_kaiser, label="Kaiser BT=%.2f"%BT_COMPOSITE,
                       fit_normal=True, alpha=0.8, orientation="horizontal")
        plot_histogram(sync_correl_rcos, label="RCOS BT=%.2f"%BT_COMPOSITE, fit_normal=True,
                       xlabel="Correlation", title="- Sync Correlation", alpha=0.8, orientation="horizontal")
        plt.legend()
    plt.show()


    ind, val = detect_sync(sync_correl_kaiser, sync_code, FRAME_PAYLOAD, OVERSAMPLING)
    for n, i in enumerate(ind):
        print(i, val[n])
