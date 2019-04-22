""" Simulates Tx+Rx for GMSK
    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
from lib._signal import generate_random_bitstream, make_signal
from lib.modulation import generate_msk_baseband, generate_gmsk_baseband, upconvert_baseband, downconvert_rf, demodulate_gmsk
from lib.plot import *  # Assuming all plotting methods come from here
from lib.transforms import butter_lpf, cheby2_lpf, fir_filter, fir_correlate, add_noise, sum_signals, filter_and_downsample, gaussian_fade, fft_resample
from lib.analysis import measure_rms
from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter, kaiser_composite_tx_rx_filter, rcos_composite_tx_rx_filter, gmsk_tx_filter
from lib.sync import get_precomputed_codes, make_sync_fir, code_string, frame_data, detect_sync

if __name__ == "__main__":
    #gaussian_to_raised_cosine(0.3, 16, 4)

    N_BITS = 5000                   # number of data bits in simulation
    BITRATE = 64000 + 2400          # Audio rate + sync overhead
    OVERSAMPLING = 8                # Rate beyond bit rate which baseband will sample
    IQ_RATE = BITRATE*OVERSAMPLING  # Baseband sampling rate

    FRAME_PAYLOAD = 640             # Number of data bits per fram
    SYNC_CODE_LEN = 24              # Length of sync field in bits, from precomputed values up to N=24
    SYNC_POS = "center"             # Where to place sync field in frame, {"center", "start"}
    SYNC_PULSE_SPAN = 8             # FIR span in symbols of pulse shape used to filter sync code for 
                                        # detection of frame synchronization
    BT_TX = 0.3                     # GMSK BT
    BT_COMPOSITE = 1.0              # response of combined Rx+Tx filter
    TX_FIR_SPAN = 4.0               # extent of Tx pulse shaping FIR filter in # of symbols
    RX_FIR_SPAN = 8.0               # Extent of Rx matched FIR filter in # of symbols

    SPECTRAL_EFF = 1.0              # bits/s/Hz, 1.0 for MSK and GMSK
    BW = BITRATE/(SPECTRAL_EFF)     # theoretical signal bandwidth
    TX_LO = 5.0*BW                  # Tx carrier, do not set too high as the simulation points will grow large
    RX_LO = 5.0*BW                  # Rx carrier
    LO_PHASE_NOISE = 0.0            # Rms in radians, applied to both Rx/Tx LO identically

    SHOW_MSK_TX = False             # Will plot spectrum of MSK at parameters as GMSK for comparison

    SIMULATE_INTERFERER = True      # Simulate a nearby interfering signal
    INTERFERER_LO = 3.0*BW          # Interferer carrier
    INTERFERER_BITRATE = BITRATE    # Interferer bitrate
    INTERFERER_N_BITS = N_BITS      # Interferer data bits
    INTERFERER_FRAME_PAYLOAD = FRAME_PAYLOAD    # Frame data payload len 
    INTERFERER_MODULATION = "MSK"               # {"MSK", "GMSK"}
    INTERFERER_BT = BT_TX                       # for GMSK only
    INTERFERER_FIR_SPAN = TX_FIR_SPAN           # for GMSK Tx FIR pulse shaping span
    INTERFERER_OVERSAMPLING = OVERSAMPLING      # Oversampling rate
    INTERFERER_CORREL_SYNC = False              # Should the sync be the same as the GMSK?

    AGWN_RMS = 0.4                  # 0.00707 -> 40 dB SNR, 0.0707 -> 20 dB, 0.223 -> 10 dB, 0.4 -> 6 dB
    FADING = True                   # Apply fading to the channel?
    FADE_FREQ = 500                 # Hz - each cycle includes a min and max peak corresponding to sigma
    FADE_SIGMA = 2.0                # Standard deviations from mean corresponding to peak-peak value, probably keep at ~2.0
    FADE_PEAK_PEAK = 40             # dB - actual fade depends alot on simulation fs, fade freq, and fade sigma

    RX_LPF = 1.0*BW                 # Corner freqency of LPF on Rx I/Q to reject aliasing/interferers
    RX_LPF_TYPE = "butter"          # Filter type "butter" or "cheby2"
    ORDER = 5                       # Filter order
    CHEBY_STOP_ATTEN = 40.0         # dB of stopband attenuation

    _3D_EYE = False                 # Plot eye diagrams in 3D!
    EYE_VPP = 3.0                   # Set if you want to force a p-p eye value. Range of plot will be 1.25*EYE_VPP
    CMAP = "nipy_spectral"          # Color map of plot, best options are (1) "nipy_spectral" and (2) "inferno"
    SAMPLE_LINES = False            # Draw lines corresponding to where samples occur
    POOLS = 8                       # For using multiple processes to compute eyes. POOLS = # logical processors on your computer

    RECOVERY_METHOD = "frame_sync"  # {"frame_sync", "constant_f", "edge"}, use "frame_sync" if data if framed, else "constant_f"

    PLOT_TX = True                  # Plot Tx-related plots (Constellation, Phase/magnitude of IQ, spectrum)
    PLOT_RX = True                  # Plot Rx-related plots (Constellation, IQ Phase/magnitude, raw demodulated data)
    PLOT_RX_EYES = True             # Plot Rx eye diagrams
    PLOT_RX_JITTER = True           # Plot jitter plots (Total interval error, jitter distribution, jitter spectrum)
    PLOT_RX_FILTER = True           # Plot Rx filter taps and responses
    PLOT_RX_DATA_PSD = True         # Plot Rx data PSD
    PLOT_SYNC = True                # Plot Sync-related plots (Correlation of sync to data, histogram of aforementioned and detected syncs)

    #
    # Begin simulation
    #

    fig_num = 0

    # make message with data from bitstream in frame structure
    sync_codes = get_precomputed_codes()
    sync_code = sync_codes[SYNC_CODE_LEN]

    message = generate_random_bitstream(length=N_BITS, bitrate=BITRATE)
    message = frame_data(message, sync_code, FRAME_PAYLOAD, BITRATE, BITRATE, sync_pos=SYNC_POS)

    # make GMSK baseband + RF
    gmsk_i, gmsk_q = generate_gmsk_baseband(message, OVERSAMPLING, bt=BT_TX,
                                            pulse_span=TX_FIR_SPAN)
    rf = upconvert_baseband(TX_LO, gmsk_i, gmsk_q, rms_phase_noise=LO_PHASE_NOISE)

    #
    # SIMULATE NEARBY INTERFERING SIGNAL
    #

    if SIMULATE_INTERFERER:
        if INTERFERER_CORREL_SYNC:
            int_sync_code = sync_code
        else:
            int_sync_code = np.random.choice([-1.0, 1.0], len(sync_code))
        message_interferer = generate_random_bitstream(length=INTERFERER_N_BITS, bitrate=INTERFERER_BITRATE)
        message_interferer = frame_data(message_interferer, int_sync_code, INTERFERER_FRAME_PAYLOAD, INTERFERER_BITRATE, INTERFERER_BITRATE, sync_pos=SYNC_POS)

        if INTERFERER_MODULATION == "MSK":
            int_i, int_q = generate_msk_baseband(message_interferer, INTERFERER_OVERSAMPLING)
        elif INTERFERER_MODULATION == "GMSK":
            int_i, int_q = generate_gmsk_baseband(message_interferer, INTERFERER_OVERSAMPLING,
                                                 bt=INTERFERER_BT, pulse_span=INTERFERER_FIR_SPAN)
        int_rf = upconvert_baseband(INTERFERER_LO, int_i, int_q, rms_phase_noise=LO_PHASE_NOISE)
        int_rf = fft_resample(int_rf, n_samples=len(rf.td)) # force rf and interferer to same len
        # add GMSK and interferer
        rf = sum_signals(rf, int_rf, bitrate="signal_a")

    # make MSK baseband + RF for comparison
    if SHOW_MSK_TX:
        msk_i, msk_q = generate_msk_baseband(message, OVERSAMPLING)
        msk_rf = upconvert_baseband(TX_LO, msk_i, msk_q, rms_phase_noise=LO_PHASE_NOISE)
        # Simulate AWGN
        msk_rf = add_noise(msk_rf, rms=AGWN_RMS)

    #
    # Simulate channel (AWGN + fading)
    #

    # Simulate AWGN
    rf = add_noise(rf, rms=AGWN_RMS)

    rf_rms = measure_rms(rf)
    print("\n* Signal SNR = %.2f dB"%(20*np.log10(rf_rms/AGWN_RMS)))

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
        plot_fd(rf, label="GMSK", title="- Tx", alpha=0.8)
        if SHOW_MSK_TX:
            plot_fd(msk_rf, label="MSK", alpha=0.8)

    # Simulate fading
    if FADING:
        rf = gaussian_fade(rf, f=FADE_FREQ, peak_peak=FADE_PEAK_PEAK, n_sigma=FADE_SIGMA) # this is really arbitrary...

    #
    # Downconvert RF and demodulate 
    #

    rf_oversampling = int(round(rf.fs/float(IQ_RATE))) # oversampling factor from baseband -> RF
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
    # SYNCHRONIZATION
    #

    kaiser_fir = kaiser_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                               fs=IQ_RATE, norm=True)
    rcos_fir = rcos_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                           fs=IQ_RATE, norm=True)
    unmatched_fir = gmsk_tx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, fs=IQ_RATE, norm=True)
    print("\n* Applying synchronization FIR sequence to demodulated Rx signal")
    print("\t%s"%code_string(sync_code))
    sync_fir_kaiser = make_sync_fir(sync_code, kaiser_fir, OVERSAMPLING)
    sync_fir_rcos = make_sync_fir(sync_code, rcos_fir, OVERSAMPLING)
    sync_fir_unmatched = make_sync_fir(sync_code, unmatched_fir, OVERSAMPLING)
    sync_correl_kaiser = fir_correlate(demod_kaiser, sync_fir_kaiser, OVERSAMPLING)
    sync_correl_rcos = fir_correlate(demod_rcos, sync_fir_rcos, OVERSAMPLING)
    sync_correl_unmatched = fir_correlate(demodulated, sync_fir_unmatched, OVERSAMPLING)

    if PLOT_SYNC:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,2,1)
        plot_td(sync_correl_unmatched, label="Unmatched", alpha=0.8)
        plot_td(sync_correl_kaiser, label="Kaiser BT=%.2f"%BT_COMPOSITE, alpha=0.8)
        plot_td(sync_correl_rcos, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- Sync Correlation", alpha=0.8)
        plt.subplot(1,2,2)
        plot_sync_detect_bellcurve(sync_correl_unmatched, sync_code, FRAME_PAYLOAD, OVERSAMPLING,
                                   label="Unmatched", orientation="horizontal")
        plot_sync_detect_bellcurve(sync_correl_kaiser, sync_code, FRAME_PAYLOAD, OVERSAMPLING,
                                   label="Kaiser BT=%.2f"%BT_COMPOSITE, orientation="horizontal")
        plot_sync_detect_bellcurve(sync_correl_rcos, sync_code, FRAME_PAYLOAD, OVERSAMPLING,
                                   label="RCOS BT=%.2f"%BT_COMPOSITE, orientation="horizontal")
        plot_histogram(sync_correl_kaiser, label="Kaiser BT=%.2f"%BT_COMPOSITE,
                       fit_normal=True, alpha=0.8, orientation="horizontal")
        plot_histogram(sync_correl_rcos, label="RCOS BT=%.2f"%BT_COMPOSITE, fit_normal=True,
                       xlabel="Correlation", title="- Sync Correlation", alpha=0.8, orientation="horizontal")
        plot_histogram(sync_correl_unmatched, label="Unmatched", title="- Sync Correlation to Data,\nDetected Sync Distributions",
                       fit_normal=True, alpha=0.8, orientation="horizontal")

    #
    # Make eye diagrams
    #

    if PLOT_RX_EYES:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,3,1)
        plot_eye_density(demodulated, _3d=_3D_EYE, pools=POOLS, title="- %s - No Matched Filter"%RECOVERY_METHOD,
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING,
                         sync_code=sync_code, pulse_fir=unmatched_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS,
                         recovery=RECOVERY_METHOD)
        plt.subplot(1,3,2)
        plot_eye_density(demod_kaiser, _3d=_3D_EYE, pools=POOLS, title="- %s - Kasier beta =%.2f"%(RECOVERY_METHOD, BT_COMPOSITE),
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING,
                         sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS,
                         recovery=RECOVERY_METHOD)
        plt.subplot(1,3,3)
        plot_eye_density(demod_rcos, _3d=_3D_EYE, pools=POOLS, title="- %s - RCOS beta =%.2f"%(RECOVERY_METHOD, BT_COMPOSITE),
                         cmap=CMAP, eye_vpp=EYE_VPP, sample_lines=SAMPLE_LINES, oversampling=OVERSAMPLING,
                         sync_code=sync_code, pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS,
                         recovery=RECOVERY_METHOD)

    #
    # Make Jitter diagrams
    #

    if PLOT_RX_JITTER:
        fig_num += 1
        plt.figure(fig_num)
        plt.subplot(1,3,1)
        plot_tie(demodulated, alpha=0.8, label="NO MATCHED FILTER", oversampling=OVERSAMPLING, sync_code=sync_code,
                 pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
        plot_tie(demod_kaiser, alpha=0.8, label="Kaiser BT=%.2f"%BT_COMPOSITE, sync_code=sync_code, pulse_fir=rcos_fir,
                 payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD, oversampling=OVERSAMPLING)
        plot_tie(demod_rcos, alpha=0.8, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- %s"%RECOVERY_METHOD, sync_code=sync_code,
                 pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD, oversampling=OVERSAMPLING)

        plt.subplot(1,3,2)
        plot_jitter_histogram(demodulated, alpha=0.8, label="NO MATCHED FILTER", oversampling=OVERSAMPLING, sync_code=sync_code,
                              pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
        plot_jitter_histogram(demod_kaiser, alpha=0.8, label="Kaiser BT=%.2f"%BT_COMPOSITE, title="- Rx", oversampling=OVERSAMPLING,
                              sync_code=sync_code, pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
        plot_jitter_histogram(demod_rcos, alpha=0.8, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- Rx", oversampling=OVERSAMPLING,
                              sync_code=sync_code, pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
        plt.subplot(1,3,3)
        plot_jitter_spectrum(demodulated, alpha=0.8, label="NO MATCHED FILTER", oversampling=OVERSAMPLING, sync_code=sync_code,
                              pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
        plot_jitter_spectrum(demod_kaiser, alpha=0.8, label="Kaiser BT=%.2f"%BT_COMPOSITE, title="- Rx", oversampling=OVERSAMPLING,
                             sync_code=sync_code, pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)
        plot_jitter_spectrum(demod_rcos, alpha=0.8, label="RCOS BT=%.2f"%BT_COMPOSITE, title="- Rx", oversampling=OVERSAMPLING,
                              sync_code=sync_code, pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, recovery=RECOVERY_METHOD)



    plt.show()

