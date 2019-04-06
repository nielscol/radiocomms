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

    TARGET_BER = 1e-6
    BITS_PER_ITER = 3e5
    OVER_FACTOR = 10
    ITERATIONS = int(round(10.0/(TARGET_BER*BITS_PER_ITER)))

    N_BITS = 300000                   # number of data bits in simulation
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

    SIMULATE_INTERFERER = False     # Simulate a nearby interfering signal
    INTERFERER_LO = 3.0*BW          # Interferer carrier
    INTERFERER_BITRATE = BITRATE    # Interferer bitrate
    INTERFERER_N_BITS = N_BITS      # Interferer data bits
    INTERFERER_FRAME_PAYLOAD = FRAME_PAYLOAD    # Frame data payload len 
    INTERFERER_MODULATION = "MSK"               # {"MSK", "GMSK"}
    INTERFERER_BT = BT_TX                       # for GMSK only
    INTERFERER_FIR_SPAN = TX_FIR_SPAN           # for GMSK Tx FIR pulse shaping span
    INTERFERER_OVERSAMPLING = OVERSAMPLING      # Oversampling rate
    INTERFERER_CORREL_SYNC = False              # Should the sync be the same as the GMSK?

    AGWN_RMS = 0.4                 # 0.00707 -> 40 dB SNR, 0.0707 -> 20 dB, 0.223 -> 10 dB, 0.4 -> 6 dB
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
    SAMPLE_LINES = True             # Draw lines corresponding to where samples occur
    POOLS = 8                       # For using multiple processes to compute eyes. POOLS = # logical processors on your computer

    RECOVERY_METHOD = "frame_sync"  # {"frame_sync", "constant_f", "edge"}, use "frame_sync" if data if framed, else "constant_f"

    PLOT_TX = False                  # Plot Tx-related plots (Constellation, Phase/magnitude of IQ, spectrum)
    PLOT_RX = False                  # Plot Rx-related plots (Constellation, IQ Phase/magnitude, raw demodulated data)
    PLOT_RX_EYES = False             # Plot Rx eye diagrams
    PLOT_RX_JITTER = False           # Plot jitter plots (Total interval error, jitter distribution, jitter spectrum)
    PLOT_RX_FILTER = False          # Plot Rx filter taps and responses
    PLOT_RX_DATA_PSD = False        # Plot Rx data PSD
    PLOT_SYNC = False                # Plot Sync-related plots (Correlation of sync to data, histogram of aforementioned and detected syncs)

    #
    # Begin simulation
    #
    kaiser_tie = np.array([], dtype=float)
    rcos_tie = np.array([], dtype=float)

    kaiser_fir = kaiser_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                               fs=IQ_RATE, norm=True)
    rcos_fir = rcos_composite_tx_rx_filter(OVERSAMPLING, SYNC_PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                           fs=IQ_RATE, norm=True)
    for N in range(ITERATIONS):
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

        #
        # Simulate channel (AWGN + fading)
        #

        # Simulate AWGN
        rf = add_noise(rf, rms=AGWN_RMS)

        rf_rms = measure_rms(rf)
        print("\n* Signal SNR = %.2f dB"%(20*np.log10(rf_rms/AGWN_RMS)))


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

        k_tie = get_tie(demod_kaiser, recovery=RECOVERY_METHOD, sync_code=sync_code, pulse_fir=kaiser_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, oversampling=OVERSAMPLING)
        k_tie -= np.round(k_tie)
        kaiser_tie = np.concatenate([kaiser_tie, k_tie])
        #r_tie = get_tie(demod_rcos, recovery=RECOVERY_METHOD, sync_code=sync_code, pulse_fir=rcos_fir, payload_len=FRAME_PAYLOAD, sync_pos=SYNC_POS, oversampling=OVERSAMPLING)
        #r_tie -= np.round(r_tie)
        #rcos_tie = np.concatenate([rcos_tie, r_tie])


    from lib.clock_recovery import fit_n_weighted_gaussians, get_tie, v_n_gaussians_model

    hist = np.histogram(kaiser_tie, bins=1000)
    hist_y = np.array(hist[0], dtype=float)
    hist_y[hist_y<10] = 0
    bins_x = np.array(hist[1][:-1]) +0.5*np.diff(np.array(hist[1]))
    cdf = np.cumsum(hist_y)
    cdf *= 1/np.amax(cdf)
    cdf = np.ma.masked_where(cdf==0.0, cdf)
    #ccdf = np.cumsum(hist_y[::-1])
    #ccdf *= 1/np.amax(ccdf[::-1])
    ccdf = 1.0 - cdf
    ccdf = np.ma.masked_where(ccdf==0.0, ccdf)
    plt.plot(bins_x, np.log10(ccdf), label="Kaiser")
    plt.plot(bins_x+1, np.log10(cdf), label="Kaiser")
    plt.xlim((0,1))
    plt.title("BER Bathtub - Kaiser Pulse Shape\n6 dB CNR, MSK Int. at fc-2BW, 40dB Fading.")

    import pickle
    ser = pickle.dumps(hist)
    f = open("bathtub_hist.dat", "wb")
    f.write(ser)
    f.close()

    #hist = np.histogram(rcos_tie, bins=1000)
    #hist_y = np.array(hist[0], dtype=float)
    #hist_y[hist_y<10] = 0
    #bins_x = np.array(hist[1][:-1]) +0.5*np.diff(np.array(hist[1]))
    #cdf = np.cumsum(hist_y)
    #cdf *= 1/np.amax(cdf)
    #cdf = np.ma.masked_where(cdf==0.0, cdf)
    #ccdf = np.cumsum(hist_y[::-1])
    #ccdf *= 1/np.amax(ccdf[::-1])
    #ccdf = 1.0 - cdf
    #ccdf = np.ma.masked_where(ccdf==0.0, ccdf)
    #plt.plot(bins_x, np.log10(ccdf), label="RCOS")
    #plt.plot(bins_x+1, np.log10(cdf), label="RCOS")
    #plt.xlim((0,1))
    plt.show()

