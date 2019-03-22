""" Methods for modulation and demodulation, and up/down conversion
"""

import numpy as np
from math import pi, sqrt, log
from lib._signal import *
from lib.tools import *
from copy import copy
from scipy.special import erfc

SQRT2 = sqrt(2.0)
SQRTLN2 = sqrt(log(2.0))

def generate_msk_baseband(message, oversampling, name="", binary_message=True, autocompute_fd=False, verbose=True, *args, **kwargs):
    """ Generates I/Q signals according to MSK modulation.
        Args:
            message: bit stream if binary_message=True, or floaring point valued signal [-1,1] otherwise
            oversampling: rate to oversample (compared to message fs)
    """
    _message = message.td.astype(float)
    if binary_message:
        # map binary {0,1} onto {-1, 1}
        _message[_message<=0] = -1
        _message[_message>0] = 1
    # Upsample message to and convolve with rectangular pulse shape
    tx_fir = np.ones(oversampling)
    upsampled = np.zeros(len(message.td)*oversampling)
    upsampled[np.arange(len(message.td))*oversampling] = _message
    upsampled = np.convolve(upsampled, tx_fir, mode="full")
    # generate MSK phase signal from baseband message
    msk_phase = (pi/2.0)*np.cumsum(upsampled)/float(oversampling)
    # Generate I/Q baseband components
    i = np.cos(msk_phase)
    q = np.sin(msk_phase)
    # make signal objects
    sig_i = make_signal(td=i, fs=message.fs*oversampling, bitrate=0.5*message.bitrate, name=name+"_msk_i_component", autocompute_fd=autocompute_fd, verbose=False)
    sig_q = make_signal(td=q, fs=message.fs*oversampling, bitrate=0.5*message.bitrate, name=name+"_msk_i_component", autocompute_fd=autocompute_fd, verbose=False)

    return sig_i, sig_q

def gmsk_pulse(t, bt, tbit):
    return (1/(2.0*tbit))*(q(2*pi*bt*(t-0.5*tbit)/SQRTLN2)-q(2*pi*bt*(t+0.5*tbit)/SQRTLN2))
v_gmsk_pulse = np.vectorize(gmsk_pulse, otypes=[float])

def generate_gmsk_baseband(message, oversampling, bt, pulse_span, name="", binary_message=True, autocompute_fd=False, verbose=True, *args, **kwargs):
    """ Generates I/Q signals according to GMSK modulation.
        Args:
            message: bit stream if binary_message=True, or floaring point valued signal [-1,1] otherwise
            oversampling: rate to oversample (compared to message fs)
            bt: rolloff factor for GMSK pulse response
            pulse_span: number of bits FIR pulse response should cover
            binary_message: if True, input assumed to be {0,1} and is remapped as needed by the modulator to {-1,1}
    """
    _message = message.td.astype(float)
    if binary_message:
        # map binary {0,1} onto {-1, 1}
        _message[_message<=0] = -1
        _message[_message>0] = 1
    # Make GMSK pulse shape
    fir_samples = int(oversampling*pulse_span+1)
    t = (np.arange(fir_samples)/float(oversampling) - 0.5*pulse_span)
    TBIT = 1.0
    tx_fir = v_gmsk_pulse(t, bt, TBIT)
    # Normalize pulse shape so integrated value is pi/2
    tx_fir *= pi/(2.0*sum(tx_fir))
    # upsample message and apply pulse shape through convolution
    upsampled = np.zeros(len(message.td)*oversampling)
    upsampled[np.arange(len(message.td))*oversampling] = _message
    upsampled = np.convolve(upsampled, tx_fir, mode="full")
    # generate GMSK phase signal from baseband message
    gmsk_phase = np.cumsum(upsampled)
    # Generate I/Q baseband components
    i = np.cos(gmsk_phase)
    q = np.sin(gmsk_phase)
    # make signal objects
    sig_i = make_signal(td=i, fs=message.fs*oversampling, bitrate=0.5*message.bitrate, name=name+"_msk_i_component", autocompute_fd=autocompute_fd, verbose=False)
    sig_q = make_signal(td=q, fs=message.fs*oversampling, bitrate=0.5*message.bitrate, name=name+"_msk_i_component", autocompute_fd=autocompute_fd, verbose=False)

    return sig_i, sig_q

def upconvert_baseband(carrier_f, i=None, q=None, amplitude=1.0, auto_upsample=True, auto_sa_cyc=20, manual_upsample_factor=1, interp_span=128, name="", autocompute_fd=False, verbose=True, *args, **kwargs):
    """ Mixes I/Q basebands components to carrier frequency. By default tries to upsample data to keep
        approximately auto_sa_cyc samples per carrier cycle. Will only upsample by integer factors, so the
        nearness of the number of samples per carrier cycle to auto_sa_cyc depends on this.
    """
    if i is None and q is None:
        raise Exception("No I/Q components passed to method")
    elif i is None:
        _i = np.zeros(len(q.td))
        _q = q.td
        fs = q.fs
        bitrate = q.bitrate
    elif q is None:
        _i = i.td
        _q = np.zeros(len(i.td))
        fs = i.fs
        bitrate = i.bitrate
    else:
        _i = i.td
        _q = q.td
        fs = i.fs
        bitrate = i.bitrate + q.bitrate
    if auto_upsample:
        interp_factor = int(round((auto_sa_cyc*carrier_f)/float(fs)))
    else:
        interp_factor = manual_upsample_factor
    if interp_factor != 1:
        if verbose:
            print("\nUpsampling %dx via sin(x)/x interpolation with I/Q data so carrier mantains sufficient samples/cycle. May be slow..."%interp_factor)
        _i = sinx_x_interp(_i, interp_factor, interp_span)
        _q = sinx_x_interp(_q, interp_factor, interp_span)
    time = np.arange(len(_i))/float(fs*interp_factor)
    rf = amplitude*(np.cos(2*pi*carrier_f*time)*_i - np.sin(2*pi*carrier_f*time)*_q)

    return make_signal(td=rf, fs=fs*interp_factor, bitrate=bitrate, name=name+"_upconverted", autocompute_fd=autocompute_fd, verbose=False)

def downconvert_rf(carrier_f, rf, name="", autocompute_fd=False, verbose=True, *args, **kwargs):
    time = np.arange(len(rf.td))/float(rf.fs)
    i = np.cos(2*pi*carrier_f*time)*rf.td
    q = np.sin(2*pi*carrier_f*time)*rf.td
    #plt.plot(rf.td)
    #plt.plot(np.cos(2*pi*carrier_f*time))
    #plt.plot(i)
    #plt.show()
    i_sig = make_signal(td=i, fs=rf.fs, bitrate=rf.bitrate/2, name=name+"_downconverted", autocompute_fd=autocompute_fd, verbose=False)
    q_sig = make_signal(td=q, fs=rf.fs, bitrate=rf.bitrate/2, name=name+"_downconverted", autocompute_fd=autocompute_fd, verbose=False)

    return i_sig, q_sig

def demodulate_gmsk(i, q, oversampling, name="", autocompute_fd=False, verbose=True, *args, **kwargs):
    demod_td = (2.0/pi)*oversampling*np.diff(np.unwrap(np.arctan2(q.td,i.td)))
    return make_signal(td=demod_td, fs=i.fs, bitrate=i.bitrate+q.bitrate, name=name+"_gmsk_demodulated", autocompute_fd=autocompute_fd, verbose=False)

def rx_filter(signal, fir_taps, oversampling):
    # Make GMSK pulse shape
    filt_td = np.convolve(signal.td, fir_taps, mode="full")/float(oversampling)
    return make_signal(td=filt_td, fs=signal.fs, bitrate=signal.bitrate, name=signal.name+"_matched_filter", autocompute_fd=False, verbose=False)


def gaussian_to_raised_cosine(bt, oversampling, pulse_span):
    # Make GMSK pulse shape
    pulse_len = int(oversampling*pulse_span)
    t = (np.arange(pulse_len)-pulse_len/2)/float(oversampling)
    gmsk_pulse_shape = 2*v_gmsk_pulse(t, bt, 1.0)
    # add an offset to pulse so sum of samples = 1.0
    gmsk_pulse_error = np.sum(gmsk_pulse_shape)-oversampling
    gmsk_pulse_shape -= gmsk_pulse_error/float(pulse_len)
    rc_pulse_shape = v_raised_cos(t, 1.0, 1.0)
    gmsk_pulse_shape += np.random.normal(0,1e-9,pulse_len)
    fir = np.fft.ifft(np.fft.fft(rc_pulse_shape)/np.fft.fft(gmsk_pulse_shape))
    fir = np.fft.fftshift(fir)
    # add an offset to pulse so sum of samples = 1.0
    #pulse_error = np.sum(pulse_shape)-oversampling
    #pulse_shape -= pulse_error/float(pulse_len)
    return fir
