""" Methods for clock recovery, jitter/TIE and zero crossing calculation
    Cole Nielsen 2019
"""

import numpy as np
from lib.tools import *
from lib.sync import make_sync_fir, detect_sync
from lib.transforms import fir_correlate
from copy import copy
import math


def frame_sync_recovery(signal, sync_code, pulse_fir, payload_len, oversampling, sync_pos="center"):
    sync_fir = make_sync_fir(sync_code, pulse_fir, oversampling)
    correlation = fir_correlate(signal, sync_fir, oversampling)
    peak_indices, peak_values = detect_sync(correlation, sync_code, payload_len, oversampling)
    crossings = []
    n_samples = len(correlation.td)
    s_len = int(len(sync_code)*oversampling)
    p_len = int(payload_len*oversampling)
    f_len = s_len + p_len
    c_offset = int(p_len/2.0)
    edge_center_delta = int(oversampling/2)
    for sync_index in peak_indices:
        if sync_pos == "center":
            lower = sync_index - c_offset - edge_center_delta
            lower = 0 if lower < 0 else lower
            upper = sync_index + s_len + c_offset - edge_center_delta
            upper = n_samples if upper > n_samples else upper
            for n in range(sync_index-c_offset, sync_index+s_len+c_offset, oversampling):
                crossing = n - edge_center_delta
                if crossing >= lower and crossing < upper:
                    crossings.append(crossing)
        elif sync_pos == "start":
            lower = sync_index
            upper = sync_index + f_len
            upper = n_samples if upper > n_samples else upper
            for n in range(sync_index, sync_index+f_len, oversampling):
                crossing = n - edge_center_delta
                if crossing >= lower and crossing < upper:
                    crossings.append(crossing)
    diff = np.diff(crossings)
    print(list(set(diff)))
    print(np.argmin(diff-16))
    return np.array(crossings)


def constant_f_recovery(td, ui_samples, est_const_f=True):
    """ Recovers a constant frequency clock
    """
    # determine zero crossings and uis/samples between zero crossings 
    crossings = crossing_times(td)
    deltas_uis, deltas_samples = get_crossing_deltas(crossings, ui_samples)
    # clock recover
    if est_const_f:
        clk_period = get_clk_period_in_samples(deltas_uis, deltas_samples)
    else:
        clk_period = ui_samples
    clk_phase = get_clk_phase(crossings, deltas_uis, clk_period, span=len(crossings))
    clk_crossings = constant_f_clk_crossings(clk_period, clk_phase, uis=sum(deltas_uis))

    return clk_crossings, clk_period, clk_phase


def crossing_times(td):
    """ Determines time of waveform zero crossings as a list of fractional index values
    """
    crossings = np.where(np.diff(np.sign(td)))[0] # returns index of values before or at crossings
    n_crossings = len(crossings)
    frac_crossings = []
    td_len = len(td)
    for cross_n, td_n in enumerate(crossings):
        if td_n+1 < td_len and td[td_n+1] != 0.0:
            frac_crossings.append(td_n-(td[td_n]/float(td[td_n+1]-td[td_n])))
        elif td_n > 0 and td[td_n] == 0.0 and td[td_n-1] != 0:
            frac_crossings.append(float(td_n))
        elif (td_n+1 < td_len and cross_n+1 < n_crossings and td[td_n] != 0.0
              and td[td_n+1] == 0 and td_n+1 != crossings[cross_n+1]):
            frac_crossings.append(float(td_n+1))
    return frac_crossings


def get_crossing_deltas(crossings, ui_samples):
    '''Given a list of zero crossing times, calculates the time in UIs and samples between each
    sequential pair of crossings, requires initial estimate of ui length

    '''
    crossing_deltas_samples = []
    crossing_deltas_uis = []

    for n in range(len(crossings)-1):
        crossing_deltas_uis.append(round((crossings[n+1]-crossings[n])/ui_samples))
        crossing_deltas_samples.append(crossings[n+1]-crossings[n])

    return crossing_deltas_uis, crossing_deltas_samples


def get_clk_period_in_samples(crossing_deltas_uis, crossing_deltas_samples):
    """ Estimates clk frequency of data, needed for accurate recovery with the phase tracking CDR method
    """
    n_samples = 0
    n_uis = 0
    for n in range(len(crossing_deltas_samples)):
        n_samples += crossing_deltas_samples[n]
        n_uis += crossing_deltas_uis[n]

    return n_samples/n_uis


def get_clk_phase(crossings, crossing_deltas_uis, ui_samples, span):
    """ Makes an initial estimate of the clock phase (in samples)
    """
    counts = 0
    offsets = []
    for n in range(span):
        if n == 0:
            offsets.append(crossings[0])
        else:
            counts += crossing_deltas_uis[n-1]
            offsets.append(crossings[n] - counts*ui_samples)
    return np.mean(offsets)


def constant_f_clk_crossings(clk_period, clk_phase, uis):
    """ Takes clk period and phase and generates a list of clock crossings
    """
    return np.arange(uis)*clk_period+clk_phase


@timer
def get_tie(signal, bits_per_sym = 1, interp_factor=10, interp_span=128,
            remove_ends=100, recovery="constant_f", est_const_f=True,
            sync_code=None, pulse_fir=None, payload_len= None,
            sync_pos="center", oversampling=None):
    _signal = copy(signal)
    _signal.td = _signal.td[remove_ends:]
    _signal.td = _signal.td[:-remove_ends]
    td = _signal.td
    interpolated = sinx_x_interp(td, interp_factor, interp_span)
    ui_samples = interp_factor*signal.fs*bits_per_sym/float(signal.bitrate)
    if recovery == "constant_f":
        clk_crossings, clk_period, clk_phase = constant_f_recovery(interpolated, ui_samples, est_const_f)
    elif recovery == "pll_second_order":
        clk_crossings, clk_period, clk_phase = pll_so_recovery(interpolated, ui_samples)
        #cdr_lock_index = int(settle_tcs*4.0/(damping*f3db)/(1/sampling_rate))
    elif recovery == "frame_sync":
        clk_crossings = frame_sync_recovery(_signal, sync_code, pulse_fir, payload_len,
                                        oversampling, sync_pos="center")
        clk_crossings *= interp_factor
        ui_samples = oversampling*interp_factor
    data_crossings = crossing_times(interpolated)
    return compute_tie_trend(clk_crossings, data_crossings, ui_samples)


def compute_tie_trend(clk_crossings, data_crossings, ui_samples):
    deltas_uis, deltas_samples = get_crossing_deltas(data_crossings, ui_samples)
    tie_trend_at_crossings = []
    clk_cycle = 0
    for n, ui_delta in enumerate(deltas_uis):
        if clk_cycle >= len(clk_crossings):
            break
        tie_trend_at_crossings.append(data_crossings[n] - clk_crossings[int(clk_cycle)])
        clk_cycle += ui_delta

    tie_trend = []
    for n, curr_tie in enumerate(tie_trend_at_crossings[:-1]):
        #if n == len(tie_trend_at_crossings)-1:
        #    break
        step = (tie_trend_at_crossings[n+1] - curr_tie)/deltas_uis[n]
        for m in range(int(deltas_uis[n])):
            tie_trend.append(curr_tie + m*step)

    return np.array(tie_trend)/float(ui_samples)

