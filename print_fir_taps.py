from lib.gmsk_rx_filter import gmsk_matched_kaiser_rx_filter, gmsk_matched_rcos_rx_filter, kaiser_composite_tx_rx_filter, rcos_composite_tx_rx_filter, gmsk_tx_filter
from lib.sync import make_sync_fir, get_precomputed_codes
import numpy as np

BT_TX = 0.3
BT_COMPOSITE = 1.0
OVERSAMPLING = 8
PULSE_SPAN = 8
IQ_RATE = 83000

def print_taps(fir):
    print("\tTap #\tValue")
    for n, tap in enumerate(fir.td):
        print("\t%d\t%.10f"%(n,tap))

kaiser_fir = kaiser_composite_tx_rx_filter(OVERSAMPLING, PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                           fs=IQ_RATE, norm=True)
rcos_fir = rcos_composite_tx_rx_filter(OVERSAMPLING, PULSE_SPAN, BT_TX, BT_COMPOSITE,
                                       fs=IQ_RATE, norm=True)
gmsk_tx_fir = gmsk_tx_filter(OVERSAMPLING, PULSE_SPAN, BT_TX, fs=IQ_RATE, norm=True)


fir_matched_kaiser = gmsk_matched_kaiser_rx_filter(OVERSAMPLING, PULSE_SPAN, BT_TX,
                                                   BT_COMPOSITE, fs=IQ_RATE)
fir_matched_rcos = gmsk_matched_rcos_rx_filter(OVERSAMPLING, PULSE_SPAN, BT_TX,
                                               BT_COMPOSITE, fs=IQ_RATE)

print("Oversampling=%d, BT Tx=%.2f, BT Composite=%.2f, Pulse span symbols=%d"%(OVERSAMPLING, BT_TX, BT_COMPOSITE, PULSE_SPAN))

print("\nGMSK Tx Pulse FIR")
print_taps(gmsk_tx_fir)

print("\nKaiser impulse FIR, BT=%.2f"%BT_COMPOSITE)
print_taps(kaiser_fir)

print("\nRCOS impulse FIR, BT=%.2f"%BT_COMPOSITE)
print_taps(rcos_fir)

print("\nGMSK-Kaiser Match Rx Filter, BT Tx=%.2f, BT Composite=%.2f"%(BT_TX,BT_COMPOSITE))
print_taps(fir_matched_kaiser)

print("\nGMSK-RCOS Match Rx Filter, BT Tx=%.2f, BT Composite=%.2f"%(BT_TX,BT_COMPOSITE))
print_taps(fir_matched_rcos)

SYNC_CODE_LEN = 24
sync_codes = get_precomputed_codes()
sync_code = sync_codes[SYNC_CODE_LEN]
sync_fir_kaiser = make_sync_fir(sync_code, kaiser_fir, OVERSAMPLING)
np.savetxt("matched_kaiser_bt_1_0.csv", fir_matched_kaiser.td, delimiter=",")
np.savetxt("kaiser_sync_bt_1_0.csv", sync_fir_kaiser.td, delimiter=",")

