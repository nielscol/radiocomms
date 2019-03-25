import matplotlib.pyplot as plt
import numpy as np
from lib.sync import get_precomputed_codes, make_sync_fir, code_string
from lib.gmsk_rx_filter import gmsk_kaiser_composite_rx_tx_response

PULSE_SPAN = 16
OVERSAMPLING = 16
BT_TX = 0.3
BT_COMPOSITE = 1.0
SYNC_CODE_LEN = 24 # precomputed codes are up to N=24

pulse_fir = gmsk_kaiser_composite_rx_tx_response(OVERSAMPLING, PULSE_SPAN, BT_TX, BT_COMPOSITE)
plt.plot(pulse_fir)
sync_codes = get_precomputed_codes()
sync_code = sync_codes[SYNC_CODE_LEN]
print(sync_code)
print(code_string(sync_code))
sync_fir = make_sync_fir(sync_code, pulse_fir, OVERSAMPLING)

autocorrel = np.correlate(sync_fir, sync_fir, mode="full")

plt.plot(sync_fir)
plt.plot(autocorrel)
plt.show()
