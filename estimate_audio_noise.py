""" Based on mathematical models for quantization noise power and noise power from
    bit errors, estimate SNR for for audio subject to quantization and bit errors

    Cole Nielsen 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import math

def sample_bit_error_noise_power(n, ber, sig_range):
    return ber*(sig_range/2.0)**2*(4.0/3.0)*(1.0 - (1.0/4.0)**n)

def quantization_noise_power(n, sig_range):
    return (sig_range**2)/(2**(2.0*n)*12.0)

BER = 1e-3

SAMPLE_BITS = 8
SAMPLING_RATE = 8000
RANGE = 2.0
RMS_THERM_NOISE = 0.0001
FULL_RANGE_SINE_POWER = RANGE**2/8.0

p_one_bit_error = binom.pmf(n=SAMPLE_BITS, k=1, p=BER)

p_mult_bit_errors = 0.0
for k in range(2, SAMPLE_BITS+1):
    p_mult_bit_errors += binom.pmf(n=SAMPLE_BITS, k=k, p=BER)

print("* BER = %E"%BER)
print("* Bits : %d,\tRate : %d Sa/s,\tBit rate : %d kb/s"%(SAMPLE_BITS,SAMPLING_RATE,SAMPLE_BITS*SAMPLING_RATE*0.001))
print("\nP(single bit error) = %E in %d bits"%(p_one_bit_error, SAMPLE_BITS))
print("P(multiple bit error) = %E in %d bits"%(p_mult_bit_errors, SAMPLE_BITS))

print("Mean time between single bit corruption = %f seconds"%((1.0/(p_one_bit_error*SAMPLING_RATE))))
print("Mean time between multi bit corruptions = %f seconds"%((1.0/(p_mult_bit_errors*SAMPLING_RATE))))

ber_noise = sample_bit_error_noise_power(n=SAMPLE_BITS, ber=BER, sig_range=RANGE)
quant_noise = quantization_noise_power(n=SAMPLE_BITS,sig_range=RANGE)

#print("Noise from bit errors = %E"%ber_noise)
#print("Noise power from quantization = %E"%quant_noise)

snr_ber = 10*math.log10(FULL_RANGE_SINE_POWER/ber_noise)
snr_quant = 10*math.log10(FULL_RANGE_SINE_POWER/quant_noise)
snr_thermal = 10*math.log10(FULL_RANGE_SINE_POWER/RMS_THERM_NOISE**2)

total_noise_power = ber_noise + quant_noise + RMS_THERM_NOISE**2

snr_total = 10*math.log10(FULL_RANGE_SINE_POWER/total_noise_power)

print("\n* SNR Analysis:")
print("\tBER induced SNR = %.2f dB"%snr_ber)
print("\tQuantization induced SNR = %.2f dB"%snr_quant)
print("\tThermal induced SNR = %.2f dB"%snr_thermal)
print("\tTotal SNR = %.2f dB"%snr_total)

enob = (snr_total - 1.76)/6.02
print("\tENOB = %.2f bits"%enob)

print("\n\tEffective bit rate = %.2f kb/s"%(enob*SAMPLING_RATE*0.001))
print("\tEffective/actual bit rate = %.2f"%((enob/SAMPLE_BITS)))
