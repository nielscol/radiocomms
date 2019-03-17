BIAS = 0x84   # define the add-in bias for 16 bit samples */
CLIP = 32635

exp_lut = [
    0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
]

def u_law_compress(sample):
    # Get the sample into sign-magnitude. 
    sign = (sample >> 8) & 0x80        #set aside the sign
    if (sign != 0): sample = -sample    #get magnitude
    if (sample > CLIP): sample = CLIP   #clip the magnitude
    # Convert from 16 bit linear to ulaw.
    sample = sample + BIAS
    exponent = exp_lut[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa)
    return ulawbyte

def u_law_expand(ulawbyte):
    exp_lut = [0,132,396,924,1980,4092,8316,16764]
    ulawbyte = ~ulawbyte
    sign = (ulawbyte & 0x80)
    exponent = (ulawbyte >> 4) & 0x07
    mantissa = ulawbyte & 0x0F
    sample = exp_lut[exponent] + (mantissa << (exponent + 3))
    if (sign != 0): sample = -sample
    return sample
import numpy as np
import math
import matplotlib.pyplot as plt

v_ul_comp = np.vectorize(u_law_compress, otypes=[np.uint8])
v_ul_exp = np.vectorize(u_law_expand, otypes=[np.int16])

bits = 16
freq = 100
samples = 1600
fs = 8000

time = np.arange(samples)/float(fs)

amplitude = (2**bits-1)/2.0
signal = np.array(np.rint(amplitude + amplitude*np.sin(2*math.pi*freq*time))-2**(bits-1),dtype=np.int16)
plt.plot(signal)

comp = v_ul_comp(signal)
recov = v_ul_exp(comp)

plt.plot(recov)
plt.show()
