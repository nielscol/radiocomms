""" Code to test functionality of u-law algorithm after translation from c to Python
"""

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

a = (np.arange(2**10, dtype=np.int64)-2**9)*2**6
b = v_ul_comp(a)
c = v_ul_exp(b)
s = ""
for n, val in enumerate(a):
    print("%d --> %d"%(val,c[n]))
    s += ("%d --> %d\n"%(val,c[n]))

f = open("example_u_law_in_out.txt", "w")
f.write(s)
f.close()

time = np.arange(samples)/float(fs)

amplitude = (2**bits-1)/2.0
signal = np.array(np.rint(amplitude + amplitude*np.sin(2*math.pi*freq*time))-2**(bits-1),dtype=np.int16)

comp = v_ul_comp(signal)

plt.figure(1)
plt.plot(comp)
recov = v_ul_exp(comp)

plt.figure(2)
plt.plot(signal)
plt.plot(recov)
#plt.show()
