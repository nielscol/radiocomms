from _signal import *
from transforms import *
from analysis import *
from plot import *


settings = dict(verbose=True)

audio = wav_to_signal('./speech.wav', **settings)
# audio = remove_dc(audio, **settings)

in_band_bw = 2000
tone_test = dict(tone_f = 1000, tone_bw = 10)

measure_in_band_sfdr(audio, in_band_bw, **settings)
measure_in_band_snr(audio, in_band_bw, **settings)
measure_sndr(audio, **tone_test)
measure_sfdr(audio, **tone_test)
audio1 = filter_and_downsample(audio, 2, ftype="fir")
audio2 = fft_downsample(audio, n=2)
audio3 = no_filter_downsample(audio, n=2)
#plot_fd(audio1, log=True, **settings)
#plot_fd(audio2, log=True, **settings)
#plot_fd(audio3, log=True, **settings)
plt.legend()
plot_td(audio, **settings)
plot_td(audio3, **settings)
plt.show()
