# Ideal, Natural, & Flat-top -Sampling
## Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
## Software required
 google colab
## Theory
Ideal sampling is a theoretical method in which a continuous-time signal is sampled using an impulse train, producing impulses whose amplitudes equal the signal values at sampling instants, but it is not physically realizable.<BR>
Natural sampling is a practical method where the signal is multiplied by a finite-width pulse train, causing the sampled signal to follow the input signal during each pulse.<BR>
Flat-top sampling (sample-and-hold) holds each sampled value constant over the pulse duration, making it suitable for digital systems, though it introduces aperture distortion that can be reduced using a low-pass filter.

## Program
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, lfilter

# -------------------- Common Parameters --------------------
fs = 1000
T = 1
t = np.arange(0, T, 1/fs)
fm = 5
message_signal = np.sin(2 * np.pi * fm * t)

# -------------------- Impulse Sampling --------------------
fs_impulse = 100
t_imp = np.arange(0, T, 1/fs_impulse)
impulse_samples = np.sin(2 * np.pi * fm * t_imp)
impulse_reconstructed = resample(impulse_samples, len(t))

# -------------------- Natural Sampling --------------------
pulse_rate = 50
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i + pulse_width] = 1

natural_sampled = message_signal * pulse_train

sampled_values = natural_sampled[pulse_train == 1]
sample_times = t[pulse_train == 1]

natural_reconstructed = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    natural_reconstructed[index:index + pulse_width] = sampled_values[i]

# -------------------- Flat-Top Sampling --------------------
pulse_indices = np.arange(0, len(t), int(fs / pulse_rate))
flat_top_signal = np.zeros_like(t)
pulse_width_samples = int(fs / (2 * pulse_rate))

for index in pulse_indices:
    if index < len(message_signal):
        flat_top_signal[index:index + pulse_width_samples] = message_signal[index]

# -------------------- Low-pass Filter --------------------
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, signal)

cutoff = 2 * fm
natural_reconstructed = lowpass_filter(natural_reconstructed, cutoff, fs)
flat_top_reconstructed = lowpass_filter(flat_top_signal, cutoff, fs)

# -------------------- Plotting --------------------
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal)
plt.title("Original Message Signal")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(t_imp, impulse_samples, basefmt=" ")
plt.plot(t, impulse_reconstructed, 'r--')
plt.title("Impulse Sampling and Reconstruction")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, natural_sampled)
plt.plot(t, natural_reconstructed, 'r--')
plt.title("Natural Sampling and Reconstruction")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, flat_top_signal)
plt.plot(t, flat_top_reconstructed, 'r--')
plt.title("Flat-Top Sampling and Reconstruction")
plt.grid(True)

plt.tight_layout()
plt.show()
```
## Output Waveform

<img width="1489" height="1189" alt="download" src="https://github.com/user-attachments/assets/54fcee06-3c83-432e-b426-e4bbbb81f2af" />

## Results
Thus, the python programs for ideal sampling, natural sampling and flat-top sampling has been executed and verified successfully.
