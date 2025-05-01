import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# 1. Generate time array
t = np.linspace(0, 1, 1000)  # 1 second, 1000 samples

# 2. Generate a triangular frequency waveform: ranges from 0 to 5 Hz and back
from scipy.signal import sawtooth
f_triangle = 5 * (1 - np.abs(sawtooth(2 * np.pi * 1 * t, width=0.5)))  # peak at 5 Hz

# 3A. Total phase (scalar) using np.trapz
total_phase = 2 * np.pi * np.trapz(f_triangle, t)

# 3B. Instantaneous phase (array) using cumulative_trapezoid
phase = 2 * np.pi * cumulative_trapezoid(f_triangle, t, initial=0)

# 4. Plotting
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, f_triangle, label='Triangular Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.title('Triangular Frequency Waveform')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, phase, label='Instantaneous Phase (radians)', color='orange')
plt.axhline(total_phase, color='gray', linestyle='--', label=f'Total Phase = {total_phase:.2f} rad')
plt.ylabel('Phase (rad)')
plt.title('Phase: Cumulative vs Total')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, np.sin(phase), label='Reconstructed Sine Wave', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal Synthesized from Instantaneous Phase')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
