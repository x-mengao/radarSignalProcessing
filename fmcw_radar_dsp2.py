# ##############################        
# This script is simulating a FMCW DARDA system from signal generation to range/doppler processing
# For educational purposes only
# Copyright (C) 2025  Xiaomeng Gao
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Reference: https://wirelesspi.com/fmcw-radar-part-1-ranging/
# Key Takeaways:
# 1. Frequency change linearly over time, the phase, which is the integral of frequency, is changing quadratically over time.
# 2. In time domain, the signal seems to be compressed to the right. 
# 3. The chirps are composed of up-chirp and down-chirp, aiming to resolve the range and velocity of the target.
# ##############################   
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from dsp_utils import compute_fft
# ##############################
# FMCW Parameters
# ##############################
# This class contains the parameters for the FMCW system
class FMCWParams:
    def __init__(self):
        self.c = 3e8                                    # Speed of light in m/s
        self.fs = 200e6                                 # Sampling frequency in Hz
        self.f_start = 77e9                                # Start frequency in Hz
        self.f_bw = 50e6                                # Bandwidth in Hz, this is the only band of interest
        self.T_chirp = 100e-6                           # Chirp duration in seconds
        self.slope = self.f_bw / self.T_chirp           # Slope of the chirp signal
        self.R_target = 200                             # Target range in meters
        self.v_target = 150                              # Target velocity in m/s

def subplot_label_tool(ax, title, xlabel, ylabel):
    """
    Set the title and labels for the subplot
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
# ##############################
# Chirp Transmit Signal Generation
# ##############################
def generate_chirp_signal(params, do_plot=True):
    """
    Generate a linear frequency modulated (LFM) chirp signal
    """
    t = np.linspace(0, params.T_chirp * 2, int(params.fs * params.T_chirp * 2), endpoint=False) # linspace is inclusive of the endpoint
    slope = params.f_bw / params.T_chirp
    period_chirp = params.T_chirp * 2
    f_triangle = signal.sawtooth(2 * np.pi * 1 / period_chirp * t, width=0.5) + 1 # Frequency profile of the triangle signal
    f_triangle = f_triangle * params.f_bw / 2 + params.f_start # Scale the triangle signal to the desired frequency range
    phase_triangle = 2 * np.pi * cumulative_trapezoid(f_triangle, t, initial=0) # Integrate the frequency to get the phase
    # f_upchirp = params.f_start + params.slope * t                   # Frequency profile of the chirp signal
    # phase_upchirp = 2 * np.pi * (params.f_start * t + 0.5 * slope * t**2) # Phase of the chirp signal
    tx_signal = np.exp(1j * phase_triangle)                          # Complex representation of TX signal

    if do_plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(t, f_triangle / 1e9)    # Plot the chirp frequency in time domain
        ax[1].plot(t, np.real(tx_signal))
        subplot_label_tool(ax[0], 'TX signal frequency profile (demo-only)', 'Time (s)', 'Frequency (GHz)')
        subplot_label_tool(ax[1], 'TX signal (real part) in time domain', 'Time (s)', 'Amplitude')
        plt.tight_layout()
        plt.show()
    return tx_signal, t

# ##############################
# Chirp Return Signal Generation
# ##############################
def generate_return_signal(params, t, do_plot=True):
    """
    Generate the return signal from a target
    """
    tau = 2 * params.R_target / params.c                            # Time delay due to roundtrip
    beat_frequency_dist = params.slope * tau                             # Beat frequency due to range
    wavelength = params.c / params.f_start
    beat_frequency_vel = 2 * params.v_target / wavelength
    print("Theoretical beat frequency for distance fd:", beat_frequency_dist)
    print("Theoretical beat frequency for velocity fv:", beat_frequency_vel)
    # f_doppler = 2 * params.v_target * params.f_start / params.c # Doppler frequency shift
    t_delay = t - tau
    period_chirp = params.T_chirp * 2

    f_triangle = signal.sawtooth(2 * np.pi * 1 / period_chirp * t, width=0.5) + 1   # Frequency profile of the triangle signal
    f_triangle = f_triangle * params.f_bw / 2 + params.f_start                      # Scale the triangle signal to the desired frequency range

    f_triangle_delay = signal.sawtooth(2 * np.pi * 1 / period_chirp * t_delay, width=0.5) + 1       # Frequency profile of the triangle signal
    f_triangle_delay = f_triangle_delay * params.f_bw / 2 + params.f_start + beat_frequency_vel     # Scale and shift the triangle signal to the desired frequency range
    phase_triangle_delay = 2 * np.pi * cumulative_trapezoid(f_triangle_delay, t_delay, initial=0)   # Integrate the frequency to get the phase
    rx_signal = np.exp(1j * phase_triangle_delay)
    if do_plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(t, f_triangle / 1e9, label='TX')          # Plot the chirp frequency in time domain
        ax[0].plot(t + tau, f_triangle_delay / 1e9, color='red',label='RX')     # Plot the chirp frequency in time domain
        ax[1].plot(t, np.real(tx_signal), label='TX')
        ax[1].plot(t, np.real(rx_signal), color='red', label='RX')
        subplot_label_tool(ax[0], 'RX signal frequency profile (demo-only)', 'Time (s)', 'Frequency (GHz)')
        subplot_label_tool(ax[1], 'RX signal (real part) in time domain', 'Time (s)', 'Amplitude')
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.show()
    return rx_signal

# ##############################
# Mixing the Received Signal with the Transmitted Signal
# a.k.a. IQ Demodulation (Complex Mixing)
# Multiply the received signal with the conjugate of the *basebanded* transmitted chirp
# The basebanded transmitted chirp is just np.exp(1j * 2 * np.pi * (0.5 * slope * t**2))
# ##############################
def mix_signals(tx_signal, rx_signal, params, do_plot=True):
    """
    Mix the received signal with the transmitted signal
    Key takeaway: use conjugate in mixing. If conjugate is not used, the result is wrong. 
    If conjugate TX, beat frequency is negative. Visually, it's RX - TX
    If conjugate RX, beat frequency is positive. Visually, it's TX - RX, hence positive beat tone.
    Or, simply do np.abs for range calculation, it has to be positive value. 
    """
    # Multiply the received signal with the conjugate of the transmitted signal
    
    # mixed_signal = rx_signal * np.conjugate(tx_signal) 
    mixed_signal = tx_signal * np.conjugate(rx_signal)
    
    if do_plot:
        fig, ax = plt.subplots(3, 2, figsize=(10, 6))
        # Plot the mixed signal in time domain
        ax[0,0].plot(t, np.real(mixed_signal))
        ax[1,0].plot(t, np.imag(mixed_signal))
        ax[2,0].plot(t, np.real(mixed_signal))
        ax[2,0].plot(t, np.imag(mixed_signal))
        subplot_label_tool(ax[0,0], 'Mixed signal (real part)', 'Time (s)', 'Amplitude')
        subplot_label_tool(ax[1,0], 'Mixed signal (imaginary part)', 'Time (s)', 'Amplitude')
        subplot_label_tool(ax[2,0], 'Mixed signal overlapped', 'Time (s)', 'Amplitude')

        # Compute and plot the FFT of the real, imaginary then mixed signal, then plot
        frequencies_r, X_magnitude_r, peaks_r = compute_fft(np.real(mixed_signal), params.fs)
        frequencies_i, X_magnitude_i, peaks_i = compute_fft(np.imag(mixed_signal), params.fs)
        frequencies_c, X_magnitude_c, peaks_c = compute_fft(mixed_signal, params.fs)
        fb_up = np.min(np.abs(frequencies_c[peaks_c]))  # Some simplifications / assumptions are made assuming fb_dn is bigger than fb_up.
        fb_dn = np.max(np.abs(frequencies_c[peaks_c]))  # Both need to consider absoluate value as beat tone
        estimated_distance = (fb_dn + fb_up) / 4 * (params.c / params.slope)
        estimated_velocity = (fb_dn - fb_up) / 4 * (params.c / params.f_start)

        ax[0,1].plot(frequencies_r, X_magnitude_r)
        ax[0,1].plot(frequencies_r[peaks_r], X_magnitude_r[peaks_r], "x")
        ax[1,1].plot(frequencies_i, X_magnitude_i)
        ax[1,1].plot(frequencies_i[peaks_i], X_magnitude_i[peaks_i], "x")
        ax[2,1].plot(frequencies_c, X_magnitude_c)
        ax[2,1].plot(frequencies_c[peaks_c], X_magnitude_c[peaks_c], "x")
        subplot_label_tool(ax[0,1], 'FFT of mixed signal (real part)', 'Frequency (Hz)', 'Magnitude')
        subplot_label_tool(ax[1,1], 'FFT of mixed signal (imag part)', 'Frequency (Hz)', 'Magnitude')
        subplot_label_tool(ax[2,1], 'FFT of mixed signal', 'Frequency (Hz)', 'Magnitude')
        ax[0,1].set_xlim(-10e6, 10e6)
        ax[1,1].set_xlim(-10e6, 10e6)
        ax[2,1].set_xlim(-10e6, 10e6)
        print("FFT bin width:", params.fs/len(mixed_signal), "Hz")
        print("FFT of mixed signal (real part):", frequencies_r[peaks_r], "Hz")
        print("FFT of mixed signal (imaginary part):", frequencies_i[peaks_i], "Hz")
        print(f"FFT of mixed signal: {frequencies_c[peaks_c]} Hz")
        print(f"Estimated distance is {estimated_distance:.2f} m| Estimated speed is {estimated_velocity:.2f} m/s" ) # :.2f prints out 2 digits after decimal
        plt.tight_layout()
        plt.show()
    return mixed_signal

tx_signal, t = generate_chirp_signal(FMCWParams(), do_plot=False)
rx_signal = generate_return_signal(FMCWParams(), t, do_plot=True)
mixed_signal = mix_signals(tx_signal, rx_signal, FMCWParams(), do_plot=True)

"""
End of FMCW chirp signal chain modeling.

Key Insight:
When the target starts moving, the beat frequency is further shifted by the Doppler frequency shift.
The resulting beat frequency becomes the sum and difference of the range and Doppler frequencies. 

Takeaways:
1. Seems f_start = 0 or 77GHz won't affect phase simulation
2. Need to make correct estimation on upchirp and downchirp beat tones (sign, find right one)
3. Most importantly, use correction equations to calculate range and velocity

            fb_dn + fb_up       c
    range = --------------- * ------
                  4           slope
    
                fb_dn - fb_up   
    velocity = -------------- * wavelength
                      4

For further reading, refer to:
https://wirelesspi.com/fmcw-radar-part-2-velocity-angle-and-radar-data-cube/
"""
