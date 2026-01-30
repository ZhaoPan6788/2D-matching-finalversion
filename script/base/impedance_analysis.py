#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: impedance_analysis.py (2D Single Frequency - L-Match Topology)
Description:
    Topology: Source -> Rs -> [ Shunt Cm1 || (Series Cm2 + Lm + Rm + Load) ]
"""

import numpy as np
from scipy.fft import fft, fftfreq

def cal_impedance_capacity(Cm1, f):
    if Cm1 <= 1e-15: return -1e15j # Open circuit protection
    w = 2.0 * np.pi * f
    return -1j / (w * Cm1)

def cal_impedance_capacity(Cm2, f):
    if Cm2 <= 1e-15: return -1e15j # Open circuit protection
    w = 2.0 * np.pi * f
    return -1j / (w * Cm2)

def cal_impedance_inductor(Lm, f):
    w = 2.0 * np.pi * f
    return 1j * w * Lm

def get_fftw(data_1d, dt, tol=0.001):
    """ Standard FFT to extract frequency components """
    N = data_1d.size
    if N == 0: return np.array([]), np.array([]), np.array([])
    
    fftw_result = fft(data_1d)
    freq_all = fftfreq(N, dt)

    amplitude_spectrum = np.abs(fftw_result) / N * 2
    amplitude_spectrum[0] = amplitude_spectrum[0] / 2
    phase_spectrum = np.angle(fftw_result)

    half_N = N // 2
    return freq_all[:half_N], amplitude_spectrum[:half_N], phase_spectrum[:half_N]

def get_base_freq_info(freq, amp, pha, target_freq):
    """ Extract amplitude and phase at the target frequency """
    if len(freq) == 0: return 0.0, 0.0, 0.0
    idx = np.argmin(np.abs(freq - target_freq))
    return freq[idx], amp[idx], pha[idx]

def get_equivalent_impedance(voltage, current, dt, target_freq):
    """ Calculate Load Impedance Z = V / I """
    # 1. FFT
    f_v, amp_v, pha_v = get_fftw(voltage, dt)
    f_i, amp_i, pha_i = get_fftw(current, dt)
    
    # 2. Extract Phasors
    _, v_mag, v_phase = get_base_freq_info(f_v, amp_v, pha_v, target_freq)
    _, i_mag, i_phase = get_base_freq_info(f_i, amp_i, pha_i, target_freq)
    
    if i_mag < 1e-10: return 50.0 # Prevent division by zero
    
    # 3. Calculate Z
    V_phasor = v_mag * np.exp(1j * v_phase)
    I_phasor = i_mag * np.exp(1j * i_phase)
    
    return V_phasor / I_phasor

def get_ref_coef(x, *args):
    """
    Calculate Gamma for Topology:
    Source(Rs) -> Node A
    Node A -> Shunt Cm1 -> Ground
    Node A -> Series Cm2 -> Series Lm -> Series Rm -> Load -> Ground
    """
    Cm1, Cm2 = x
    c = args[0] # PlasmaParameters object

    # 1. Component Impedances
    Z_Cm1 = cal_impedance_capacity(Cm1, c.freq)
    Z_Cm2 = cal_impedance_capacity(Cm2, c.freq)
    Z_Lm  = cal_impedance_inductor(c.Lm, c.freq)
    Z_Rm  = c.Rm # Resistance in matching network (if any)
    
    # 2. Calculate Input Impedance (Z_in)
    # Branch 2: Series leg (Cm2 + Lm + Rm + Load)
    Z_branch2 = Z_Cm2 + Z_Lm + Z_Rm + c.Z_load
    
    # Shunt leg: Cm1
    # Z_in = Cm1 || Branch2
    if (Z_Cm1 + Z_branch2) == 0:
        Z_in = 1e15 # Avoid singularity
    else:
        Z_in = (Z_Cm1 * Z_branch2) / (Z_Cm1 + Z_branch2)
        
    # 3. Calculate Reflection Coefficient Gamma
    # Gamma looking into the matching network from Source (Rs)
    Gamma = (Z_in - c.Rs) / (Z_in + c.Rs)
    
    return abs(Gamma)

def test_ref_coef(x, *args):
    return get_ref_coef(x, *args)