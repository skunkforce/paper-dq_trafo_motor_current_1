# -*- coding: utf-8 -*-
"""Electrical Signature Analysis Library

This is a library which contains algorithms for the analysis of synchronous and 
asynchronous motors. 

Example:
    literal blocks::

        $ python3
        >>> from lib import ESA
        >>> i_d, i_q = ESA.dq(data['I1'], data['I2'], data['I3'])

Todo:
    * 

.. _PEP 8 -- Style Guide for Python Code:
   https://www.python.org/dev/peps/pep-0008/

"""

from matplotlib.colors import Normalize
import numpy as np

from . import signal_processing as DSP

def calc_fft(signal, fs):
    return DSP.power_fft(signal, fs)

def rms(signal):
    return DSP.rms(signal)

def correlation(x, y):
    r = np.corrcoef(x, y)
    return np.abs(r[0, 1])

def dq(i_a, i_b, i_c):
    if not isinstance(i_a, np.ndarray):
        i_a = np.array(i_a)
        i_b = np.array(i_b)
        i_c = np.array(i_c)

    v1 = np.sqrt(2/3)
    v2 = 1 / np.sqrt(6)
    v3 = v2
    v4 = 1 / np.sqrt(2)
    v5 = v4
    
    i_d =  v1 * i_a - v2 * i_b - v3 * i_c,
    i_q = v4 * i_b - v5 * i_c,
    
    return np.squeeze(i_d), np.squeeze(i_q)

def epva(i_d, i_q, fs=500):
    """Enhanced Park's Vector Approach

    Args:
        i_d ([type]): [description]
        i_q ([type]): [description]
        fs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    if not isinstance(i_d, np.ndarray):
        i_d = np.array(i_d)
        i_q = np.array(i_q)

    result = np.sqrt(np.abs(i_d +  1j *i_q))
    
    freq, fft = DSP.power_fft(result, fs=fs)
    
    return freq, fft

def csa(signal, fs=500):
    """Current Signature Analysis

    Args:
        signal ([type]): [description]
        fs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    freq, fft = DSP.power_fft(signal, fs=fs)
    return freq, fft

def vsa(signal, fs=500):
    """Voltage Signature Analysis

    Args:
        signal ([type]): [description]
        fs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    freq, fft = DSP.power_fft(signal, fs=fs)
    return freq, fft

def ipsa(v_ll, i_l, fs=500, norm=True):
    """Instantaneous Power Signature Analysis

    Args:
        v_ll ([type]): [description]
        i_l ([type]): [description]
        fs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    p_l = np.multiply(v_ll, i_l-np.mean(i_l))
    freq, fft = DSP.power_fft(p_l, fs=fs)
    if norm:
        fft = DSP.decibel(fft, normalize=True)
    
    return freq, fft
    

def mcsa(signal, fs=500):
    """Motor Current Signature Analysis

    Args:
        signal ([type]): [description]
        fs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    freq, fft = csa(signal, fs=fs)
    return freq, fft

def dq0(i_a, i_b, i_c, theta=0):
       
    current = np.array([i_a-np.mean(i_a), i_b-np.mean(i_b), i_c-np.mean(i_c)])
    
    D = np.sqrt(2/3) * np.matrix([[1, -1/2, -1/2],
              [0, np.sqrt(3)/2, -np.sqrt(3)/2],
              [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])
    
    D = np.sqrt(2/3) * np.matrix(  [[np.cos(theta),   np.cos(theta - ((2*np.pi)/3)),  np.cos(theta - ((4*np.pi)/3))],
                                [-np.sin(theta), -np.sin(theta - ((2*np.pi)/3)), -np.sin(theta - ((4*np.pi)/3))],
                                [1/2, 1/2, 1/2]]
                                )
    
    out = D * current
    i_d = out[0]
    i_q = out[1]
    i_0 = out[2]
    
    return np.array(i_d).flatten(), np.array(i_q).flatten(), np.array(i_0).flatten()

def apply_calib(signal, gain, offset):
    """Applies gain and offset factors to signal (y = mx + b)

    Args:
        signal ([type]): [description]
        gain ([type]): [description]
        offset ([type]): [description]
    """
    return np.multiply(gain, signal) + offset