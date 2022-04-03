import numpy as np

def power_fft(signal, fs):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    signal = signal - np.mean(signal)
    fft = np.power(np.abs(np.fft.rfft(signal)),2)
    freq = np.fft.rfftfreq(signal.size, d=1/fs)
    
    return freq, fft

def complex_fft(signal, fs):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    signal = signal - np.mean(signal)
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1/fs)
    
    return freq, fft

def decibel(fft, normalize=False):
    if normalize == True:
        fft = fft/max(fft)
    return 20 * np.log10(fft)

def rms(signal):
    signal = np.array(signal)
    signal = signal - np.mean(signal)
    rms = np.sqrt(np.mean(np.power(signal, 2)))
    return rms

def power_cepstrum(signal, fs):
    ceps =  np.power(
                np.abs(
                    np.fft.irfft(
                        np.log(
                            np.abs(
                                np.power(
                                    np.fft.rfft(signal)
                                , 2)
                            )
                        )
                    )
                )
            ,2)
    
    return 

def complex_cepstrum(signal, fs):
    """Computes cepstrum."""
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    frame_size = signal.size
    windowed_signal = np.hamming(frame_size) * signal
    
    X = np.fft.rfft(windowed_signal)
    log_X = np.log(np.abs(X))
    cepstrum = np.fft.rfft(log_X)

    dt = 1/fs
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    
    return quefrency_vector, cepstrum