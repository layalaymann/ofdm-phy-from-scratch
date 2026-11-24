import numpy as np


def qpsk_map(bits):
   
    bits = np.asarray(bits, dtype=int)
    b = bits.reshape(-1, 2) 
    real_part = 1 - 2 * b[:, 0]  
    imag_part = 1 - 2 * b[:, 1]  
    return (real_part + 1j * imag_part) / np.sqrt(2)


def qpsk_demap(symbols):
    
    symbols = np.asarray(symbols)
    b0 = (np.real(symbols) < 0).astype(int)  
    b1 = (np.imag(symbols) < 0).astype(int)  
    return np.column_stack((b0, b1)).ravel()


def awgn(x, snr_db):
    
    x = np.asarray(x, dtype=complex)
    snr_linear = 10 ** (snr_db / 10)
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    )
    return x + noise