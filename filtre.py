import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt,butter





def filtre_notch(signal, fs, notch_freq, Q):
    # Calculer les coefficients du filtre
    b, a = iirnotch(notch_freq / (fs / 2), Q)

    # Appliquer le filtre
    filtered_signal = filtfilt(b, a, signal)

    """

    # Tracer le signal original et le signal filtré
    t = np.arange(len(signal)) / fs  # time vector
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, color='red', label='Signal ECG original')
    plt.plot(t, filtered_signal, color='blue', label='Signal ECG filtré')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.title('Comparaison du signal ECG original et du signal ECG filtré')
    plt.legend()
    plt.grid()
    plt.show()
    
    """
    return filtered_signal


def filter_passe_HB(signal, fs, type_filtre, fc):
    # Define the filter parameters
    # cutoff frequency
    order = 4  # filter order

    # Generate the filter coefficients
    b, a = butter(order, fc / (fs / 2), btype=type_filtre)

    # Pad the input signal with zeros at the beginning and end
    padlen = len(b) - 1  # length of padding on each end
    x_padded = np.pad(signal, padlen, mode='edge')

    # Apply the filter using filtfilt
    y = filtfilt(b, a, x_padded)

    # Remove the padding from the output signal
    y = y[padlen:-padlen]

    """
    # Plot the original and filtered signals
    t = np.arange(len(signal)) / fs  # time vector
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'b-', label='Signal ECG original')
    plt.plot(t, y, 'r-', linewidth=2, label='Signal ECG filtré')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.title('Comparaison du signal ECG original et du signal ECG filtré')
    plt.legend()
    plt.show()
    """
    return y