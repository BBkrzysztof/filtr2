import numpy as np
from scipy.signal.windows import triang 

def generate_triangle_signal(length_per_period, num_periods):
    """
    Generates triangle signal.

    Args:
        length_per_period: Number of points in one period of signal
        num_periods: Number of periods in generated signal

    Returns:
        Triangle signal.
    """
    single_period = triang(length_per_period)
    return np.tile(single_period, num_periods) 

def add_noise(signal, noise_level):
    """
    Generates noise to the signal.

    Args:
        signal: Signal on which noise will be applied.
        noise_level (float, 0 to 1): Variance of noise that will be applied on signal.  

    Returns:
        Triangle signal with noise applied.
    """
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def moving_average_filter(signal, window_size):
    """
    Applies a moving average filter to signal.
    
    Args:
        signal: Signal to be filtered.
        windows_size: Size of the moving average window.

    Returns:
        Filtered signal.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def calculate_mse(original, estimated):
    """
    Calculates Mean Squared Error (MSE) between the original and estimated signals.

    Args:
        original: Original (true) signal.
        estimated: Estimated (filtered) signal.

    Returns:
        MSE value.
    """
    n = len(original)
    mse = np.sum((original - estimated) ** 2) / n
    return mse