import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import triang
from helpers import generate_triangle_signal
from scipy.signal import sawtooth

# Parametry symulacji
num_samples = 1000  # Liczba próbek
beta = 100  # Skalowanie początkowej macierzy kowariancji
lambda_factor = 0.999  # Współczynnik zapominania

# Rzeczywiste parametry (z zmiennym theta_true2)
theta_true0  = 0.6  # Stały parametr
theta_true1  = 0.4  # Stały parametr
t = np.linspace(0, 10, num_samples)  # Oś czasu
theta_true2  = 0.4*generate_triangle_signal(500,2)


# Początkowe wartości estymacji parametrów
theta_est = np.zeros(3)
theta_est[0] =0.1
P = beta * np.eye(3)  # P(0) = β * I (macierz kowariancji)
theta_est_history =[theta_est.copy()]

# Inicjalizacja sygnału sterującego i wyjścia systemu
u = np.zeros(num_samples)
y = np.zeros(num_samples)

# wartość zadana
d_k = 2

# Tablica do przechowywania błędu kwadratowego
error_squared = np.zeros(num_samples)

for k in range(2, num_samples):
    if theta_est[0] == 0:
        theta_est[0] = 1e-6
    u[k] = (d_k - theta_est[1] * u[k - 1] - theta_est[2] * u[k - 2]) / theta_est[0]
    
    # Generowanie wyjścia systemu z szumem normalnym
    y[k] = (
        theta_true0 * u[k]
        + theta_true1 * u[k - 1]
        + theta_true2[k] * u[k - 2]
        + np.random.normal(0, 0.1)
    )

    # Obliczanie błędu kwadratowego
    error_squared[k] = (y[k] - d_k) ** 2

    # Regressor Φ_k
    phi_k = np.array([u[k], u[k - 1], u[k - 2]]).reshape(-1, 1)

    # Błąd predykcji
    y_pred = phi_k.T @ theta_est
    error = y[k] - y_pred

    # Aktualizacja wektora wzmocnienia
    P_phi = P @ phi_k
    gain = P_phi / (lambda_factor + phi_k.T @ P_phi)

    # Aktualizacja estymat parametrów
    theta_est = theta_est + gain.flatten() * error

    # Aktualizacja macierzy kowariancji
    P = (1 / lambda_factor) * (P - gain @ phi_k.T @ P)

    # Zapis estymacji parametrów
    theta_est_history.append(theta_est.copy())


# Konwersja historii estymat do tablicy
theta_est_history = np.array(theta_est_history)

# Obliczanie średniego błędu kwadratowego w czasie
cumulative_mse = np.cumsum(error_squared) / np.arange(1, num_samples + 1)
time_steps = np.arange(num_samples)

# Wykres wyjścia systemu y_k w czasie
plt.figure(figsize=(12, 6))
plt.plot(y, label='Wyjście $y_k$', linewidth=2)
plt.plot(time_steps, theta_true2, label=r'$theta-true_2$', linewidth=2)
plt.axhline(y=d_k, color='r', linestyle='--', label='Wartość zadana $d_k=1$')
plt.xlabel('Czas (k)')
plt.ylabel('Wyjście $y_k$')
plt.title('$y_k=1$')
plt.legend()
plt.grid()
plt.show()


# Wykres średniego błędu kwadratowego w czasie
plt.figure(figsize=(12, 6))
plt.plot(time_steps, cumulative_mse, label='Średni błąd kwadratowy', linewidth=2)
plt.xlabel('Czas (k)')
plt.ylabel('Średni błąd kwadratowy')
plt.title('Ocena jakości sterowania - średni błąd kwadratowy')
plt.legend()
plt.grid()
plt.show()
