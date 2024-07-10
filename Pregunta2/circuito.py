import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parámetros del circuito
R = 2110 / 87  # Resistencia total en ohmios
L = 0.0883  # Inductancia en henrios (88.3 mH)
C = 10e-6  # Capacitancia en faradios (10 µF)
omega = 60  # Velocidad angular en rad/s
Vg_amplitude = 55  # Amplitud del voltaje de la fuente en voltios

# Tiempo de simulación
t = np.linspace(0, 0.1, 1000)  # Vector de tiempo de 0 a 0.1 segundos con 1000 puntos
dt = t[1] - t[0]  # Paso de tiempo

# Inicialización de variables
i = np.zeros_like(t)  # Corriente en el circuito (inicialmente cero)
u = np.zeros_like(t)  # Derivada de la corriente (inicialmente cero)

# Función para Vg(t)
Vg = lambda t: Vg_amplitude * np.sin(omega * t)

# Método de Runge-Kutta de 4º orden
for n in range(len(t) - 1):
    # Calcula k1 para i y u
    k1_i = dt * u[n]
    k1_u = dt * (Vg(t[n]) - R * u[n] - (1/C) * i[n]) / L

    # Calcula k2 para i y u
    k2_i = dt * (u[n] + 0.5 * k1_u)
    k2_u = dt * (Vg(t[n] + 0.5 * dt) - R * (u[n] + 0.5 * k1_u) - (1/C) * (i[n] + 0.5 * k1_i)) / L

    # Calcula k3 para i y u
    k3_i = dt * (u[n] + 0.5 * k2_u)
    k3_u = dt * (Vg(t[n] + 0.5 * dt) - R * (u[n] + 0.5 * k2_u) - (1/C) * (i[n] + 0.5 * k2_i)) / L

    # Calcula k4 para i y u
    k4_i = dt * (u[n] + k3_u)
    k4_u = dt * (Vg(t[n] + dt) - R * (u[n] + k3_u) - (1/C) * (i[n] + k3_i)) / L

    # Actualiza el valor de la corriente i
    i[n + 1] = i[n] + (1/6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)

    # Actualiza el valor de la derivada de la corriente u
    u[n + 1] = u[n] + (1/6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

# Función de ajuste (serie de Fourier con términos sinusoidales y cosenoidales)
def fourier_series(t, *a):
    result = a[0]
    n_terms = (len(a) - 1) // 2
    for n in range(1, n_terms + 1):
        result += a[2*n-1] * np.sin(n * omega * t) + a[2*n] * np.cos(n * omega * t)
    return result

# Estimación inicial de parámetros (por ejemplo, cero para todos)
initial_guess = [0] * 11  # Por ejemplo, 5 términos senoidales y 5 cosenoidales más el término constante

# Ajuste de curva
params, params_covariance = curve_fit(fourier_series, t, i, p0=initial_guess)

# Función ajustada
i_fit = fourier_series(t, *params)

# Mostrar los parámetros de ajuste
print("Parámetros de la serie de Fourier ajustada:")
for n in range(len(params)):
    print(f"a[{n}] = {params[n]}")

# Graficar resultados
plt.figure(figsize=(10, 8))

# Gráfica de la corriente i(t)
plt.subplot(2, 1, 1)
plt.plot(t, i, label='i(t) simulada')
plt.plot(t, i_fit, label='i(t) ajustada', linestyle='--')
plt.title('Corriente i(t)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente [A]')
plt.legend()

# Ajustar la disposición de las gráficas
plt.tight_layout()
plt.show()

# Imprimir la función ajustada con desfase
def print_fourier_series_with_phase(params):
    terms = [f"{params[0]:.4f}"]
    n_terms = (len(params) - 1) // 2
    for n in range(1, n_terms + 1):
        A_n = np.sqrt(params[2*n-1]**2 + params[2*n]**2)
        phi_n = np.arctan2(params[2*n], params[2*n-1])
        if A_n > 1e-4:  # Umbral para términos significativos
            terms.append(f"{A_n:.4f} * sin({n} * 60 * t + {phi_n:.4f})")
    return " + ".join(terms)

print("Función i(t) ajustada con desfase:")
print(f"i(t) = {print_fourier_series_with_phase(params)}")
