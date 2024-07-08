import numpy as np
import matplotlib.pyplot as plt

# Parámetros del circuito
R = 24.25  # Resistencia total en ohmios
L = 0.05  # Inductancia en henrios (50 mH)
C = 10e-6  # Capacitancia en faradios (10 µF)
omega = 60  # Velocidad angular en rad/s
Vg_amplitude = 165  # Amplitud del voltaje de la fuente en voltios

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

# Calcula el voltaje en la resistencia R
V_R = R * i

# Graficar resultados
plt.figure(figsize=(10, 8))

# Gráfica de la corriente i(t)
plt.subplot(2, 1, 1)
plt.plot(t, i, label='i(t)')
plt.title('Corriente i(t)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Corriente [A]')
plt.legend()

# Gráfica del voltaje en la resistencia V_R(t)
plt.subplot(2, 1, 2)
plt.plot(t, V_R, label='V_R(t)', color='orange')
plt.title('Voltaje en la Resistencia V_R(t)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.legend()

# Ajustar la disposición de las gráficas
plt.tight_layout()
plt.show()
