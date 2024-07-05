import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Ruta de la imagen
ruta_imagen = 'C:/Users/Usuario/Desktop/ProyectoCN/Pregunta1/avion.jpg'

# Leer la imagen
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
imagen_color = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"Error: No se pudo leer la imagen '{ruta_imagen}'.")
else:
        # Si es necesario, rotar la imagen
    imagen = cv2.rotate(imagen, cv2.ROTATE_180)


    # Aplicar umbral
    _, umbral = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        print("Error: No se encontraron contornos en la imagen.")
    else:
        # Asumimos que el contorno más grande es la figura deseada
        contorno = max(contornos, key=cv2.contourArea)

        # Extraer los puntos del contorno superior
        contorno_superior = contorno[:, 0, :]
        contorno_superior = sorted(contorno_superior, key=lambda x: x[0])  # Ordenar por la coordenada x

        # Dividir en coordenadas X e Y
        X = np.array([p[0] for p in contorno_superior])
        Y = np.array([p[1] for p in contorno_superior])

        # Filtrar solo la parte superior del contorno
        filtro_superior = Y < np.mean(Y)
        X_superior = X[filtro_superior]
        Y_superior = Y[filtro_superior]

        # Verificar si hay valores NaN en los datos
        if np.any(np.isnan(X_superior)) or np.any(np.isnan(Y_superior)):
            print("Error: Se encontraron valores NaN en los datos.")
        elif len(X_superior) < 2 or len(Y_superior) < 2:
            print("Error: No hay suficientes puntos para la interpolación.")
        else:
            # Asegurarse de que los puntos X sean estrictamente crecientes
            X_unicos, indices_unicos = np.unique(X_superior, return_index=True)
            Y_unicos = Y_superior[indices_unicos]

            # Interpolación usando splines cúbicos
            try:
                spline = CubicSpline(X_unicos, Y_unicos)

                # Generar valores para graficar el spline interpolado
                X_interpolados = np.linspace(min(X_unicos), max(X_unicos), 1000)
                Y_interpolados = spline(X_interpolados)

                # Graficar la imagen original
                plt.imshow(cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB))
                plt.plot(X_superior, Y_superior, 'o', label='Puntos del contorno superior', markersize=2)
                plt.plot(X_interpolados, Y_interpolados, '-', label='Interpolación usando Splines Cúbicos', color='red')
                plt.gca().invert_yaxis()  # Invertir el eje y para que la imagen se vea correctamente
                plt.legend()
                plt.show()
            except Exception as e:
                print(f"Error durante la interpolación: {e}")
