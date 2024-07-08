import cv2
import numpy as np
import matplotlib.pyplot as plt
from interpolar import compute_spline

# Ruta de la imagen
ruta_imagen = 'avion.jpg'

# Leer la imagen
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
imagen_color = cv2.imread(ruta_imagen)

   
#img = cv2.rotate(imagen_color, cv2.ROTATE_180)
kernel = np.ones((5,5), np.uint8)

hsv=cv2.cvtColor(imagen_color, cv2.COLOR_BGR2HSV)
lower_gray = np.array([94, 0, 0])
upper_gray = np.array([166, 255, 255])

Mask = cv2.inRange(hsv, lower_gray, upper_gray)
Mask = cv2.erode(Mask, kernel, iterations=1)
Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
Mask = cv2.dilate(Mask, kernel, iterations=1)

Mask = cv2.bitwise_not(Mask)

cv2.imshow('mask', Mask)

    # Aplicar umbral
_, umbral = cv2.threshold(Mask, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
contornos, _ = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contornos) == 0:
        print("Error: No se encontraron contornos en la imagen.")
else:
        # Asumimos que el contorno más grande es la figura deseada
        contorno = max(contornos, key=cv2.contourArea)

        # Extraer los puntos del contorno superior
        contorno_superior = contorno[:, 0, :]
        contorno_superior = sorted(contorno_superior, key=lambda x: x[0])  # Ordenar por la coordenada x

        # Dividir en coordenadas X e Y
       
        highest_y_per_x={}
        for p in contorno_superior:
            if p[0] not in highest_y_per_x:
                 highest_y_per_x[p[0]]=p[1]
            else:
                highest_y_per_x[p[0]]=max(highest_y_per_x[p[0]], p[1])
        filtered_point=[]
        for x,y in highest_y_per_x.items():
            filtered_point.append([x,y])
        Y = np.array([p[1] for p in filtered_point])
        X = np.array([p[0] for p in filtered_point])

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
            # # Asegurarse de que los puntos X sean estrictamente crecientes
            X_unicos, indices_unicos = np.unique(X_superior, return_index=True)
            Y_unicos = Y_superior[indices_unicos]
            
                    
                

            # Interpolación usando splines cúbicos
            try:
            
                spline = compute_spline(X_unicos, Y_unicos)
            
                    # Generar valores para graficar el spline interpolado
                X_interpolados = np.linspace(min(X_superior), max(X_superior), 1000)
                    #Y_interpolados = spline(X_interpolados)
                y_interpolados = [spline(y) for y in X_interpolados]

                    # Graficar la imagen original
                plt.imshow(cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB))
                plt.plot(X_superior, Y_superior, 'o', label='Puntos del contorno superior', markersize=2)
                plt.plot(X_interpolados, y_interpolados, '-', label='Interpolación usando Splines Cúbicos', color='red')
                    #plt.gca().invert_yaxis()  # Invertir el eje y para que la imagen se vea correctamente
                plt.legend()
                plt.show()
            except Exception as e:
                print(f"Error durante la interpolación: {e}")

   
