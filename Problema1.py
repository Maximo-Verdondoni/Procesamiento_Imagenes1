import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local_histograma(img, win_size=(31, 31)):
    """
    Aplica ecualización local del histograma píxel a píxel mediante ventana deslizante.
    
    Parámetros:
      img: Matriz 2D en escala de grises (np.uint8).
      win_size: Dimensiones (M, N) de la ventana local de procesamiento.
    """
    M, N = win_size
    
    # 1. Margen de acolchado (padding) para centrar la ventana en cada píxel
    # Se utiliza división entera para obtener el radio desde el píxel central
    pad_r = M // 2
    pad_c = N // 2
    
    # 2. Extensión de bordes mediante replicación
    # Se replica el último valor en lugar de agregar ceros para evitar un escalón
    # artificial de intensidad en las fronteras de la imagen.
    img_pad = cv2.copyMakeBorder(img, pad_r, pad_r, pad_c, pad_c, cv2.BORDER_REPLICATE)
    
    alto, ancho = img.shape
    img_out = np.zeros_like(img)
    total_pixeles = M * N  # Cantidad total de muestras dentro del vecindario local

    # 3. Recorrido de la ventana deslizante sobre las coordenadas originales
    for i in range(alto):
        for j in range(ancho):
            # Extracción del vecindario de tamaño M x N
            ventana = img_pad[i : i + M, j : j + N]
            
            # Píxel central de la vecindad actual
            val_central = img_pad[i + pad_r, j + pad_c]
            
            # 4. Cálculo directo de la CDF (función de distribución acumulada) local:
            # En la ecualización, el nuevo valor depende de la suma acumulada del histograma
            # hasta la intensidad del píxel analizado.
            # 'ventana <= val_central' genera una máscara booleana con valor True en aquellos
            # píxeles locales menores o iguales al central. np.sum() cuenta cuántos son.
            conteo_menores_o_iguales = np.sum(ventana <= val_central)
            
            # 5. Transformación y escalado al rango completo [0, 255]:
            # La fracción (conteo / total_pixeles) representa la probabilidad acumulada local P(r <= r_central).
            # Se multiplica por 255, se redondea y se convierte a entero sin signo de 8 bits (uint8).
            img_out[i, j] = np.uint8(np.round((conteo_menores_o_iguales / total_pixeles) * 255))
            
    return img_out


# --- Programa Principal ---

# Lectura en escala de grises forzada (IMREAD_GRAYSCALE)
img = cv2.imread('TP1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# --- 1. Procesamiento con múltiples ventanas ---
ventanas = [3,11, 31, 51, 61, 81]
salidas_locales = {}

print("Procesando ventanas locales...")
for w in ventanas:
    print(f"Calculando para ventana {w}x{w}...")
    salidas_locales[w] = ecualizacion_local_histograma(img, win_size=(w, w))

# Ecualización global
img_eq_global = cv2.equalizeHist(img)

# --- 2. Configuración del panel de visualización (2 filas x 4 columnas) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Lista de imágenes con sus respectivos títulos para graficar ordenadamente
imagenes_a_mostrar = [
    ("Original (Fig. 1)", img),
    ("Ecualización Global", img_eq_global),
    ("Ecualización Local 3x3", salidas_locales[3]),
    ("Ecualización Local 11x11", salidas_locales[11]),
    ("Ecualización Local 31x31", salidas_locales[31]),
    ("Ecualización Local 51x51", salidas_locales[51]),
    ("Ecualización Local 61x61", salidas_locales[61]),
    ("Ecualización Local 81x81", salidas_locales[81]),
]

# Recorremos la lista y graficamos en la grilla
for ax, (titulo, im) in zip(axes.flat, imagenes_a_mostrar):
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo, fontsize=11)
    ax.axis('off')

# Como son 7 imágenes en una grilla de 8 (2x4), ocultamos el casillero sobrante
axes.flat[-1].axis('off')

plt.tight_layout()
plt.savefig('resultado_ecualizacion_local.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Figura guardada exitosamente en 'resultado_ecualizacion_local.png'")