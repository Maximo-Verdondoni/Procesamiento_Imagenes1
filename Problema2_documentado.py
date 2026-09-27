"""
PROBLEMA 2 - PROCESAMIENTO DE EXÁMENES CON OPENCV
=================================================

OBJETIVO GENERAL
----------------
Este programa recibe imágenes de exámenes y realiza automáticamente:

1. Detecta la grilla.
2. Extrae las 10 celdas de preguntas.
3. Localiza la zona probable de la respuesta.
4. Detecta componentes conexas que pueden ser letras.
5. Clasifica A/B/C/D usando propiedades geométricas y topológicas.
6. Corrige contra una clave oficial.
7. Valida Name, Date y Class.
8. Genera un resumen visual.

IDEA FUNDAMENTAL
----------------
El pipeline combina:

- PROYECCIONES: sumas de píxeles por filas/columnas para encontrar la grilla.
- COMPONENTES CONEXAS: agrupación de píxeles conectados para detectar objetos.
- CARACTERÍSTICAS GEOMÉTRICAS: ancho, alto, área y relaciones entre ellos.
- TOPOLOGÍA: cantidad de huecos internos de una letra.

RESOLUTION-INDEPENDENCE
-----------------------
Los tamaños geométricos importantes se expresan como proporciones del tamaño
real de la imagen, tabla, celda o ROI. Por ejemplo:

    int(w * 0.015)

significa aproximadamente "1.5 % del ancho", no "15 píxeles".

Las secciones finales de debugging sirven para visualizar resultados
intermedios y comprobar cada etapa.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CLAVE OFICIAL DE RESPUESTAS
# ==========================================================

# Diccionario que guarda cuál es la respuesta correcta de cada pregunta.
# La clave del diccionario es el número de pregunta y el valor es la letra
# correcta.
RESPUESTAS_CORRECTAS = {
    1: 'C', 2: 'B', 3: 'A', 4: 'D', 5: 'B',
    6: 'B', 7: 'A', 8: 'B', 9: 'D', 10: 'D'
}


# ==========================================================
# PASO 1: Segmentar la grilla del examen (Proyecciones)
# ==========================================================

def segmentar_grilla(img_gray):
    """
    Encuentra las líneas de la grilla del examen.

    Entrada:
        img_gray -> imagen del examen en escala de grises.

    Salida:
        x_coords -> coordenadas X de las líneas verticales principales.
        y_coords -> coordenadas Y de las líneas horizontales.

    La idea general es:
        1. Convertir la imagen en una imagen binaria.
        2. Sumar tinta por columna para encontrar líneas verticales.
        3. Usar una línea vertical como referencia para conocer
           el rango vertical de la tabla.
        4. Sumar tinta por fila para encontrar líneas horizontales.
    """

    # ------------------------------------------------------
    # 1. Obtener dimensiones de la imagen
    # ------------------------------------------------------

    # img_gray.shape devuelve (alto, ancho) porque la imagen es una matriz.
    #
    # h -> cantidad de filas de píxeles = alto de la imagen.
    # w -> cantidad de columnas de píxeles = ancho de la imagen.
    h, w = img_gray.shape

    # ------------------------------------------------------
    # 2. Binarización de la imagen
    # ------------------------------------------------------

    # Una imagen en escala de grises tiene normalmente valores entre 0 y 255:
    #     0   -> negro
    #     255 -> blanco
    #
    # Queremos separar:
    #     tinta -> 1
    #     fondo -> 0
    #
    # Por eso consideramos "oscuro" todo píxel menor que 120.
    #
    # La comparación img_gray < 120 produce una matriz booleana:
    #     True  -> el píxel es oscuro
    #     False -> el píxel no es oscuro
    #
    # astype(np.uint8) convierte:
    #     True  -> 1
    #     False -> 0
    img_bin = (img_gray < 120).astype(np.uint8)

    # ======================================================
    # 1. Detección de las líneas verticales principales
    # ======================================================

    # Sumamos todos los píxeles de cada COLUMNA.
    #
    # axis=0 significa que reducimos la dimensión vertical y obtenemos
    # una suma para cada columna X.
    #
    # Si una columna pasa por una línea vertical de la tabla, tendrá
    # muchos píxeles de tinta y, por lo tanto, una suma grande.
    #
    # Ejemplo conceptual:
    #     columna normal -> pocos píxeles oscuros -> suma pequeña
    #     línea vertical -> muchos píxeles oscuros -> suma grande
    # Cada posición de proj_cols representa una columna y contiene su cantidad de tinta.
    # Las líneas verticales aparecen como picos de esta proyección.
    proj_cols = np.sum(img_bin, axis=0)

    # Definimos un umbral para decidir si una columna tiene suficiente tinta
    # como para considerarla parte de una línea vertical.
    #
    # Si la imagen tiene h filas, una línea vertical que recorra gran parte
    # del alto tendrá una suma relativamente grande.
    #
    # 0.40 significa 40 % del alto de la imagen.
    th_col = h * 0.40

    # np.where(...) devuelve las posiciones que cumplen la condición.
    # En este caso, las columnas donde la proyección supera el umbral.
    #
    # El resultado contiene todas las columnas que parecen formar parte
    # de alguna línea vertical.
    cols_lineas = np.where(proj_cols > th_col)[0]

    # Una misma línea vertical suele tener varios píxeles de grosor.
    # Entonces podemos obtener varias columnas consecutivas, por ejemplo:
    #     [100, 101, 102, 103]
    # aunque todo eso represente UNA sola línea física.
    #
    # np.diff(cols_lineas) calcula la diferencia entre columnas consecutivas.
    # Si tenemos:
    #     [100, 101, 102, 150, 151]
    # obtenemos:
    #     [1, 1, 48, 1]
    #
    # El salto de 48 indica que comenzó otro grupo.
    #
    # En vez de usar px fijos, usamos una tolerancia proporcional al ancho
    # (aprox 1.5% del ancho de la imagen) con un mínimo seguro de 3 px

    dist_cols = max(3, int(w * 0.015))
    #
    # +1 se utiliza porque np.split necesita la posición donde comenzar
    # cada nuevo bloque.
    grupos_cols = np.split(
        cols_lineas,
        np.where(np.diff(cols_lineas) > dist_cols)[0] + 1
    )

    # Cada grupo representa una única línea vertical.
    #
    # Para obtener una sola coordenada X representativa de cada línea,
    # calculamos el promedio de las columnas pertenecientes al grupo.
    #
    # Ejemplo:
    #     [100,101,102] -> media = 101
    #
    # int(...) convierte la coordenada resultante a entero.
    # if len(g) > 0 evita procesar grupos vacíos.
    x_coords = [int(np.mean(g)) for g in grupos_cols if len(g) > 0]

    # Ordenamos las coordenadas de menor a mayor.
    # Así quedan de izquierda a derecha.
    x_coords = sorted(x_coords)

    # ------------------------------------------------------
    # 3. Identificar los límites de la primera columna
    # ------------------------------------------------------

    # Tomamos las dos primeras líneas verticales detectadas.
    #
    # x_coords[0] -> borde izquierdo de la primera columna.
    # x_coords[1] -> borde derecho de la primera columna.
    c1_x0, c1_x1 = x_coords[0], x_coords[1]

    # Calculamos el ancho de la primera columna en píxeles.
    # Esto se usa después para definir el umbral de las líneas horizontales.
    ancho_c1 = c1_x1 - c1_x0

    # ======================================================
    # 2. Rango vertical de la tabla de preguntas
    # ======================================================

    # Miramos UNA SOLA COLUMNA de la imagen: precisamente la que corresponde
    # a la línea vertical izquierda de la primera columna de la tabla.
    #
    # img_bin[:, c1_x0]
    #     :        -> tomar todas las filas Y
    #     c1_x0    -> tomar solamente la columna X de la línea vertical
    #
    # Entonces obtenemos una lista de valores 0/1 a lo largo de esa línea.
    #
    # Al buscar == 1 estamos preguntando:
    # "¿En qué filas hay tinta sobre esta línea vertical?"
    #
    # np.where(...)[0] devuelve las coordenadas Y donde la respuesta es True.
    filas_con_linea_vert = np.where(img_bin[:, c1_x0] == 1)[0]

    # El primer elemento es la primera fila donde aparece la línea vertical.
    # Esa posición se toma como el comienzo vertical de la tabla.
    y_inicio_caja = filas_con_linea_vert[0]

    # El último elemento es la última fila donde aparece la línea vertical.
    # Esa posición se toma como el final vertical de la tabla.
    y_fin_caja = filas_con_linea_vert[-1]

    alto_caja = y_fin_caja - y_inicio_caja

    # ======================================================
    # 3. Detección de las líneas horizontales
    # ======================================================

    # Ahora hacemos el proceso "al revés" del utilizado para las columnas.
    #
    # En vez de sumar por COLUMNAS, sumamos por FILAS.
    #
    # Primero recortamos únicamente el ancho de la primera columna:
    #     c1_x0:c1_x1
    #
    # y dejamos todas las filas:
    #     :
    #
    # axis=1 significa que sumamos horizontalmente, por lo que obtenemos
    # una suma para cada fila Y.
    #
    # Una línea horizontal tiene muchos píxeles oscuros distribuidos a lo largo
    # del ancho de la columna y, por lo tanto, produce un pico en esta proyección.
    # Cada posición de proj_rows representa una fila y contiene su cantidad de tinta.
    # Las líneas horizontales aparecen como picos.
    proj_rows = np.sum(img_bin[:, c1_x0:c1_x1], axis=1)

    # Umbral para decidir si una fila tiene suficiente tinta como para
    # pertenecer a una línea horizontal.
    #
    # ancho_c1 * 0.50 significa que exigimos aproximadamente 50 % del ancho
    # de la primera columna cubierto por tinta.
    th_row = ancho_c1 * 0.50

    # ------------------------------------------------------
    # 4. Filtrar solamente las filas relevantes
    # ------------------------------------------------------

    # Analizamos solamente el rango vertical de la tabla.
    #
    # proj_rows[y_inicio_caja:y_fin_caja + 2]
    #     -> recorta la proyección para no mirar el resto de la imagen.
    #
    # Luego:
    #     > th_row
    #     -> conserva únicamente las posiciones que tienen suficiente tinta.
    #
    # np.where(...)[0]
    #     -> devuelve los índices que cumplen la condición.
    #
    # IMPORTANTE:
    # Cuando hacemos un slicing, los índices obtenidos por np.where son
    # relativos al recorte y no a la imagen original.
    #
    # Por eso sumamos y_inicio_caja al final para recuperar las coordenadas
    # Y reales de la imagen.
    #
    # Ejemplo:
    #     y_inicio_caja = 100
    #     np.where(...) -> [5, 6, 7]
    #
    # Las coordenadas reales son:
    #     [105, 106, 107]
    idx_filas = np.where(
        proj_rows[y_inicio_caja:y_fin_caja + 2] > th_row
    )[0] + y_inicio_caja

    # ------------------------------------------------------
    # 5. Agrupar filas que pertenecen a la misma línea
    # ------------------------------------------------------

    # Una línea horizontal normalmente tiene varios píxeles de grosor.
    # Por eso, una misma línea puede producir varias filas consecutivas:
    #
    #     [105, 106, 107]
    #
    # Queremos tratarlas como UNA sola línea.
    #
    # np.diff(idx_filas) calcula los saltos entre posiciones consecutivas.
    # Ejemplo:
    #
    #     idx_filas = [105,106,107,180,181,182]
    #     np.diff(...) = [1,1,73,1,1]
    #
    # Los saltos pequeños corresponden al grosor de una misma línea.
    # El salto grande separa una línea de la siguiente.
    #
    # Usamos una tolerancia proporcional a la altura de la tabla (aprox 2.5% del alto de la caja, con un mínimo de 3 px)
    dist_rows = max(3, int(alto_caja * 0.025))
    #
    # np.where(...)[0] encuentra las posiciones de esos saltos.
    # np.split(...) divide el array justamente en esos puntos.
    grupos_rows = np.split(
        idx_filas,
        np.where(np.diff(idx_filas) > dist_rows)[0] + 1
    )

    # ------------------------------------------------------
    # 6. Representar cada grupo con una sola coordenada Y
    # ------------------------------------------------------

    # Ahora tenemos algo conceptual como:
    #
    #     grupos_rows = [
    #         [105,106,107],
    #         [180,181,182],
    #         [260,261,262]
    #     ]
    #
    # Cada grupo es UNA línea horizontal, pero todavía contiene varias
    # coordenadas debido al grosor de la línea.
    #
    # Calculamos la media de cada grupo para quedarnos con una coordenada
    # representativa.
    #
    # Ejemplo:
    #     [105,106,107] -> 106
    #
    # int(...) convierte la media en un entero.
    # if len(g) > 0 evita calcular la media de un grupo vacío.
    y_coords = [
        int(np.mean(g))
        for g in grupos_rows
        if len(g) > 0
    ]

    # Ordenamos las coordenadas Y de menor a mayor.
    # En una imagen, menor Y significa más arriba y mayor Y significa más abajo.
    y_coords = sorted(y_coords)

    # Devolvemos las coordenadas X y Y de las líneas detectadas.
    return x_coords, y_coords


# ==========================================================
# PASO 2: Extraer celdas y regiones de respuesta (ROIs)
# ==========================================================

def extraer_celdas(img_gray, x_coords, y_coords):
    """
    Extrae las 10 celdas correspondientes a las preguntas.

    La tabla está organizada en dos columnas:
        - preguntas 1 a 5 en la columna izquierda.
        - preguntas 6 a 10 en la columna derecha.

    Utiliza las coordenadas de la grilla para recortar cada celda.
    Los márgenes de recorte se calculan en base a las dimensiones reales de cada celda.
    """

    # una línea adicional corresponde al encabezado de la tabla.
    #
    # Si hay 7 coordenadas, ignoramos la primera y usamos las 6 restantes
    # para delimitar las 5 preguntas.
    if len(y_coords) == 7:
        y_preguntas = y_coords[1:]
    else:
        # Si no hay 7, usamos directamente las coordenadas detectadas.
        y_preguntas = y_coords

    # Las dos primeras líneas verticales delimitan la columna izquierda.
    # Allí están las preguntas 1, 2, 3, 4 y 5.
    col_izq = (x_coords[0], x_coords[1])

    # Las dos últimas líneas verticales delimitan la columna derecha.
    # Allí están las preguntas 6, 7, 8, 9 y 10.
    col_der = (x_coords[2], x_coords[3])

    #Calculamos el ancho de la celdas izquierdas y derechas (aunque son iguales)
    ancho_izq = col_izq[1] - col_izq[0]
    ancho_der = col_der[1] - col_der[0]

    # Margen horizontal proporcional al ancho de cada columna (aprox 2%, mínimo 1 px)
    pad_x_izq = max(1, int(ancho_izq * 0.02))
    pad_x_der = max(1, int(ancho_der * 0.02))

    # Diccionario donde guardaremos cada celda.
    #
    # La estructura final será aproximadamente:
    #     celdas[1]  -> imagen de la pregunta 1
    #     celdas[2]  -> imagen de la pregunta 2
    #     ...
    #     celdas[10] -> imagen de la pregunta 10
    celdas = {}

    # Recorremos cinco posiciones porque hay 5 preguntas por columna.
    for i in range(5):
        # y0 -> límite superior de la celda.
        # y1 -> límite inferior de la celda.
        y0, y1 = y_preguntas[i], y_preguntas[i + 1]
        alto_celda = y1 - y0

        # Margen vertical proporcional a la altura de la celda (aprox 2%, mínimo 1 px)
        pad_y = max(1, int(alto_celda * 0.02))

        # --------------------------------------------------
        # Preguntas 1 a 5: columna izquierda
        # --------------------------------------------------
        #
        # Recortamos la imagen usando slicing:
        #     img_gray[filas, columnas]
        #
        # y0+pad_y y y1-pad_y dejan un pequeño margen para no incluir exactamente
        # la línea de la grilla.
        #
        # Lo mismo para X: sumamos/restamos pad_x_izq píxeles para quitar los bordes.
        celdas[i + 1] = img_gray[
            y0 + pad_y : y1 - pad_y,
            col_izq[0] + pad_x_izq : col_izq[1] - pad_x_izq
        ]

        # --------------------------------------------------
        # Preguntas 6 a 10: columna derecha
        # --------------------------------------------------
        #
        # La misma celda horizontal se utiliza para la pregunta i+6,
        # pero ahora sobre la segunda columna de la tabla.
        celdas[i + 6] = img_gray[
            y0 + pad_y : y1 - pad_y,
            col_der[0] + pad_x_der : col_der[1] - pad_x_der
        ]

    # Devolvemos las 10 celdas recortadas.
    return celdas


# ==========================================================
# Extraer ROI de la letra marcada mediante componentes conexas
# ==========================================================

def obtener_rois_letras_por_componentes(celdas):
    """
    Para cada celda:
        1. Binariza la imagen.
        2. Detecta componentes conexas.
        3. Busca el componente que parece ser el guion largo "_______".
        4. Recorta la región inmediatamente superior al guion.

    Esa región es la ROI donde esperamos encontrar la letra A/B/C/D.
    """

    # Diccionario donde guardaremos una ROI de respuesta por pregunta.
    rois_respuesta = {}

    # Recorremos cada pregunta y su celda correspondiente.
    for num_p, celda in celdas.items():

        # celda.shape devuelve (alto, ancho).
        #
        # ch -> cantidad de filas de la celda.
        # cw -> cantidad de columnas de la celda.
        ch, cw = celda.shape

        # Binarizamos la celda.
        #
        # Todo píxel menor a 130 se considera tinta.
        # Multiplicamos por 255 porque las funciones de OpenCV que trabajan
        # con imágenes binarias suelen usar 0 para fondo y 255 para tinta.
        #
        # Resultado:
        #     255 -> tinta
        #     0   -> fondo
        bin_c = ((celda < 130) * 255).astype(np.uint8)

        # Detectamos componentes conexas.
        #
        # connectedComponentsWithStats devuelve:
        #     num_labels -> cantidad total de componentes, incluyendo fondo.
        #     labels     -> imagen donde cada píxel tiene el ID de su componente.
        #     stats      -> propiedades de cada componente (x, y, ancho, alto, área).
        #     _          -> centroides, que aquí no necesitamos.
        #
        # 8 significa conectividad de 8 vecinos: se consideran conectados
        # también los píxeles que tocan diagonalmente.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bin_c,
            8,
            cv2.CV_32S
        )

        # Variable donde guardaremos la mejor candidata a ser el guion.
        mejor_linea = None

        # Guarda el ancho máximo encontrado entre las candidatas.
        # Lo usamos para quedarnos con el guion más largo.
        max_w = 0

        # Umbrales geométricos adaptativos según la resolución de la celda
        min_w_linea = cw * 0.15
        max_h_linea = max(2, int(ch * 0.15))

        # --------------------------------------------------
        # Buscar el componente con forma de guion
        # --------------------------------------------------
        #
        # Empezamos en 1 porque el componente 0 es el fondo.
        for i in range(1, num_labels):

            # stats[i] contiene:
            #     x      -> coordenada X superior izquierda.
            #     y      -> coordenada Y superior izquierda.
            #     w_box  -> ancho del rectángulo envolvente.
            #     h_box  -> alto del rectángulo envolvente.
            #     area   -> cantidad de píxeles del componente.
            x, y, w_box, h_box, _ = stats[i]

            # Buscamos algo con forma horizontal:
            #     w_box > min_w_linea         -> suficientemente ancho.
            #     h_box <= max_h_linea        -> relativamente bajo.
            #     w/h >= 4                    -> mucho más ancho que alto.
            #
            # max(h_box,1) evita dividir por cero.
            if w_box > min_w_linea and h_box <= max_h_linea and (w_box / max(h_box, 1)) >= 4.0:

                # Además exigimos que esté antes del 80 % de la altura
                # de la celda. Así evitamos confundirlo con algún elemento
                # horizontal demasiado abajo.
                #
                # También nos quedamos con el candidato más ancho.
                if y < ch * 0.80 and w_box > max_w:
                    max_w = w_box
                    mejor_linea = (x, y, w_box, h_box)

        # --------------------------------------------------
        # Crear la ROI que contiene la letra
        # --------------------------------------------------

        if mejor_linea is not None:
            # Recuperamos los datos del rectángulo que representa el guion.
            lx, ly, lw, lh = mejor_linea

            # La letra está por encima del guion.
            # ly es la coordenada Y donde comienza el guion.

            # La altura de la ventana de la letra es proporcional al alto de celda (~13%)
            alto_letra = int(ch * 0.13)
            # Despegue mínimo del guion proporcional (~2% del alto)
            pad_guion = max(1, int(ch * 0.02))

            # max(0, ...) evita que el índice resulte negativo.
            y_top = max(0, ly - alto_letra)
            # Terminamos una fila antes del comienzo del guion.
            y_bot = max(0, ly - pad_guion)
            # La ROI horizontal se limita al ancho del propio guion.
            # max(0, ...) evita coordenadas negativas.
            x_left = max(0, lx)
            # min(cw, ...) evita pasarnos del ancho de la celda.
            x_right = min(cw, lx + lw)

            # Recortamos la región de la celda.
            #
            # Resultado:
            #     ROI = zona donde debería estar la letra marcada.
            roi = celda[y_top:y_bot, x_left:x_right]
        else:
            # Si no encontramos ningún guion suficientemente convincente,
            # devolvemos una ROI blanca de tamaño fijo.
            #
            # Esto representa una región "vacía" proporcional al tamaño 
            # de la celda y permite que el resto del pipeline continúe sin romperse.
            roi = np.ones((int(ch * 0.40), int(cw * 0.30)), dtype=np.uint8) * 255

        # Guardamos la ROI asociada a la pregunta.
        rois_respuesta[num_p] = roi

    # Devolvemos las 10 ROIs de respuesta.
    return rois_respuesta


# ==========================================================
# PASO 3: Limpieza y extracción de componentes de letras
# ==========================================================

def limpiar_roi_letra(roi):
    # Elimina componentes cuyo tamaño no sea compatible con una letra.
    """
    Toma una ROI de respuesta y trata de aislar las letras manuscritas.

    Devuelve:
        mascara_limpia -> imagen binaria donde solo quedan las letras válidas.
        letras_coords  -> información geométrica de cada letra detectada.
    """
    rh, rw = roi.shape

    # -------------------------------   -----------------------
    # 1. Binarizar la ROI
    # ------------------------------------------------------

    # Consideramos tinta todo píxel menor a 140.
    # Nuevamente:
    #     255 -> tinta
    #     0   -> fondo
    bin_roi = ((roi < 140) * 255).astype(np.uint8)

    # ------------------------------------------------------
    # 2. Buscar componentes conexas
    # ------------------------------------------------------

    # Cada conjunto de píxeles conectados puede representar una letra,
    # un pequeño ruido, etc.
    #
    # 8 -> conectividad de 8 vecinos.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_roi,
        8,
        cv2.CV_32S
    )

    # Lista donde almacenaremos la geometría de las letras válidas.
    letras_coords = []

    # Creamos una imagen completamente negra del mismo tamaño que bin_roi.
    # Sobre ella iremos copiando únicamente los componentes que cumplen
    # nuestros criterios de tamaño.
    mascara_limpia = np.zeros_like(bin_roi)

    # ------------------------------------------------------
    # 3. Filtrar componentes
    # ------------------------------------------------------

    # Umbrales adaptativos relativos a la ROI
    min_h = max(3, int(rh * 0.25))
    max_h = int(rh * 0.95)
    min_w = max(2, int(rw * 0.05))
    max_w = int(rw * 0.85)
    min_area = max(5, int((rh * rw) * 0.015))

    # El componente 0 es el fondo, por eso empezamos en 1.
    for i in range(1, num_labels):

        # Extraemos las propiedades geométricas del componente.
        #
        # x,y -> posición de la esquina superior izquierda.
        # w,h -> ancho y alto del rectángulo envolvente.
        # area -> cantidad de píxeles de tinta del componente.
        x, y, w, h, area = stats[i]

        # Estos filtros buscan componentes compatibles con una letra:
        #
        #     min_h <= h <= max_h -> altura razonable.
        #     min_w <= w <= max_w -> ancho razonable.
        #     area >= min_area   -> suficiente cantidad de píxeles.
        #
        # La idea es eliminar ruido muy pequeño o componentes que tengan
        # un tamaño claramente incompatible con una letra.
        if min_h <= h <= max_h and min_w <= w <= max_w and area >= min_area:
            # Guardamos sus coordenadas y el ID del componente.
            letras_coords.append((x, y, w, h, i))

            # labels == i crea una máscara booleana con los píxeles
            # que pertenecen exactamente a este componente.
            # Los pintamos de blanco (255) sobre mascara_limpia.
            mascara_limpia[labels == i] = 255

    # ------------------------------------------------------
    # 4. Ordenar las letras de izquierda a derecha
    # ------------------------------------------------------

    # key=lambda item: item[0] significa ordenar usando la coordenada X.
    letras_coords.sort(key=lambda item: item[0])

    # Devolvemos la máscara limpia y la lista de letras detectadas.
    return mascara_limpia, letras_coords


# ==========================================================
# PASO 4: Clasificador de las letras por huecos
# ==========================================================

def contar_huecos(crop_bin):
    """
    Cuenta cuántos huecos internos tiene una letra.

    crop_bin utiliza:
        255 = tinta
        0   = fondo

    Topológicamente esperamos:
        A -> 1 hueco
        B -> 2 huecos
        C -> 0 huecos
        D -> 1 hueco

    Por eso el conteo de huecos permite separar varios casos.
    """

    # ------------------------------------------------------
    # 1. Convertir fondo en 1 y tinta en 0
    # ------------------------------------------------------

    # crop_bin == 0 identifica los píxeles que son fondo.
    # astype(np.uint8) transforma True/False en 1/0.
    #
    # Resultado:
    #     fondo -> 1
    #     tinta -> 0
    fondo = (crop_bin == 0).astype(np.uint8)

    # ------------------------------------------------------
    # 2. Agregar un marco de fondo alrededor de la imagen
    # ------------------------------------------------------

    # Añadimos una fila arriba y abajo y una columna a izquierda y derecha.
    # Todas esas posiciones nuevas tienen valor 1, o sea, fondo.
    #
    # ¿Por qué?
    # Porque así garantizamos que exista una región de fondo exterior
    # conectada al borde de la imagen.
    #
    # De esta forma podemos distinguir:
    #     fondo exterior -> no es un hueco
    #     fondo completamente encerrado por la letra -> sí es un hueco
    fondo = cv2.copyMakeBorder(
        fondo,
        1, 1, 1, 1,
        cv2.BORDER_CONSTANT,
        value=1
    )

    # ------------------------------------------------------
    # 3. Componentes conexas del fondo
    # ------------------------------------------------------

    # Buscamos componentes conexas usando conectividad 4.
    #
    # Con conectividad 4, un píxel se conecta solamente con:
    #     arriba
    #     abajo
    #     izquierda
    #     derecha
    #
    # Esto evita que un hueco se conecte accidentalmente con el exterior
    # solamente a través de un pequeño contacto diagonal.
    num, labels = cv2.connectedComponents(
        fondo,
        connectivity=4
    )

    # labels[0,0] corresponde al componente que ocupa la esquina superior
    # izquierda del marco. Ese componente representa el fondo exterior.
    exterior = labels[0, 0]

    # np.unique(labels) obtiene todos los IDs de componentes presentes.
    #
    # Quitamos:
    #     0         -> etiqueta reservada / componente que no corresponde
    #     exterior  -> el fondo exterior
    #
    # Lo que queda son regiones de fondo encerradas dentro de la letra.
    # Cada una de esas regiones se interpreta como un hueco.
    huecos = set(np.unique(labels)) - {0, exterior}

    # La cantidad de huecos es simplemente la cantidad de IDs encontrados.
    return len(huecos)


def clasificar_letra(crop_bin):
    """
    Clasifica una letra como A, B, C o D.

    Se hace en dos pasos:

        1. Topología: contar cantidad de huecos.
           C -> 0
           A/D -> 1
           B -> 2

        2. Si hay un solo hueco, distinguir A de D usando el borde izquierdo.
           La D posee una barra vertical izquierda mucho más llena que la A.

    Entrada:
        crop_bin -> imagen binaria de una sola letra.
    """

    # Obtenemos alto y ancho de la letra recortada.
    h, w = crop_bin.shape

    # ------------------------------------------------------
    # 1. Clasificación por cantidad de huecos
    # ------------------------------------------------------

    huecos = contar_huecos(crop_bin)

    # B normalmente contiene dos regiones cerradas.
    # Si hay 2 o más, clasificamos como B.
    if huecos >= 2:
        return 'B'

    # C no tiene un hueco cerrado.
    if huecos == 0:
        return 'C'

    # Si llegamos acá, huecos == 1.
    # Las candidatas son A y D.

    # ------------------------------------------------------
    # 2. Distinguir A de D mediante el borde izquierdo
    # ------------------------------------------------------

    # Tomamos aproximadamente la sexta parte del ancho de la letra.
    # max(2, ...) garantiza un mínimo de 2 columnas.
    ancho_borde = max(2, w // 6)

    # Extraemos esa franja izquierda de la imagen.
    borde_izq = crop_bin[:, :ancho_borde]

    # Para cada fila preguntamos si hay al menos un píxel de tinta.
    # np.any(..., axis=1) produce un True por cada fila que contenga
    # alguna parte de la letra.
    #
    # np.mean(...) calcula la proporción de filas ocupadas.
    #
    # Intuición:
    #     D -> su barra izquierda ocupa casi toda la altura -> valor alto.
    #     A -> la izquierda no está llena verticalmente -> valor más bajo.
    frac_izq = np.mean(
        np.any(borde_izq == 255, axis=1)
    )

    # Con un valor de 0.6 o más consideramos que el borde izquierdo
    # está suficientemente ocupado como para ser una D.
    # En caso contrario clasificamos como A.
    return 'D' if frac_izq >= 0.6 else 'A'


# ==========================================================
# PASO 5: Corrección de respuestas (Punto A)
# ==========================================================

def corregir_examen(rois):
    # Une detección, clasificación y comparación contra la clave oficial.
    """
    Corrige las 10 preguntas de un examen.

    Para cada pregunta:
        - detecta cuántas letras aparecen;
        - si no hay ninguna -> VACIA;
        - si hay más de una -> MULTIPLE;
        - si hay exactamente una -> la clasifica como A/B/C/D;
        - compara con RESPUESTAS_CORRECTAS.

    Devuelve:
        resultados -> diccionario pregunta -> (respuesta, estado)
        correctas_total -> cantidad total de respuestas correctas.
    """

    # Diccionario donde guardaremos el resultado individual de cada pregunta.
    resultados = {}

    # Contador de respuestas correctas.
    correctas_total = 0

    print("\n--- RESULTADOS DEL EXAMEN ---")

    # Recorremos las preguntas 1 a 10.
    for num_p in range(1, 11):

        # La ROI obtenida anteriormente puede contener algunos píxeles superiores
        # que no necesitamos. Por eso eliminamos algunas filas proporcionales a la roi 1px
        pad_top = max(1, int(rois[num_p].shape[0] * 0.10))
        roi_recortado = rois[num_p][pad_top:, :]

        # Limpia la ROI y devuelve:
        #     mascara -> solo componentes que parecen letras
        #     letras  -> lista con las letras detectadas
        mascara, letras = limpiar_roi_letra(roi_recortado)

        # Cantidad de letras detectadas.
        cant = len(letras)

        # --------------------------------------------------
        # Caso 1: ninguna letra
        # --------------------------------------------------
        if cant == 0:
            resp_alumno = "VACIA"
            estado = "MAL"

        # --------------------------------------------------
        # Caso 2: más de una letra
        # --------------------------------------------------
        elif cant > 1:
            resp_alumno = "MULTIPLE"
            estado = "MAL"

        # --------------------------------------------------
        # Caso 3: exactamente una letra
        # --------------------------------------------------
        else:
            # letras[0] tiene:
            #     x, y -> posición de la letra
            #     w,h -> tamaño
            #     _    -> ID de componente, que aquí no usamos.
            x, y, w, h, _ = letras[0]

            # Recortamos exactamente el rectángulo de la única letra.
            # Así el clasificador no recibe ruido de la ROI completa.
            crop_letra = mascara[y:y + h, x:x + w]

            # Clasificamos la letra mediante el número de huecos y,
            # si hace falta, la ocupación de su borde izquierdo.
            resp_alumno = clasificar_letra(crop_letra)

            # Comparamos la respuesta detectada contra la clave oficial.
            if resp_alumno == RESPUESTAS_CORRECTAS[num_p]:
                estado = "OK"

                # Sumamos un acierto.
                correctas_total += 1
            else:
                estado = "MAL"

        # Guardamos para esa pregunta:
        #     respuesta detectada
        #     estado de corrección
        resultados[num_p] = (resp_alumno, estado)

        # Mostramos el detalle de la pregunta en consola.
        print(
            f"Pregunta {num_p:2d}: {estado} "
            f"(Respuesta: {resp_alumno} | "
            f"Correcta: {RESPUESTAS_CORRECTAS[num_p]})"
        )

    # Al terminar las 10 preguntas, mostramos la cantidad de aciertos.
    print(f"\nTotal Aciertos: {correctas_total}/10")

    # La consigna considera aprobado a partir de 6 aciertos.
    print(
        "Condición:",
        "APROBADO" if correctas_total >= 6 else "DESAPROBADO"
    )

    return resultados, correctas_total


# ==========================================================
# PASO 6: Extraer y validar campos del encabezado
# ==========================================================

def extraer_campos_encabezado(img_gray, y_coords, x_coords):
    """
    Extrae las tres zonas del encabezado:
        - Name
        - Date
        - Class

    Los recortes se definen por proporciones del ancho total de la imagen.
    De esta manera evitamos incluir las etiquetas impresas "Name:", "Date:"
    y "Class:" y nos concentramos en la parte completada por el alumno.
    """

    # Si y_coords tiene 7 elementos, el primero corresponde al límite
    # superior adicional del encabezado de la tabla y se utiliza como base.
    if len(y_coords) >= 7:
        y_base = y_coords[0]
        # La altura de un renglón la tomamos de la primera celda
        alto_renglon = y_coords[2] - y_coords[1]
    elif len(y_coords) >= 2: 
        y_base = y_coords[0]
        alto_renglon = y_coords[1] - y_coords[0]

    # ------------------------------------------------------
    # Definir el rango vertical de los campos
    # ------------------------------------------------------

    #Rango vertical relativo a la escala del documento (~60% del alto de un renglón)
    alto_encabezado = int(alto_renglon * 0.60)
    y0 = max(0, y_base - alto_encabezado)

    # y1 es la línea base que tomamos como límite inferior del texto.
    y1 = y_base

    # Obtenemos el ancho total de la imagen.
    w = img_gray.shape[1]

    # ------------------------------------------------------
    # 1. Campo Name
    # ------------------------------------------------------

    # El recorte comienza en el 10 % del ancho y termina en el 43 %.
    # Esto busca excluir la etiqueta fija "Name:" y centrarse en lo escrito.
    crop_name = img_gray[
        y0:y1,
        int(w * 0.10):int(w * 0.43)
    ]

    # ------------------------------------------------------
    # 2. Campo Date
    # ------------------------------------------------------

    # El recorte comienza en 52 % y termina en 65 % del ancho.
    crop_date = img_gray[
        y0:y1,
        int(w * 0.52):int(w * 0.65)
    ]

    # ------------------------------------------------------
    # 3. Campo Class
    # ------------------------------------------------------

    # El recorte comienza en 75 % para evitar parte de la etiqueta "Class:"
    # y llega hasta el 95 % del ancho.
    crop_class = img_gray[
        y0:y1,
        int(w * 0.75):int(w * 0.95)
    ]

    # Devolvemos los tres recortes en un diccionario.
    return {
        'Name': crop_name,
        'Date': crop_date,
        'Class': crop_class
    }

def analizar_campo_texto(crop_img, modo='name'):
    """
    Analiza uno de los campos de texto del encabezado.
    Detecta espacios entre palabras adaptándose al espaciado real del texto.
    """
    ch, cw = crop_img.shape

    # Binarización original fija < 135
    bin_crop = ((crop_img < 135) * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_crop,
        8,
        cv2.CV_32S
    )

    caracteres = []
    # Filtro mínimo adaptativo para trazos reales
    min_h = max(3, int(ch * 0.18))
    min_area = max(3, int((ch * cw) * 0.002))

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if h >= min_h and area >= min_area:
            caracteres.append((x, y, w, h))

    caracteres.sort(key=lambda c: c[0])
    cant_caracteres = len(caracteres)

    if cant_caracteres == 0:
        return 0, 0

    # Calculamos todos los espacios horizontales entre componentes consecutivos (caracteres)
    espacios = []
    anchos = []
    for k in range(len(caracteres) - 1):
        x_fin_actual = caracteres[k][0] + caracteres[k][2]
        x_ini_siguiente = caracteres[k + 1][0]
        espacios.append(max(0, x_ini_siguiente - x_fin_actual))
        anchos.append(caracteres[k][2])
    anchos.append(caracteres[-1][2])

    if not espacios:
        return cant_caracteres, 1

    # Definición adaptativa del umbral de espacio:
    # El ancho medio de un carácter es un excelente estimador de escala invariante
    ancho_medio_char = np.median(anchos) if len(anchos) > 0 else 5.0

    if modo == 'name':
        # Un espacio entre palabras suele ser de al menos el 60% del ancho de un carácter
        # o un salto que claramente duplica el espaciado intra-palabra habitual
        espacio_base = np.median(espacios) if len(espacios) > 0 else 2.0
        umbral_espacio = max(3.0, max(espacio_base * 1.8, ancho_medio_char * 0.55))
    elif modo == 'date':
        # Para la fecha, permitimos espacios más anchos sin que cuenten como palabras separadas
        umbral_espacio = max(5.0, ancho_medio_char * 1.4)
    else:
        umbral_espacio = max(3.0, ancho_medio_char * 0.6)

    cant_palabras = 1
    for esp in espacios:
        if esp >= umbral_espacio:
            cant_palabras += 1

    return cant_caracteres, cant_palabras


def validar_encabezado(campos):
    """
    Valida las restricciones del encabezado.

    Name:
        - al menos 2 palabras
        - entre 1 y 25 caracteres

    Date:
        - exactamente 8 caracteres
        - una sola palabra

    Class:
        - exactamente 1 carácter
    """

    # Diccionario donde guardaremos el resultado de cada campo.
    res_val = {}

    print("\n--- VALIDACIÓN DE ENCABEZADO ---")

    # ======================================================
    # 1. Validar Name
    # ======================================================

    # 1. Validar Name
    chars_name, words_name = analizar_campo_texto(campos['Name'], modo='name')
    val_name = "OK" if (words_name >= 2 and 1 <= chars_name <= 25) else "MAL"
    res_val['Name'] = (val_name, chars_name, words_name)
    print(f"Name:  {val_name} ({words_name} palabra(s), {chars_name} caracteres)")

    # 2. Validar Date
    chars_date, words_date = analizar_campo_texto(campos['Date'], modo='date')
    val_date = "OK" if (chars_date == 8 and words_date == 1) else "MAL"
    res_val['Date'] = (val_date, chars_date, words_date)
    print(f"Date:  {val_date} ({words_date} palabra(s), {chars_date} caracteres)")

    # 3. Validar Class
    chars_class, words_class = analizar_campo_texto(campos['Class'], modo='class')
    val_class = "OK" if chars_class == 1 else "MAL"
    res_val['Class'] = (val_class, chars_class, words_class)
    print(f"Class: {val_class} ({chars_class} caracter(es))")

    return res_val


# ==========================================================
# PASO 7: Integración completa
# ==========================================================

# Importamos os para poder trabajar con rutas y obtener solamente
# el nombre del archivo a partir de una ruta completa.
import os


def procesar_lote_examenes(lista_rutas):
    # Orquesta el pipeline completo para cada archivo de entrada.
    """
    PIPELINE FINAL DE PROCESAMIENTO DE IMÁGENES.

    Recibe una lista de rutas de exámenes y procesa cada uno completo:
        1. Lee la imagen.
        2. Detecta la grilla.
        3. Extrae celdas y ROIs.
        4. Corrige las preguntas.
        5. Valida el encabezado.
        6. Guarda la información necesaria para el resumen final.
        7. Genera una imagen resumen para todos los alumnos.
    """

    # Lista donde acumularemos un registro por examen/alumno.
    resumen_alumnos = []

    # Recorremos cada ruta recibida.
    for ruta in lista_rutas:
        nombre_archivo = os.path.basename(ruta)

        # Separador visual para distinguir cada examen en la consola.
        print("\n" + "=" * 55)
        print(f"EVALUANDO ARCHIVO: {nombre_archivo}")
        print("=" * 55)

        # --------------------------------------------------
        # 1. Cargar la imagen
        # --------------------------------------------------

        # cv2.IMREAD_GRAYSCALE hace que OpenCV cargue la imagen
        # directamente en escala de grises, por lo que cada píxel
        # queda representado con un solo valor.
        img_gray = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

        # Si OpenCV no pudo leer la imagen, devuelve None.
        if img_gray is None:
            print(f"Error: no se pudo leer el archivo en {ruta}")
            continue

        # --------------------------------------------------
        # 2. Detección de grilla y extracción de regiones
        # --------------------------------------------------

        # Detectamos las coordenadas X e Y de la grilla.
        x_coords, y_coords = segmentar_grilla(img_gray)

        # A partir de la grilla, extraemos las 10 celdas de preguntas.
        celdas = extraer_celdas(img_gray, x_coords, y_coords)

        # A partir de cada celda, aislamos la región donde debería estar
        # la letra marcada.
        rois = obtener_rois_letras_por_componentes(celdas)

        # --------------------------------------------------
        # 3. Corregir preguntas (Punto A)
        # --------------------------------------------------

        # Corregimos las diez respuestas.
        # resultados_preguntas contiene el detalle de cada pregunta y
        # aciertos contiene solamente la cantidad total.
        resultados_preguntas, aciertos = corregir_examen(rois)

        # Se considera aprobado si se obtienen al menos 6 aciertos.
        aprobado = (aciertos >= 6)

        # --------------------------------------------------
        # 4. Validar encabezado (Punto B)
        # --------------------------------------------------

        # Extraemos Name, Date y Class.
        campos = extraer_campos_encabezado(
            img_gray,
            y_coords,
            x_coords
        )

        # Validamos las condiciones de los tres campos.
        res_encabezado = validar_encabezado(campos)

        # --------------------------------------------------
        # 5. Guardar información para el Punto D
        # --------------------------------------------------

        # Guardamos solamente la información necesaria para construir
        # posteriormente la imagen resumen.
        resumen_alumnos.append({
            # Nombre del archivo de origen.
            'archivo': nombre_archivo,

            # Copia del recorte correspondiente al campo Name.
            # copy() crea una copia independiente del array.
            'crop_name': campos['Name'].copy(),

            # Cantidad de aciertos del examen.
            'aciertos': aciertos,

            # True si tiene 6 o más aciertos; False en caso contrario.
            'aprobado': aprobado
        })

    # ==========================================================
    # PUNTO D: Generación de la imagen resumen
    # ==========================================================

    # Cantidad de exámenes procesados correctamente.
    n_examenes = len(resumen_alumnos)

    # Si no hay ninguno, no hay nada que resumir.
    if n_examenes == 0:
        return

    # Altura reservada para cada alumno dentro del mosaico final.
    alto_fila = 60

    # Ancho total del lienzo de salida.
    ancho_lienzo = 480

    # Alto total:
    #     50 -> espacio reservado para el título.
    #     n_examenes * alto_fila -> espacio de todas las filas.
    alto_total = 50 + n_examenes * alto_fila

    # ------------------------------------------------------
    # Crear lienzo blanco
    # ------------------------------------------------------

    # Creamos una matriz de:
    #     alto_total x ancho_lienzo x 3
    #
    # El último 3 representa los tres canales de una imagen BGR.
    # np.ones(...) produce unos y al multiplicar por 255 obtenemos blanco.
    mosaico = (
        np.ones(
            (alto_total, ancho_lienzo, 3),
            dtype=np.uint8
        ) * 255
    )

    # ------------------------------------------------------
    # Escribir el título
    # ------------------------------------------------------

    # putText dibuja texto directamente sobre la imagen.
    #
    # Parámetros principales:
    #     mosaico                    -> imagen donde escribimos.
    #     texto                      -> texto a dibujar.
    #     (20,30)                    -> posición inicial.
    #     FONT_HERSHEY_SIMPLEX       -> tipo de letra.
    #     0.7                        -> escala.
    #     (0,0,0)                    -> color BGR negro.
    #     2                          -> grosor.
    cv2.putText(
        mosaico,
        "RESULTADOS FINALES - ALUMNOS",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    # ------------------------------------------------------
    # Agregar una fila por alumno
    # ------------------------------------------------------

    # enumerate nos da:
    #     i   -> índice 0,1,2,...
    #     alu -> diccionario del alumno actual
    for i, alu in enumerate(resumen_alumnos):

        # Calculamos la coordenada Y donde comienza la fila actual.
        # El +50 deja debajo del título.
        y_pos = 50 + i * alto_fila

        # --------------------------------------------------
        # Preparar la imagen del nombre
        # --------------------------------------------------

        # El recorte original está en escala de grises.
        # cvtColor lo convierte de 1 canal a 3 canales BGR para poder
        # insertarlo en el lienzo, que también tiene 3 canales.
        c_bgr = cv2.cvtColor(
            alu['crop_name'],
            cv2.COLOR_GRAY2BGR
        )

        # Redimensionamos el nombre a 170x32 píxeles para que todos
        # los alumnos ocupen exactamente el mismo espacio.
        c_resized = cv2.resize(c_bgr, (170, 32))

        # Insertamos esa imagen dentro del mosaico.
        #
        # y_pos+5:y_pos+37 -> 32 píxeles de altura.
        # 15:185            -> 170 píxeles de ancho.
        mosaico[
            y_pos + 5:y_pos + 37,
            15:185
        ] = c_resized

        # --------------------------------------------------
        # Elegir texto y color según condición
        # --------------------------------------------------

        if alu['aprobado']:
            # En OpenCV los colores están en orden BGR.
            # (0,150,0) representa verde.
            color = (0, 150, 0)

            # Mostramos APROBADO y la cantidad de aciertos.
            texto = f"APROBADO  ({alu['aciertos']}/10)"
        else:
            # (0,0,220) representa rojo en BGR.
            color = (0, 0, 220)

            # Mostramos DESAPROBADO y la cantidad de aciertos.
            texto = f"DESAPROBADO ({alu['aciertos']}/10)"

        # --------------------------------------------------
        # Dibujar marco, resultado y separador
        # --------------------------------------------------

        # Marco rectangular alrededor del nombre.
        cv2.rectangle(
            mosaico,
            (13, y_pos + 3),
            (187, y_pos + 39),
            color,
            2
        )

        # Escribimos el resultado a la derecha del nombre.
        cv2.putText(
            mosaico,
            texto,
            (205, y_pos + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        # Línea horizontal gris para separar visualmente a los alumnos.
        cv2.line(
            mosaico,
            (10, y_pos + alto_fila - 2),
            (ancho_lienzo - 10, y_pos + alto_fila - 2),
            (220, 220, 220),
            1
        )

    # ------------------------------------------------------
    # Mostrar imagen del Punto D
    # ------------------------------------------------------

    # Creamos una figura cuyo tamaño vertical depende de la cantidad
    # de exámenes procesados.
    plt.figure(figsize=(8, 2.3 * n_examenes))

    # OpenCV trabaja normalmente en BGR mientras que matplotlib espera RGB.
    # Por eso convertimos antes de mostrar.
    plt.imshow(
        cv2.cvtColor(mosaico, cv2.COLOR_BGR2RGB)
    )

    plt.title(
        "Punto d: Resumen de Aprobados y Desaprobados",
        fontsize=13
    )

    # Desactivamos los ejes porque la imagen final es un resultado visual,
    # no un gráfico donde nos interesen las coordenadas.
    plt.axis("off")

    # Mostramos la figura.
    plt.show()

    # ------------------------------------------------------
    # Guardar entregable
    # ------------------------------------------------------

    # Guardamos el mosaico final como archivo PNG.
    # El resultado queda en el directorio de trabajo actual.
    cv2.imwrite("resumen_aprobados.png", mosaico)

    print(
        "\nImagen resumen generada y guardada como 'resumen_aprobados.png'."
    )




##DEBUGGING


# ==========================================================
# EJECUCIÓN Y GRÁFICOS PASO A PASO PARA DEBUGGEAR
# ==========================================================

# ----------------------------------------------------------
# 1. Carga de una imagen para inspeccionar el funcionamiento
# ----------------------------------------------------------

# Intentamos leer examen_2.png usando la carpeta TP1.
#
# IMREAD_GRAYSCALE hace que la imagen se cargue directamente en escala de grises.
img = cv2.imread(
    'TP1/examen_2.png',
    cv2.IMREAD_GRAYSCALE
)

# Si la ruta anterior no existe o no pudo abrirse, intentamos una ruta alternativa
# donde la imagen esté directamente en el directorio actual.
if img is None:
    img = cv2.imread(
        'examen_3.png',
        cv2.IMREAD_GRAYSCALE
    )

# ----------------------------------------------------------
# 2. Detectar grilla y visualizar líneas
# ----------------------------------------------------------

# Ejecutamos el primer paso del pipeline.
# Obtenemos:
#     x_coords -> líneas verticales
#     y_coords -> líneas horizontales
x_coords, y_coords = segmentar_grilla(img)

# Estas dos líneas están comentadas para no imprimir las coordenadas
# automáticamente. Se pueden descomentar durante el debugging.
#print("Coordenadas X (4 columnas):", x_coords)
#print(f"Coordenadas Y ({len(y_coords)} líneas horizontales):", y_coords)

# ----------------------------------------------------------
# Crear una copia coloreada de la imagen para dibujar las líneas
# ----------------------------------------------------------

# La imagen original está en escala de grises (1 canal).
# Para dibujar líneas de distintos colores necesitamos una imagen BGR (3 canales).
vis_lineas = cv2.cvtColor(
    img,
    cv2.COLOR_GRAY2BGR
)

# Dibujamos una línea vertical roja para cada coordenada X encontrada.
for x in x_coords:
    cv2.line(
        vis_lineas,
        (x, 0),
        (x, vis_lineas.shape[0]),
        (0, 0, 255),
        2
    )

# Dibujamos una línea horizontal azul para cada coordenada Y encontrada.
for y in y_coords:
    cv2.line(
        vis_lineas,
        (0, y),
        (vis_lineas.shape[1], y),
        (255, 0, 0),
        2
    )

# ----------------------------------------------------------
# Mostrar visualmente el resultado de la segmentación
# ----------------------------------------------------------

plt.figure(figsize=(7, 9))

# Convertimos BGR -> RGB para que matplotlib interprete correctamente
# los colores.
plt.imshow(
    cv2.cvtColor(vis_lineas, cv2.COLOR_BGR2RGB)
)

plt.title("Paso 1: Detección de Grilla")
plt.axis("off")
plt.show()

# ----------------------------------------------------------
# 3. Extraer celdas y ROIs de respuestas
# ----------------------------------------------------------

# Utilizamos las coordenadas de la grilla para recortar las 10 preguntas.
celdas = extraer_celdas(
    img,
    x_coords,
    y_coords
)

# A partir de las celdas, localizamos la región donde debería encontrarse
# la letra marcada en cada pregunta.
rois = obtener_rois_letras_por_componentes(celdas)

# ----------------------------------------------------------
# Analizar individualmente cada pregunta durante el debugging
# ----------------------------------------------------------

# Recorremos las 10 preguntas.
for num_p in range(1, 11):

    # Eliminamos las primeras dos filas de la ROI para centrarnos en la zona útil.
    roi_recortado = rois[num_p][2:, :]

    # Limpiamos la ROI para conservar solamente los componentes que parecen letras.
    mascara, letras = limpiar_roi_letra(roi_recortado)

    # Solo si encontramos exactamente una letra hacemos el análisis geométrico.
    if len(letras) == 1:

        # Recuperamos las coordenadas de la única letra detectada.
        x, y, w, h, _ = letras[0]

        # Recortamos únicamente esa letra desde la máscara.
        crop = mascara[y:y + h, x:x + w]

        # Calculamos el ancho de la franja izquierda utilizada por el clasificador
        # A/D. Se mantiene aquí como información de debugging.
        ancho = max(2, w // 6)

        # Calculamos qué proporción de filas tiene tinta en esa franja izquierda.
        # Es conceptualmente la misma característica utilizada para decidir
        # si una letra con un hueco es A o D.
        frac = np.mean(
            np.any(crop[:, :ancho] == 255, axis=1)
        )

        # Esta impresión está comentada para que el bloque no llene la consola.
        # Se puede descomentar para observar:
        #     - cantidad de huecos
        #     - ocupación izquierda
        #     - letra clasificada
        #print(
        #    f"P{num_p}: huecos={contar_huecos(crop)} "
        #    f"frac_izq={frac:.2f} -> {clasificar_letra(crop)}"
        #)

# Si no se detectó exactamente una letra, estas líneas también podrían
# utilizarse durante el debugging para saber cuántos componentes aparecieron.
#    else:
#        print(f"P{num_p}: {len(letras)} componentes (VACIA/MULTIPLE)")

# ----------------------------------------------------------
# Visualizar las 10 ROIs en una grilla 2x5
# ----------------------------------------------------------

# Creamos 10 subgráficos organizados en 2 filas y 5 columnas.
fig, axes = plt.subplots(
    2,
    5,
    figsize=(13, 4)
)

# zip asocia cada número de pregunta con un eje del gráfico.
for num_p, ax in zip(range(1, 11), axes.flatten()):

    # Mostramos la ROI original en escala de grises.
    ax.imshow(
        rois[num_p],
        cmap='gray'
    )

    # Título con número de pregunta.
    ax.set_title(f"Pregunta {num_p}")

    # Quitamos las marcas de los ejes.
    ax.axis("off")

# Título general del conjunto de gráficos.
plt.suptitle(
    "Paso 2: ROIs de Respuestas en escala de grises",
    fontsize=12
)

# Ajustamos automáticamente los espacios para evitar solapamientos.
plt.tight_layout()

# Mostramos la figura.
plt.show()

# ----------------------------------------------------------
# 4. Visualizar máscaras binarias limpias
# ----------------------------------------------------------

# Otra figura 2x5, ahora para observar qué componentes sobreviven
# después de aplicar los filtros de tamaño.
fig, axes = plt.subplots(
    2,
    5,
    figsize=(13, 4)
)

# Esta impresión queda comentada para no mostrarla por defecto.
#print("\n--- DETECCIÓN DE COMPONENTES POR PREGUNTA ---")

# Analizamos las 10 preguntas.
for num_p, ax in zip(range(1, 11), axes.flatten()):

    # Quitamos las primeras dos filas, igual que en corregir_examen().
    roi_recortado = rois[num_p][2:, :]

    # Limpiamos la ROI y obtenemos las letras detectadas.
    mascara, letras = limpiar_roi_letra(roi_recortado)

    # Esta impresión también queda comentada para no saturar la consola.
    #print(f"Pregunta {num_p}: {len(letras)} letra(s) detectada(s)")

    # Mostramos la máscara limpia.
    ax.imshow(
        mascara,
        cmap='gray'
    )

    # En el título indicamos cuántas letras fueron detectadas.
    ax.set_title(
        f"P{num_p} ({len(letras)} letras)"
    )

    # Ocultamos los ejes.
    ax.axis("off")

# Título general.
plt.suptitle(
    "Paso 3: Máscaras binarias limpias (Fondo Negro)",
    fontsize=12
)

# Ajustamos el layout.
plt.tight_layout()

# Mostramos la figura.
plt.show()

# ----------------------------------------------------------
# 5. Corrección automática final del examen individual
# ----------------------------------------------------------

# Ejecutamos la función que corrige las 10 preguntas.
#
# resultados -> detalle de cada pregunta.
# total      -> cantidad de aciertos.
resultados, total = corregir_examen(rois)


# ----------------------------------------------------------
# 6. Validación del encabezado
# ----------------------------------------------------------

# Extraemos las tres zonas del encabezado.
campos = extraer_campos_encabezado(
    img,
    y_coords,
    x_coords
)

# ----------------------------------------------------------
# Visualizar los tres campos
# ----------------------------------------------------------

# Creamos tres gráficos en una sola fila.
fig, axes = plt.subplots(
    1,
    3,
    figsize=(11, 2)
)

# Campo Name.
axes[0].imshow(
    campos['Name'],
    cmap='gray'
)
axes[0].set_title("Campo Name")
axes[0].axis("off")

# Campo Date.
axes[1].imshow(
    campos['Date'],
    cmap='gray'
)
axes[1].set_title("Campo Date")
axes[1].axis("off")

# Campo Class.
axes[2].imshow(
    campos['Class'],
    cmap='gray'
)
axes[2].set_title("Campo Class")
axes[2].axis("off")

# Título general.
plt.suptitle(
    "Paso 6: Recortes del Encabezado",
    fontsize=12
)

# Ajustamos espacios.
plt.tight_layout()

# Mostramos los tres recortes.
plt.show()

# ----------------------------------------------------------
# Validar restricciones del encabezado
# ----------------------------------------------------------

# Ejecutamos la función que comprueba Name, Date y Class.
estado_encabezado = validar_encabezado(campos)


# ==========================================================
# FUNCIÓN FINAL
# ==========================================================

# Lista de rutas de todos los exámenes que queremos procesar.
archivos = [
    'TP1/examen_1.png',
    'TP1/examen_2.png',
    'TP1/examen_3.png',
    'TP1/examen_4.png',
    'TP1/examen_5.png'
]

# Ejecutamos el pipeline completo sobre todos los archivos.
# Dentro de esta función se generan:
#     - correcciones por pregunta
#     - validación del encabezado
#     - imagen resumen final
procesar_lote_examenes(archivos)
