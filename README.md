# Procesamiento de Imágenes - Trabajo Práctico N° 1

Repositorio con las soluciones implementadas en Python y OpenCV para el Trabajo Práctico N° 1 de Procesamiento de Imágenes (UA-LCD).

## Estructura del Repositorio

```text
.
├── TP1/
│   ├── Imagen_con_detalles_escondidos.tif
│   ├── examen_1.png
│   ├── examen_2.png
│   ├── examen_3.png
│   ├── examen_4.png
│   └── examen_5.png
├── Problema1.py
├── Problema2_documentado.py
├── requirements.txt
└── README.md
```

## Requisitos Previos e Instalación

Se recomienda utilizar un entorno virtual con **Python 3.10+**.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Maximo-Verdondoni/Procesamiento_Imagenes1.git
cd Procesamiento_Imagenes1
```

### 2. Crear y activar el entorno virtual

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### Dependencias

El archivo `requirements.txt` contiene:

```text
opencv-python
numpy
matplotlib
```

## Ejecución de los Scripts

### Problema 1: Ecualización Local del Histograma

Implementa el algoritmo de ecualización local del histograma píxel a píxel mediante una ventana deslizante de tamaño $M \times N$, con extensión de bordes por replicación (`cv2.BORDER_REPLICATE`).

El objetivo es revelar patrones ocultos en distintas áreas de bajo contraste y comparar el resultado frente a la ecualización global (`cv2.equalizeHist`).

Se utilizan las siguientes ventanas:

- $3 \times 3$
- $11 \times 11$
- $31 \times 31$
- $51 \times 51$
- $61 \times 61$
- $81 \times 81$

**Comando de ejecución:**

```bash
python Problema1.py
```

**Salidas generadas:**

- Progreso del cómputo para cada ventana en la terminal.
- Archivo `resultado_ecualizacion_local.png` con la grilla comparativa de 2x4.

### Problema 2: Corrección Automática de Exámenes Multiple Choice

Pipeline de visión por computadora para corregir exámenes escaneados sin depender de coordenadas fijas a priori.

#### 1. Segmentación de grilla

Se utilizan proyecciones acumuladas por filas y columnas (`np.sum`) con umbrales relativos a la resolución de la imagen para detectar automáticamente las líneas principales de la grilla.

#### 2. Extracción de celdas

A partir de la grilla detectada se recortan automáticamente las 10 preguntas, organizadas en 2 columnas de 5 renglones.

#### 3. Localización de respuestas (ROI)

Se detecta la línea de respuesta `_______` mediante componentes conexas (`cv2.connectedComponentsWithStats`) y, a partir de ella, se obtiene la región de interés (ROI) correspondiente a la respuesta.

#### 4. Clasificación morfológica y topológica

Se analiza la forma de las respuestas para determinar la opción seleccionada:

- Detección de respuestas vacías o múltiples, ambas clasificadas como `MAL`.
- Conteo de huecos topológicos sobre el fondo complementario utilizando conectividad 4:
  - `0` huecos → `C`
  - `2` huecos → `B`
- Para las letras con `1` hueco se utiliza la densidad del borde izquierdo vertical para diferenciar:
  - `A`
  - `D`

#### 5. Validación del encabezado

Se valida automáticamente la información del encabezado mediante componentes conexas y cálculo dinámico del espaciado.

Se comprueban los siguientes campos:

- **Name:** $\geq 2$ palabras y $\leq 25$ caracteres.
- **Date:** 8 caracteres y 1 palabra.
- **Class:** 1 carácter.

#### 6. Mosaico resumen

Se genera un tablero resumen utilizando el recorte original del nombre (`crop_name`) de cada estudiante.

El resultado indica el estado de aprobación:

- **Aprobado:** $\geq 6$ aciertos.
- **Desaprobado:** menos de 6 aciertos.

**Comando de ejecución:**

```bash
python Problema2_documentado.py
```

**Salidas generadas:**

- Reporte detallado por consola, pregunta por pregunta:
  - `OK` / `MAL`
  - `APROBADO` / `DESAPROBADO`
  - Resultado de la validación de encabezados.
- Procesamiento del lote completo (`examen_1.png` a `examen_5.png`).
- Ventanas interactivas de inspección visual:
  - Detección de líneas de grilla.
  - ROIs.
  - Máscaras binarias.
  - Campos del encabezado.
- Archivo `resumen_aprobados.png` guardado en la raíz del repositorio con el consolidado del lote de exámenes.
