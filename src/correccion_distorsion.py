from pathlib import Path

import cv2
import numpy as np

# Resolución de las fotos con las que se hizo la calibración (iPhone, formato 4:3)
ANCHO_CALIB, ALTO_CALIB = 4032, 3024

# calibracion.npz está en la carpeta raíz del proyecto (una carpeta por encima de src)
RUTA_CALIBRACION = Path(__file__).resolve().parent.parent / "calibracion.npz"


def cargar_calibracion(ruta=RUTA_CALIBRACION):
    """Carga la matriz K, los coeficientes de distorsión y el RMS de la calibración."""
    datos = np.load(ruta)
    return datos["K"], datos["dist"], float(datos["rms"])


def adaptar_K(K, ancho, alto):
    """Adapta K de la resolución de las fotos (4032x3024) a la del vídeo.

    El vídeo 16:9 es la imagen 4:3 escalada al mismo ancho y recortada
    por arriba y por abajo, así que escalamos fx, fy, cx, cy y desplazamos cy.
    """
    s = ancho / ANCHO_CALIB
    K_video = K.astype(np.float64).copy()
    K_video[0, 0] *= s  # fx
    K_video[1, 1] *= s  # fy
    K_video[0, 2] *= s  # cx
    K_video[1, 2] *= s  # cy

    # Píxeles que se recortan arriba (y abajo) al pasar de 4:3 a 16:9
    recorte = (ALTO_CALIB * s - alto) / 2
    K_video[1, 2] -= recorte
    return K_video


def crear_mapas(K, dist, ancho, alto):
    """Calcula UNA sola vez los mapas de corrección para la resolución del vídeo."""
    K_video = adaptar_K(K, ancho, alto)
    mapa1, mapa2 = cv2.initUndistortRectifyMap(
        K_video, dist, None, K_video, (ancho, alto), cv2.CV_16SC2
    )
    return mapa1, mapa2


def corregir(frame, mapa1, mapa2):
    """Aplica la corrección de distorsión a un fotograma (rápido, para tiempo real)."""
    return cv2.remap(frame, mapa1, mapa2, cv2.INTER_LINEAR)