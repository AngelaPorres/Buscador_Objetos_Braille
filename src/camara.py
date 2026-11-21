import time
import cv2

from correccion_distorsion import cargar_calibracion, crear_mapas, corregir

# Dirección del vídeo que emite la app DroidCam del iPhone.
# La IP sale en la pantalla de la app: si cambias de WiFi, cámbiala aquí.
URL_CAMARA = "http://192.168.68.100:4747/video"


def main():
    K, dist, rms = cargar_calibracion()
    print(f"Calibración cargada (RMS = {rms:.3f} px)")

    cap = cv2.VideoCapture(URL_CAMARA)
    if not cap.isOpened():
        print(f"No se ha podido abrir la cámara en {URL_CAMARA}")
        return

    mapa1 = mapa2 = None
    corregir_activo = True

    t_anterior = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se ha podido leer el fotograma")
            break

        # Con el primer fotograma ya sabemos la resolución: calculamos los mapas una sola vez
        if mapa1 is None:
            alto, ancho = frame.shape[:2]
            print("Resolución (alto, ancho, canales):", frame.shape)
            mapa1, mapa2 = crear_mapas(K, dist, ancho, alto)

        if corregir_activo:
            frame = corregir(frame, mapa1, mapa2)

        # Cálculo de FPS (suavizado para que el número no salte tanto)
        t_actual = time.perf_counter()
        fps_instantaneo = 1.0 / (t_actual - t_anterior)
        t_anterior = t_actual
        fps = 0.9 * fps + 0.1 * fps_instantaneo

        texto = "Corregida" if corregir_activo else "Original"
        cv2.putText(frame, f"FPS: {fps:.1f}  |  {texto} (d para cambiar)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camara", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("d"):
            corregir_activo = not corregir_activo

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()