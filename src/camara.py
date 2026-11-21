import time
import cv2

# Dirección del vídeo que emite la app DroidCam del iPhone.
# La IP sale en la pantalla de la app: si cambias de WiFi, cámbiala aquí.
URL_CAMARA = "http://192.168.68.100:4747/video"


def main():
    cap = cv2.VideoCapture(URL_CAMARA)
    if not cap.isOpened():
        print(f"No se ha podido abrir la cámara en {URL_CAMARA}")
        return

    t_anterior = time.perf_counter()
    fps = 0.0
    primera = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se ha podido leer el fotograma")
            break

        # Imprimimos la resolución solo una vez
        if primera:
            print("Resolución (alto, ancho, canales):", frame.shape)
            primera = False

        # Cálculo de FPS (suavizado para que el número no salte tanto)
        t_actual = time.perf_counter()
        fps_instantaneo = 1.0 / (t_actual - t_anterior)
        t_anterior = t_actual
        fps = 0.9 * fps + 0.1 * fps_instantaneo

        # Escribimos los FPS en la esquina superior izquierda
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camara", frame)

        # Pulsar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()