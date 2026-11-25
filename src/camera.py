"""Read the iPhone video stream, correct lens distortion and show the FPS."""
import time

import cv2

from undistortion import build_undistort_maps, load_calibration, undistort_frame

# Video stream served by the DroidCam app on the iPhone.
# The IP is shown in the app: update it if you change Wi-Fi network.
CAMERA_URL = "http://172.20.10.1:4747/video"


def main():
    K, dist, rms = load_calibration()
    print(f"Calibration loaded (RMS = {rms:.3f} px)")

    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"Could not open the camera at {CAMERA_URL}")
        return

    map1 = map2 = None
    undistort_enabled = True

    prev_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame")
            break

        # With the first frame we know the resolution: build the maps only once
        if map1 is None:
            height, width = frame.shape[:2]
            print("Resolution (height, width, channels):", frame.shape)
            map1, map2 = build_undistort_maps(K, dist, width, height)

        if undistort_enabled:
            frame = undistort_frame(frame, map1, map2)

        # FPS (smoothed so the number does not jump around)
        now = time.perf_counter()
        instant_fps = 1.0 / (now - prev_time)
        prev_time = now
        fps = 0.9 * fps + 0.1 * instant_fps

        mode = "Undistorted" if undistort_enabled else "Original"
        cv2.putText(frame, f"FPS: {fps:.1f}  |  {mode} (d to toggle)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

        # 'q' quits, 'd' toggles the distortion correction
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            undistort_enabled = not undistort_enabled

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()