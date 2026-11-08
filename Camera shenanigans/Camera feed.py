import cv2
import numpy as np

# =========================
# CAMERA SETTINGS
# =========================
camera_index = 1
width = 1280
height = 800
fps = 120

# =========================
# ARUCO SETTINGS
# =========================
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

params = cv2.aruco.DetectorParameters()

params.minMarkerPerimeterRate = 0.01
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 35
params.adaptiveThreshWinSizeStep = 4

params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.maxErroneousBitsInBorderRate = 0.5

detector = cv2.aruco.ArucoDetector(dictionary, params)

# =========================
# CAMERA INITIALIZATION
# =========================
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open camera 1.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

print(
    "Resolution:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    "x",
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# =========================
# TRACKING SETTINGS
# =========================
fixed_ids = [0, 1, 2]
moving_id = 3

samples_required = 30

fixed_samples = {i: [] for i in fixed_ids}
fixed_positions = {}

window_name = "Tow Tank Tracking"

print("\n========== ARUCO TRACKING ==========")
print("IDs 0, 1, 2: Fixed reference markers")
print("ID 3: Moving marker")
print(f"Collecting {samples_required} samples per fixed marker")
print()
print("R: Reset fixed marker calibration")
print("Q: Quit")
print("X: Close window and stop program")
print("====================================\n")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# =========================
# MAIN LOOP
# =========================
try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera.")
            break

        if frame.ndim == 2:
            gray = frame
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()

        # =========================
        # DETECT MARKERS
        # =========================
        corners, ids, _ = detector.detectMarkers(gray)

        moving_position = None

        if ids is not None:

            for marker_corners, marker_id in zip(
                corners, ids.flatten()
            ):

                marker_id = int(marker_id)

                if marker_id not in fixed_ids + [moving_id]:
                    continue

                pts = marker_corners[0]
                center = pts.mean(axis=0)

                cx = float(center[0])
                cy = float(center[1])

                # =========================
                # FIXED MARKERS
                # =========================
                if marker_id in fixed_ids:

                    if marker_id not in fixed_positions:

                        fixed_samples[marker_id].append(center.copy())

                        if len(fixed_samples[marker_id]) >= samples_required:

                            average = np.mean(
                                fixed_samples[marker_id],
                                axis=0
                            )

                            fixed_positions[marker_id] = average

                            print(
                                f"ID {marker_id} locked at "
                                f"({average[0]:.2f}, {average[1]:.2f})"
                            )

                # =========================
                # MOVING MARKER
                # =========================
                elif marker_id == moving_id:

                    moving_position = center

        # =========================
        # DRAW FIXED MARKERS
        # =========================
        for marker_id in fixed_ids:

            if marker_id in fixed_positions:

                cx, cy = fixed_positions[marker_id]

                cv2.circle(
                    display,
                    (round(cx), round(cy)),
                    6,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    display,
                    f"ID {marker_id} FIXED ({cx:.1f}, {cy:.1f})",
                    (round(cx) + 10, round(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            else:

                progress = len(fixed_samples[marker_id])

                cv2.putText(
                    display,
                    f"ID {marker_id}: {progress}/{samples_required}",
                    (20, 40 + marker_id * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

        # =========================
        # DRAW MOVING MARKER
        # =========================
        if moving_position is not None:

            cx, cy = moving_position

            cv2.circle(
                display,
                (round(cx), round(cy)),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                f"ID 3 MOVING ({cx:.1f}, {cy:.1f})",
                (round(cx) + 10, round(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        # =========================
        # STATUS
        # =========================
        locked_count = len(fixed_positions)

        cv2.putText(
            display,
            f"Fixed references: {locked_count}/3",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # =========================
        # DISPLAY
        # =========================
        cv2.imshow(window_name, display)

        # Process window events
        key = cv2.waitKey(1) & 0xFF

        # Stop if window is closed
        if cv2.getWindowProperty(
            window_name, cv2.WND_PROP_VISIBLE
        ) < 1:
            print("Window closed. Stopping program.")
            break

        # Reset fixed marker calibration
        if key == ord("r"):

            fixed_samples = {i: [] for i in fixed_ids}
            fixed_positions = {}

            print("\nFixed marker calibration reset.")

        # Quit
        elif key == ord("q"):
            break

finally:
    # =========================
    # CLEANUP
    # =========================
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Program stopped.")