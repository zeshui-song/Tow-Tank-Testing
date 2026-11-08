
import cv2
import numpy as np
import os

# =========================
# CAMERA SETTINGS
# =========================
camera_index = 1
width = 1280
height = 800
fps = 120

output_dir = r"C:\Users\zsong\Desktop\Tow Tank Testing\Ar+Ch Uco Markers"
os.makedirs(output_dir, exist_ok=True)

calibration_file = os.path.join(output_dir, "camera_calibration.npz")

# =========================
# CHARUCO BOARD SETTINGS
# =========================
squares_x = 16
squares_y = 10
square_size_mm = 25
marker_size_mm = 18

min_corners = 8
min_samples = 15

# Ignore corners on the seam between the printed sheets
exclude_seam = True
seam_x_mm = 200
seam_margin_mm = 12

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_250
)

params = cv2.aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.01
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

legacy_pattern = False

def create_detector():
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size_mm,
        marker_size_mm,
        dictionary
    )

    board.setLegacyPattern(legacy_pattern)

    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        params
    )

    return board, detector

board, detector = create_detector()
marker_detector = cv2.aruco.ArucoDetector(dictionary, params)

# =========================
# CAMERA INITIALIZATION
# =========================
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open camera 1.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read camera.")

height, width = frame.shape[:2]
image_size = (width, height)

# =========================
# CALIBRATION STORAGE
# =========================
object_points = []
image_points = []

camera_matrix = None
dist_coeffs = None
map1 = None
map2 = None

show_corrected = False

# No ROI selected initially
roi = None

window_name = "ChArUco Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# =========================
# INSTRUCTIONS
# =========================
print("\n========== CAMERA CALIBRATION ==========")
print("Board: 16 x 10 squares")
print("Square size: 25 mm")
print("Marker size: 18 mm")
print("Dictionary: DICT_4X4_250")
print()
print("R     : Select board region")
print("L     : Toggle legacy pattern")
print("SPACE : Capture calibration sample")
print("C     : Calculate calibration")
print("U     : Toggle corrected preview")
print("Q     : Quit")
print()
print("1. Hold the board in front of the camera.")
print("2. Press R and select the entire board.")
print("3. Check that ChArUco corners are detected.")
print("4. If corners remain zero, press L.")
print("5. Press SPACE to capture.")
print("6. Move the board and press R again.")
print("7. Capture at least 15 varied samples.")
print("8. Press C to calibrate.")
print("========================================\n")

# =========================
# MAIN LOOP
# =========================
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

    marker_count = 0
    corner_count = 0
    valid_obj = None
    valid_img = None

    # =========================
    # DETECT ONLY INSIDE ROI
    # =========================
    if roi is not None:

        x, y, w, h = roi
        cropped = gray[y:y+h, x:x+w]

        if cropped.size > 0:

            # Detect individual markers for diagnostics
            marker_corners, marker_ids, _ = (
                marker_detector.detectMarkers(cropped)
            )

            marker_count = (
                len(marker_ids)
                if marker_ids is not None else 0
            )

            # Draw marker boundaries in full image
            if marker_ids is not None:
                shifted_marker_corners = []

                for corners in marker_corners:
                    shifted = corners.copy()
                    shifted[:, :, 0] += x
                    shifted[:, :, 1] += y
                    shifted_marker_corners.append(shifted)

                cv2.aruco.drawDetectedMarkers(
                    display,
                    shifted_marker_corners,
                    marker_ids
                )

            # Detect ChArUco intersections
            try:
                cc, ci, _, _ = detector.detectBoard(cropped)

                if (
                    cc is not None
                    and ci is not None
                    and len(cc) > 0
                    and len(cc) == len(ci)
                ):

                    # Translate ROI coordinates to full image
                    cc = np.asarray(
                        cc, dtype=np.float32
                    ).reshape(-1, 1, 2)

                    cc[:, :, 0] += x
                    cc[:, :, 1] += y

                    ci = np.asarray(
                        ci, dtype=np.int32
                    ).reshape(-1, 1)

                    obj, img = board.matchImagePoints(cc, ci)

                    if obj is not None and img is not None:

                        obj = np.asarray(
                            obj, dtype=np.float32
                        ).reshape(-1, 1, 3)

                        img = np.asarray(
                            img, dtype=np.float32
                        ).reshape(-1, 1, 2)

                        # Remove corners near physical sheet seam
                        if exclude_seam:
                            mask = (
                                np.abs(obj[:, 0, 0] - seam_x_mm)
                                > seam_margin_mm
                            )

                            obj = obj[mask]
                            img = img[mask]

                        if len(obj) == len(img) and len(obj) > 0:

                            valid_obj = obj
                            valid_img = img
                            corner_count = len(img)

                            # Draw accepted calibration corners
                            for point in img.reshape(-1, 2):
                                px, py = np.round(point).astype(int)

                                cv2.circle(
                                    display,
                                    (px, py),
                                    3,
                                    (0, 255, 0),
                                    -1
                                )

            except cv2.error as error:
                print("Detection error:", error)

        # Draw selected ROI
        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

    # =========================
    # DISPLAY STATUS
    # =========================
    cv2.putText(
        display,
        f"ArUco markers: {marker_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    cv2.putText(
        display,
        f"ChArUco corners: {corner_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    cv2.putText(
        display,
        f"Samples: {len(object_points)}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    cv2.putText(
        display,
        f"Legacy pattern: {legacy_pattern}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
    )

    if corner_count >= min_corners:
        status = "READY - PRESS SPACE"
        color = (0, 255, 0)
    elif roi is None:
        status = "PRESS R TO SELECT BOARD"
        color = (0, 255, 255)
    else:
        status = "NO CORNERS - TRY L OR RESELECT ROI"
        color = (0, 0, 255)

    cv2.putText(
        display,
        status,
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, color, 2
    )

    # =========================
    # PREVIEW
    # =========================
    if show_corrected and map1 is not None:

        corrected = cv2.remap(
            frame, map1, map2, cv2.INTER_LINEAR
        )

        comparison = np.hstack((display, corrected))
        preview = cv2.resize(comparison, (1280, 400))

        cv2.imshow(window_name, preview)

    else:
        cv2.imshow(window_name, display)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # SELECT REGION
    # =========================
    if key == ord("r"):

        print("\nSelect the entire ChArUco board.")
        print("Press ENTER to confirm, or C to cancel.")

        selection = cv2.selectROI(
            "Select Board",
            frame,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyWindow("Select Board")

        sx, sy, sw, sh = map(int, selection)

        if sw > 0 and sh > 0:
            roi = (sx, sy, sw, sh)
            print("Board region selected:", roi)
        else:
            print("Selection cancelled.")

    # =========================
    # TOGGLE LEGACY PATTERN
    # =========================
    elif key == ord("l"):

        if len(object_points) > 0:
            print("Pattern changed: clearing old samples.")
            object_points.clear()
            image_points.clear()

        legacy_pattern = not legacy_pattern
        board, detector = create_detector()

        print("Legacy pattern:", legacy_pattern)

    # =========================
    # CAPTURE SAMPLE
    # =========================
    elif key == ord(" "):

        if corner_count >= min_corners:

            object_points.append(valid_obj.copy())
            image_points.append(valid_img.copy())

            print(
                f"Captured sample {len(object_points)} "
                f"with {corner_count} corners"
            )

        else:
            print(
                f"Not enough corners: {corner_count}/{min_corners} "
                f"(ArUco markers: {marker_count})"
            )

    # =========================
    # CALIBRATE
    # =========================
    elif key == ord("c"):

        if len(object_points) < min_samples:

            print(
                f"Need at least {min_samples} samples. "
                f"Currently have {len(object_points)}."
            )

        else:

            print("\nCalculating calibration...")

            try:
                rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
                    cv2.calibrateCamera(
                        object_points,
                        image_points,
                        image_size,
                        None,
                        None
                    )
                )

                print("\nRMS reprojection error:", rms)
                print("\nCamera matrix:\n", camera_matrix)
                print("\nDistortion coefficients:\n", dist_coeffs)

                map1, map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    dist_coeffs,
                    None,
                    camera_matrix,
                    image_size,
                    cv2.CV_16SC2
                )

                np.savez(
                    calibration_file,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    width=width,
                    height=height,
                    rms=rms
                )

                print("\nCalibration saved to:")
                print(calibration_file)
                print("Press U to toggle corrected preview.")

                show_corrected = True

            except cv2.error as error:
                print("Calibration failed:", error)

    # =========================
    # TOGGLE PREVIEW
    # =========================
    elif key == ord("u"):

        if map1 is None:
            print("Calibrate first by pressing C.")
        else:
            show_corrected = not show_corrected

    # =========================
    # QUIT
    # =========================
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
