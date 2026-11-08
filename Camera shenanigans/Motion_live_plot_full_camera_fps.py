from collections import deque
from datetime import datetime
import cv2
import json
import os
import re
import threading
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial

# --- Configuration ---
BAUD_RATE = 115200

WORKING_DIR = r"C:\Users\zsong\Desktop\Tow Tank Testing"
GCODE_FILE_PATH = os.path.join(WORKING_DIR, "Test Gcode.txt")
CALIBRATION_FILE_PATH = os.path.join(WORKING_DIR, "calibration.txt")

# Maximum number of points shown on the live scrolling window (300 pts @ 10Hz = ~30s)
BUFFER_SIZE = 300
SAMPLING_INTERVAL = 0.1   # 100ms = ~10 Hz
TARE_SAMPLE_COUNT = 300   # 30s tare baseline (300 samples @ 10 Hz)

CAMERA_INDEX = 1
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 800
CAMERA_FPS = 120              # Requested camera capture rate
PREVIEW_FPS = 10             # Matplotlib plot/preview refresh rate
PRE_ROLL_SECONDS = 1.0
POST_ROLL_SECONDS = 5.0
MOVE_THRESHOLD_PX = 2.5      # ID 3 displacement over ~0.3 s
VIDEO_DIR = os.path.join(WORKING_DIR, "Recordings")



def probe_serial_device(port, baudrate=BAUD_RATE):
    """Identify firmware over serial only. Return GRBL, LOADCELL, or None.

    Allow the USB serial reset/startup delay, then actively request an ID.
    Both cached verification and full discovery use the same handshake.
    """
    grbl_pattern = re.compile(
        r"<(?:Idle|Run|Hold|Jog|Alarm|Door|Check|Home|Sleep)(?:[|,>:])", re.I
    )
    with serial.Serial(port, baudrate, timeout=0.15, write_timeout=0.5) as ser:
        # Opening an Uno or Mega serial port may reset the board.
        time.sleep(2.0)
        ser.reset_input_buffer()

        # Send both read-only requests. Grbl recognizes '?' but not IDENTIFY?,
        # so identify Grbl first and do not send IDENTIFY? to it.
        for attempt in range(2):
            ser.write(b"?")
            ser.flush()
            deadline = time.monotonic() + 0.65
            while time.monotonic() < deadline:
                line = ser.readline().decode('utf-8', errors='ignore')
                if grbl_pattern.search(line):
                    return "GRBL"
                if "DEVICE:LOADCELL" in line:
                    return "LOADCELL"

            # The load-cell sketch replies to this message; Grbl doesn't.
            ser.write(b"IDENTIFY?\n")
            ser.flush()
            deadline = time.monotonic() + 0.9
            while time.monotonic() < deadline:
                line = ser.readline().decode('utf-8', errors='ignore')
                if "DEVICE:LOADCELL" in line:
                    return "LOADCELL"
                if grbl_pattern.search(line):
                    return "GRBL"
    return None


def identify_saved_port(port, expected, baudrate=BAUD_RATE):
    """Check a saved port using the SAME serial handshake as discovery."""
    try:
        actual = probe_serial_device(port, baudrate)
    except (serial.SerialException, OSError) as exc:
        print(f"  Saved {expected} port {port} unavailable: {exc}")
        return False
    if actual != expected:
        print(f"  Saved {expected} port {port} responded as {actual or 'unknown'}.")
        return False
    print(f"  Confirmed {expected} on {port}")
    return True


def get_arduino_ports(calibration_path=CALIBRATION_FILE_PATH):
    """Reuse verified cached COM assignments; rediscover and save only on a cache miss."""
    calibration = {}
    if os.path.isfile(calibration_path):
        with open(calibration_path, 'r', encoding='utf-8') as f:
            calibration = json.load(f)
        if not isinstance(calibration, dict):
            raise ValueError("Calibration file must contain a JSON object.")

    grbl_port = calibration.get("grbl_port")
    loadcell_port = calibration.get("loadcell_port")
    if (isinstance(grbl_port, str) and isinstance(loadcell_port, str)
            and grbl_port and loadcell_port and grbl_port != loadcell_port):
        print(f"\nChecking saved ports: Grbl = {grbl_port}, Load cell = {loadcell_port}")
        if (identify_saved_port(grbl_port, "GRBL")
                and identify_saved_port(loadcell_port, "LOADCELL")):
            print("Saved COM assignments confirmed. Skipping full port scan.\n")
            return grbl_port, loadcell_port
        print("Saved connection not detected; rescanning connected serial ports...")
    else:
        print("\nNo valid saved COM assignments; discovering Arduinos...")

    grbl_port, loadcell_port = discover_arduino_ports()
    calibration["grbl_port"] = grbl_port
    calibration["loadcell_port"] = loadcell_port

    # Replace the JSON file as a whole: plain text appended after JSON would
    # break the existing json.load() calibration reader.
    tmp_path = calibration_path + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(calibration, f, indent=4)
            f.write("\n")
        os.replace(tmp_path, calibration_path)
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    print(f"Saved updated COM assignments to {calibration_path}")
    return grbl_port, loadcell_port


def discover_arduino_ports(baudrate=BAUD_RATE):
    """Scan serial ports and identify firmware using the same handshake."""
    from serial.tools import list_ports

    devices = {}
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports were found. Check the USB dock and cables.")

    print("\nSearching serial ports for load-cell and Grbl Arduinos...")
    for info in ports:
        port = info.device
        print(f"  Checking {port}...")
        try:
            kind = probe_serial_device(port, baudrate)
        except (serial.SerialException, OSError) as exc:
            print(f"    Skipping {port}: {exc}")
            continue
        if kind is None:
            print("    No recognized Arduino response")
            continue
        if kind in devices:
            raise RuntimeError(
                f"Multiple {kind} devices found ({devices[kind]} and {port}). "
                "Disconnect duplicates before running."
            )
        devices[kind] = port
        print(f"    Identified {kind} on {port}")
        if len(devices) == 2:
            break

    missing = {"LOADCELL", "GRBL"} - devices.keys()
    if missing:
        raise RuntimeError(
            f"Cannot identify {', '.join(sorted(missing))}. "
            "Verify both boards are connected, upload the updated load-cell sketch, "
            "and close Arduino Serial Monitor or other programs using the ports."
        )
    print(f"Detected connections: Load cell = {devices['LOADCELL']}, Grbl = {devices['GRBL']}\n")
    return devices["GRBL"], devices["LOADCELL"]




def measure_live_tare(ser, sample_count=TARE_SAMPLE_COUNT):
    """Measures a 30s baseline to calculate zero-offset in raw bits."""
    print("\n" + "=" * 75)
    print(f"TARE MEASUREMENT ({sample_count * SAMPLING_INTERVAL:.0f}s)")
    print("=" * 75)
    print("Ensure system is unloaded and stationary...")

    ser.reset_input_buffer()
    drag_samples = []
    lift_samples = []

    while len(drag_samples) < sample_count:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line and ',' in raw_line:
                try:
                    parts = raw_line.split(',')
                    drag_samples.append(float(parts[0]))
                    lift_samples.append(float(parts[1]))
                    print(f" Sampling Tare: [{len(drag_samples):03d}/{sample_count}] | Drag: {drag_samples[-1]:9.1f} | Lift: {lift_samples[-1]:9.1f}   ", end='\r')
                except ValueError:
                    pass
        time.sleep(SAMPLING_INTERVAL)

    drag_tare = float(np.mean(drag_samples))
    lift_tare = float(np.mean(lift_samples))
    print(f"\n\n[TARE COMPLETE] Offset -> Drag: {drag_tare:.2f} bits | Lift: {lift_tare:.2f} bits\n")

    return drag_tare, lift_tare


class LoadCellReader(threading.Thread):
    def __init__(self, ser, drag_tare=0.0, lift_tare=0.0, maxlen=300):
        super().__init__(daemon=True)
        self.ser = ser
        self.drag_tare = drag_tare
        self.lift_tare = lift_tare
        self.running = False
        self.start_time = None

        self.times = deque(maxlen=maxlen)
        self.drag_vals = deque(maxlen=maxlen)
        self.lift_vals = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def run(self):
        try:
            self.running = True
            self.ser.reset_input_buffer()
            self.start_time = time.time()

            buffer = ""
            while self.running:
                if self.ser.in_waiting:
                    chunk = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
                    if '\n' in buffer:
                        lines = buffer.split('\n')
                        buffer = lines[-1]
                        for line in lines[:-1]:
                            raw_line = line.strip()
                            if raw_line and ',' in raw_line:
                                try:
                                    drag_str, lift_str = raw_line.split(',')
                                    raw_drag = float(drag_str)
                                    raw_lift = float(lift_str)

                                    # Tare subtraction only (raw bits)
                                    tared_drag = raw_drag - self.drag_tare
                                    tared_lift = raw_lift - self.lift_tare

                                    elapsed = time.time() - self.start_time

                                    with self.lock:
                                        self.times.append(elapsed)
                                        self.drag_vals.append(tared_drag)
                                        self.lift_vals.append(tared_lift)
                                except ValueError:
                                    continue
                time.sleep(0.01)
        except serial.SerialException as e:
            print(f"[LoadCell Error] {e}")

    def get_plot_data(self):
        with self.lock:
            return list(self.times), list(self.drag_vals), list(self.lift_vals)

    def stop(self):
        self.running = False


class MotionTracker:
    """Tracks position and signed directional velocity (+ forward, - backward) reported by Grbl."""
    def __init__(self, maxlen=300):
        self.times = deque(maxlen=maxlen)
        self.pos_vals = deque(maxlen=maxlen)
        self.vel_vals = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def record_motion(self, elapsed, pos_mm, signed_feed_mm_s):
        with self.lock:
            self.times.append(elapsed)
            self.pos_vals.append(pos_mm)
            self.vel_vals.append(signed_feed_mm_s)

    def get_plot_data(self):
        with self.lock:
            return list(self.times), list(self.pos_vals), list(self.vel_vals)


class GrblController:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.io_lock = threading.Lock()

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        time.sleep(2)
        with self.io_lock:
            self.ser.reset_input_buffer()
            self.ser.write(b"\r\n\r\n")
            time.sleep(1)
            self.ser.flushInput()

    def send_command(self, cmd):
        if not self.ser or not self.ser.is_open:
            return ""

        with self.io_lock:
            formatted_cmd = (cmd.strip() + '\n').encode('utf-8')
            self.ser.write(formatted_cmd)

            responses = []
            current_line = ""
            while True:
                if self.ser.in_waiting > 0:
                    char = self.ser.read(1).decode('utf-8', errors='ignore')
                    if char in ('\n', '\r'):
                        clean = current_line.strip()
                        if clean:
                            responses.append(clean)
                            if clean.startswith('ok') or clean.startswith('error'):
                                break
                        current_line = ""
                    else:
                        current_line += char
                else:
                    time.sleep(0.01)

            return " | ".join(responses)

    def query_status(self):
        """Polls Grbl with '?' and parses coordinates and programmed/real-time feed rate."""
        if not self.ser or not self.ser.is_open:
            return None, None, None

        with self.io_lock:
            try:
                self.ser.write(b'?')
                time.sleep(0.02)
                resp = ""
                while self.ser.in_waiting > 0:
                    resp += self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
            except Exception:
                return None, None, None

        state = None
        pos_x = None
        feed_rate_mm_s = None

        if '<' in resp and '>' in resp:
            status_match = re.search(r'<([^,>]+)', resp)
            if status_match:
                state = status_match.group(1)

            pos_match = re.search(r'(?:MPos|WPos):([-\d.]+),([-\d.]+),([-\d.]+)', resp)
            if pos_match:
                try:
                    pos_x = float(pos_match.group(1))
                except ValueError:
                    pass

            # Grbl v1.1 reports feed/spindle as |FS:feed,spindle| (feed in mm/min)
            fs_match = re.search(r'\|FS:([-\d.]+),', resp)
            if fs_match:
                try:
                    feed_mm_min = float(fs_match.group(1))
                    feed_rate_mm_s = feed_mm_min / 60.0
                except ValueError:
                    pass

            # Grbl v0.9 reports |F:feed|
            if feed_rate_mm_s is None:
                f_match = re.search(r'\|F:([-\d.]+)', resp)
                if f_match:
                    try:
                        feed_mm_min = float(f_match.group(1))
                        feed_rate_mm_s = feed_mm_min / 60.0
                    except ValueError:
                        pass

        return state, pos_x, feed_rate_mm_s

    def wait_until_idle(self):
        if not self.ser or not self.ser.is_open:
            return

        time.sleep(0.2)
        print("\n>>> All G-code commands buffered. Running to completion... <<<\n")

        while True:
            state, _, _ = self.query_status()
            if state == 'Idle':
                print("\n>>> Motion complete. Grbl returned to Idle. <<<")
                break
            elif state == 'Alarm':
                print("\n[Grbl Warning] Machine entered Alarm.")
                break
            time.sleep(0.05)

    def emergency_stop(self):
        if self.ser and self.ser.is_open:
            try:
                with self.io_lock:
                    self.ser.write(b'!')
                    time.sleep(0.02)
                    self.ser.write(b'\x18')
                    time.sleep(0.05)
                    self.ser.reset_input_buffer()
                print("[Grbl] E-Stop sent - motors halted.")
            except Exception as e:
                print(f"[Grbl E-Stop Error] {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def load_gcode_file(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    commands = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line.startswith('>>>'):
                clean_line = clean_line.replace('>>>', '').strip()
            clean_line = re.sub(r'\(.*?\)', '', clean_line).strip()
            if ';' in clean_line:
                clean_line = clean_line.split(';')[0].strip()
            if clean_line.startswith('$') and '=' in clean_line:
                key, val = clean_line.split('=', 1)
                clean_line = f"{key.strip()}={val.strip()}"
            if clean_line:
                commands.append(clean_line)
    return commands


def grbl_poller(grbl, motion_tracker, shared_state):
    """Continuously queries Grbl status at ~20Hz to sample position and signed directional velocity."""
    last_x = 0.0
    last_time = time.time()
    last_sign = 1.0

    while not shared_state["stop_polling"]:
        current_wall_time = time.time()
        elapsed = current_wall_time - shared_state["global_start_time"]
        _, pos_x, feed_mm_s = grbl.query_status()

        if pos_x is not None:
            dt = current_wall_time - last_time
            dx = pos_x - last_x

            # Determine sign direction (+ forward, - backward)
            if abs(dx) > 0.005:
                direction = 1.0 if dx > 0 else -1.0
                last_sign = direction
            else:
                direction = last_sign

            # Use hardware reported feed rate with sign, fallback to numerical diff dx/dt
            if feed_mm_s is not None:
                signed_velocity = direction * feed_mm_s if abs(dx) > 0.001 else 0.0
            else:
                signed_velocity = (dx / dt) if dt > 0 else 0.0

            last_x = pos_x
            last_time = current_wall_time
        else:
            signed_velocity = 0.0

        motion_tracker.record_motion(elapsed, last_x, signed_velocity)
        time.sleep(0.05)


def grbl_worker(grbl, commands):
    """Streams G-code sequence to Grbl."""
    try:
        for idx, cmd in enumerate(commands, start=1):
            print(f"[{idx}/{len(commands)}] >>> G-CODE: {cmd}")
            ack = grbl.send_command(cmd)
            print(f"[{idx}/{len(commands)}] <<< ACK:    {ack}")

        grbl.wait_until_idle()
    except Exception as e:
        print(f"[Grbl Worker Error] {e}")



class CameraTracker(threading.Thread):
    """Detect markers at camera speed, independently of the 10 FPS plot preview.

    The recording contains every frame Python successfully acquires. The graph
    portion reuses its most recent 10 FPS rendering.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_position = None
        self.latest_time = None
        self.latest_plot = np.full((800, 600, 3), 255, dtype=np.uint8)
        self.samples = {i: [] for i in (0, 1, 2)}
        self.fixed = {}
        self.prebuffer = deque()
        self.motion_history = deque()
        self.writer = None
        self.video_path = None
        self.last_motion = None
        self.video_start = None
        self.written_frames = 0
        self.capture_times = deque(maxlen=100)
        self.output_fps = None
        self.was_moving = False
        self.trigger_armed = True

    def set_plot(self, rgb_plot):
        # Called from the Matplotlib thread, approximately 10 times per second.
        with self.lock:
            self.latest_plot = cv2.cvtColor(
                cv2.resize(rgb_plot, (600, 800)), cv2.COLOR_RGB2BGR
            )

    def compose(self, camera_bgr, plot_bgr):
        # Output 1600x800. Keep the camera aspect ratio; do not stretch it.
        h, w = camera_bgr.shape[:2]
        resized_h = round(1000 * h / w)
        resized = cv2.resize(camera_bgr, (1000, resized_h))
        canvas = np.zeros((800, 1600, 3), dtype=np.uint8)
        top = (800 - resized_h) // 2
        canvas[top:top + resized_h, :1000] = resized
        canvas[:, 1000:] = plot_bgr
        return canvas

    def motion_detected(self, pos, now):
        # Detect motion directly from captured camera frames, not plot refreshes.
        if pos is None:
            return False
        self.motion_history.append((now, np.asarray(pos).copy()))
        while self.motion_history and now - self.motion_history[0][0] > 0.30:
            self.motion_history.popleft()
        if len(self.motion_history) < 2:
            return False
        return (np.linalg.norm(self.motion_history[-1][1] -
                               self.motion_history[0][1]) >= MOVE_THRESHOLD_PX)

    def write_video_frame(self, image, timestamp):
        if self.writer is None:
            return
        # Constant-FPS MP4: write each acquired frame once. If camera capture
        # varies significantly, playback timing is only approximate.
        self.writer.write(image)
        self.written_frames += 1

    def start_recording(self, now):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self.video_path = os.path.join(
            VIDEO_DIR, datetime.now().strftime('tow_tank_%Y%m%d_%H%M%S.mp4')
        )
        # Use observed camera capture rate, never the preview frame rate.
        if len(self.capture_times) > 10:
            duration = self.capture_times[-1] - self.capture_times[0]
            measured = (len(self.capture_times) - 1) / duration if duration > 0 else 30
        else:
            measured = 30
        self.output_fps = max(1, min(120, round(measured)))
        self.writer = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*'mp4v'),
            self.output_fps, (1600, 800)
        )
        if not self.writer.isOpened():
            self.writer = None
            print('[Video] Cannot open MP4 writer')
            return
        print(f'[Video] Recording at {self.output_fps} FPS (measured camera capture rate)')
        print(f'[Video] Started: {self.video_path}')
        for ts, jpg, plot in self.prebuffer:
            previous_camera = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if previous_camera is not None:
                self.write_video_frame(self.compose(previous_camera, plot), ts)
        self.prebuffer.clear()
        self.last_motion = now
        self.trigger_armed = False

    def finish_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f'[Video] Saved: {self.video_path} ({self.written_frames} frames)')
            self.written_frames = 0
            self.motion_history.clear()
            self.was_moving = False

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print('[Camera] Cannot open camera 1.')
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        print('[Camera] Reported mode:',
              cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'x', cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
              'at', cap.get(cv2.CAP_PROP_FPS), 'FPS (requested', CAMERA_FPS, ')')

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 4
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.maxErroneousBitsInBorderRate = 0.5
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    print('[Camera] Frame read failed')
                    break
                now = time.monotonic()
                self.capture_times.append(now)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
                corners, ids, _ = detector.detectMarkers(gray)
                moving_position = None
                if ids is not None:
                    for pts, mid in zip(corners, ids.flatten()):
                        mid = int(mid)
                        center = pts[0].mean(axis=0)
                        if mid in self.samples and mid not in self.fixed:
                            self.samples[mid].append(center.copy())
                            if len(self.samples[mid]) == 30:
                                self.fixed[mid] = np.mean(self.samples[mid], axis=0)
                                print(f'[Camera] Fixed marker {mid} locked')
                        elif mid == 3:
                            moving_position = center.copy()
                for mid, pos in self.fixed.items():
                    x, y = np.rint(pos).astype(int)
                    cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
                    cv2.putText(display, f'ID {mid} FIXED', (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                if moving_position is not None:
                    x, y = np.rint(moving_position).astype(int)
                    cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                    cv2.putText(display, f'ID 3 ({x}, {y})', (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.putText(display, f'Fixed: {len(self.fixed)}/3', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                with self.lock:
                    self.latest_frame = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    self.latest_position = moving_position
                    self.latest_time = now
                    plot = self.latest_plot.copy()

                moving = self.motion_detected(moving_position, now)
                if moving:
                    self.last_motion = now

                if self.writer is None:
                    # JPEG prebuffer avoids storing hundreds of full raw frames.
                    success, encoded = cv2.imencode('.jpg', display,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        self.prebuffer.append((now, encoded.tobytes(), plot))
                    while self.prebuffer and now - self.prebuffer[0][0] > PRE_ROLL_SECONDS:
                        self.prebuffer.popleft()
                    # Re-arm only after the marker has been stationary for 5 s.
                    if not moving and self.last_motion is not None and now - self.last_motion > POST_ROLL_SECONDS:
                        self.trigger_armed = True
                    if moving and self.trigger_armed:
                        self.start_recording(now)
                else:
                    self.write_video_frame(self.compose(display, plot), now)
                    if self.last_motion is not None and now - self.last_motion >= POST_ROLL_SECONDS:
                        self.finish_recording()

        finally:
            self.finish_recording()
            cap.release()

    def snapshot(self):
        with self.lock:
            return self.latest_frame, (None if self.latest_position is None
                                       else self.latest_position.copy()), self.latest_time

    def stop(self):
        self.running = False


def main():
    try:
        commands = load_gcode_file(GCODE_FILE_PATH)
        print(f"Loaded {len(commands)} commands from: {GCODE_FILE_PATH}")
    except Exception as e:
        print(f"G-code File Error: {e}")
        return

    try:
        grbl_port, loadcell_port = get_arduino_ports()
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as e:
        print(f"[Port Detection Error] {e}")
        return

    loadcell_ser = None
    grbl = None
    loadcell = None
    camera_tracker = None
    motion_tracker = MotionTracker(maxlen=BUFFER_SIZE)
    shared_state = {
        "global_start_time": time.time(),
        "stop_polling": False
    }

    try:
        # 1. Initialize Load Cell Hardware
        print(f"\nConnecting to Load Cell Arduino on {loadcell_port}...")
        loadcell_ser = serial.Serial(loadcell_port, BAUD_RATE, timeout=0.1)
        time.sleep(2)

        # 2. Perform 30-Second Baseline Zero Tare
        live_drag_tare, live_lift_tare = measure_live_tare(loadcell_ser, sample_count=TARE_SAMPLE_COUNT)

        # Align global time baseline
        shared_state["global_start_time"] = time.time()

        # 3. Start Live Load Cell Reader Thread (Streaming Tared ADC Bits)
        loadcell = LoadCellReader(
            loadcell_ser,
            drag_tare=live_drag_tare,
            lift_tare=live_lift_tare,
            maxlen=BUFFER_SIZE
        )
        loadcell.start()

        # 4. Camera tracking runs independently of Matplotlib.
        camera_tracker = CameraTracker()
        camera_tracker.start()

        # 5. Connect Grbl Controller & Start Streaming + Polling Threads
        grbl = GrblController(grbl_port, BAUD_RATE)
        grbl.connect()

        poller_thread = threading.Thread(
            target=grbl_poller,
            args=(grbl, motion_tracker, shared_state),
            daemon=True
        )
        poller_thread.start()

        # 6. Show camera next to the original four telemetry plots.
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        fig = plt.figure(figsize=(16, 9), dpi=100)
        gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.38, wspace=0.23)
        ax_cam = fig.add_subplot(gs[:, 0])
        ax_drag = fig.add_subplot(gs[0, 1])
        ax_lift = fig.add_subplot(gs[1, 1], sharex=ax_drag)
        ax_pos = fig.add_subplot(gs[2, 1], sharex=ax_drag)
        ax_vel = fig.add_subplot(gs[3, 1], sharex=ax_drag)
        fig.canvas.manager.set_window_title("Tow Tank Live Telemetry + Camera")
        blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        camera_image = ax_cam.imshow(blank)
        ax_cam.set_title("Camera + ArUco Tracking")
        ax_cam.axis("off")


        line_drag, = ax_drag.plot([], [], color='#d62728', lw=1.8, label="Drag (Raw Bits Δ)")
        line_lift, = ax_lift.plot([], [], color='#1f77b4', lw=1.8, label="Lift (Raw Bits Δ)")
        line_pos, = ax_pos.plot([], [], color='#2ca02c', lw=1.8, label="Gantry Position (mm)")
        line_vel, = ax_vel.plot([], [], color='#9467bd', lw=1.8, label="Signed Velocity (+Fwd / -Bwd)")

        ax_drag.set_ylabel("Drag (Bits Δ)")
        ax_drag.legend(loc="upper right")
        ax_drag.grid(True)

        ax_lift.set_ylabel("Lift (Bits Δ)")
        ax_lift.legend(loc="upper right")
        ax_lift.grid(True)

        ax_pos.set_ylabel("Position (mm)")
        ax_pos.legend(loc="upper right")
        ax_pos.grid(True)

        ax_vel.set_xlabel("Time (seconds)")
        ax_vel.set_ylabel("Velocity (mm/s)")
        ax_vel.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_vel.legend(loc="upper right")
        ax_vel.grid(True)

        def update_plot(frame):
            cam_frame, pos, cam_time = camera_tracker.snapshot()
            if cam_frame is not None:
                camera_image.set_data(cam_frame)
            t, d, l = loadcell.get_plot_data()
            tm, p, v = motion_tracker.get_plot_data()

            if len(t) > 1:
                line_drag.set_data(t, d)
                line_lift.set_data(t, l)

                if tm and len(tm) > 1:
                    interp_p = np.interp(t, tm, p)
                    interp_v = np.interp(t, tm, v)
                    line_pos.set_data(t, interp_p)
                    line_vel.set_data(t, interp_v)
                elif p:
                    line_pos.set_data(t, [p[-1]] * len(t))
                    line_vel.set_data(t, [v[-1] if v else 0.0] * len(t))

                # Auto-scroll X-axis window
                ax_drag.set_xlim(t[0], max(t[-1], t[0] + 5))

                # Auto-scale Y-axes dynamically with 5% margin
                min_d, max_d = min(d), max(d)
                margin_d = (max_d - min_d) * 0.05 or 10.0
                ax_drag.set_ylim(min_d - margin_d, max_d + margin_d)

                min_l, max_l = min(l), max(l)
                margin_l = (max_l - min_l) * 0.05 or 10.0
                ax_lift.set_ylim(min_l - margin_l, max_l + margin_l)

                pos_data = line_pos.get_ydata()
                if len(pos_data) > 0:
                    min_p, max_p = min(pos_data), max(pos_data)
                    margin_p = (max_p - min_p) * 0.05 or 1.0
                    ax_pos.set_ylim(min_p - margin_p, max_p + margin_p)

                vel_data = line_vel.get_ydata()
                if len(vel_data) > 0:
                    min_v, max_v = min(vel_data), max(vel_data)
                    margin_v = (max_v - min_v) * 0.10 or 5.0
                    ax_vel.set_ylim(min_v - margin_v, max_v + margin_v)

            return line_drag, line_lift, line_pos, line_vel, camera_image

        def update_plot_snapshot(event):
            if event.canvas is not fig.canvas:
                return
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            # Cache the plot column only. Camera thread records at capture speed.
            camera_tracker.set_plot(rgb[:, int(rgb.shape[1] * 0.53):].copy())

        fig.canvas.mpl_connect("draw_event", update_plot_snapshot)
        ani = animation.FuncAnimation(fig, update_plot, interval=1000 // PREVIEW_FPS, blit=False,
                                      cache_frame_data=False)

        # Wait for the camera to warm up while Grbl connects, then launch motion.
        gcode_thread = threading.Thread(target=grbl_worker, args=(grbl, commands), daemon=True)
        gcode_thread.start()
        plt.show()

    except KeyboardInterrupt:
        print("\nAborting via Ctrl+C...")
    except Exception as e:
        print(f"\n[Runtime Error] {e}")
    finally:
        shared_state["stop_polling"] = True
        if camera_tracker:
            camera_tracker.stop()
            camera_tracker.join(timeout=2)

        if grbl:
            grbl.emergency_stop()
            grbl.close()
        if loadcell:
            loadcell.stop()
        if loadcell_ser and loadcell_ser.is_open:
            loadcell_ser.close()

        print("Test ended. All serial connections closed.")


if __name__ == '__main__':
    main()