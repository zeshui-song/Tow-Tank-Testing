from collections import deque
import os
import re
import threading
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial

# --- Configuration ---
GRBL_PORT = 'COM8'        # Grbl Mega2560
LOADCELL_PORT = 'COM9'    # Load Cell Arduino
BAUD_RATE = 115200

WORKING_DIR = r"C:\Users\zsong\Desktop\Tow Tank Testing"
GCODE_FILE_PATH = os.path.join(WORKING_DIR, "Test Gcode.txt")

# Maximum number of points shown on the live scrolling window (300 pts @ 10Hz = ~30s)
BUFFER_SIZE = 300
SAMPLING_INTERVAL = 0.1   # 100ms = ~10 Hz
TARE_SAMPLE_COUNT = 300   # 30s tare baseline (300 samples @ 10 Hz)


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


def main():
    try:
        commands = load_gcode_file(GCODE_FILE_PATH)
        print(f"Loaded {len(commands)} commands from: {GCODE_FILE_PATH}")
    except Exception as e:
        print(f"G-code File Error: {e}")
        return

    loadcell_ser = None
    grbl = None
    loadcell = None
    motion_tracker = MotionTracker(maxlen=BUFFER_SIZE)
    shared_state = {
        "global_start_time": time.time(),
        "stop_polling": False
    }

    try:
        # 1. Initialize Load Cell Hardware
        print(f"\nConnecting to Load Cell Arduino on {LOADCELL_PORT}...")
        loadcell_ser = serial.Serial(LOADCELL_PORT, BAUD_RATE, timeout=0.1)
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

        # 4. Connect Grbl Controller & Start Streaming + Polling Threads
        grbl = GrblController(GRBL_PORT, BAUD_RATE)
        grbl.connect()

        poller_thread = threading.Thread(
            target=grbl_poller,
            args=(grbl, motion_tracker, shared_state),
            daemon=True
        )
        poller_thread.start()

        gcode_thread = threading.Thread(
            target=grbl_worker,
            args=(grbl, commands),
            daemon=True
        )
        gcode_thread.start()

        # 5. Setup Matplotlib 4-Subplot Live Telemetry Window
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        fig, (ax_drag, ax_lift, ax_pos, ax_vel) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig.canvas.manager.set_window_title("Tow Tank Live Telemetry (Raw Bits, Position & Velocity)")

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

            return line_drag, line_lift, line_pos, line_vel

        ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False)

        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("\nAborting via Ctrl+C...")
    except Exception as e:
        print(f"\n[Runtime Error] {e}")
    finally:
        shared_state["stop_polling"] = True

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