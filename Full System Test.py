from collections import deque
from datetime import datetime
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
GRBL_PORT = 'COM8'         # Grbl Mega2560
LOADCELL_PORT = 'COM9'     # Load Cell Arduino
BAUD_RATE = 115200

WORKING_DIR = r"C:\Users\zsong\Desktop\Tow Tank Testing"
GCODE_FILE_PATH = os.path.join(WORKING_DIR, "Test Gcode.txt")
CALIBRATION_FILE_PATH = os.path.join(WORKING_DIR, "calibration.txt")

# Maximum number of points shown on the live scrolling window (e.g. 300 pts @ 10Hz = ~30s)
BUFFER_SIZE = 300

# Warm-up, Settling & Tare Baseline Settings
WARMUP_MIN_DURATION_SEC = 60
WARMUP_EXTEND_SEC = 30
SETTLING_DURATION_SEC = 60     # Post-polarity hydro/mechanical settling time
MAX_STABLE_SLOPE_PCT = 0.10   # Maximum allowable drift slope (% full-scale/min)
MAX_HX711_BITS = 8388607.0     # 24-bit ADC positive full scale for percentage drift
SAMPLING_INTERVAL = 0.1        # 100ms = ~10 Hz
TARE_SAMPLE_COUNT = 300        # 30s tare baseline (300 samples @ 10 Hz)
DIRECTION_SAMPLE_COUNT = 30    # 3s sampling for live load direction test


def load_scale_factors(filepath):
    """Loads conversion scale factors from the JSON calibration file (magnitude preserved)."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        cal_data = json.load(f)

    return {
        "drag_scale": abs(float(cal_data.get("drag_scale_factor_N", 1.0))),
        "lift_scale": abs(float(cal_data.get("lift_scale_factor_N", 1.0))),
    }


def compute_drift_metrics(samples, time_step=SAMPLING_INTERVAL):
    """Calculates linear drift rate in % FS/min over current sample buffer."""
    if len(samples) < 30:
        return 0.0
    y = np.array(samples)
    x = np.arange(len(y)) * time_step
    slope_bits_per_sec, _ = np.polyfit(x, y, 1)
    slope_pct_per_min = (slope_bits_per_sec * 60.0 / MAX_HX711_BITS) * 100.0
    return slope_pct_per_min


def wait_for_user_choice(prompt_msg):
    """Halts execution until user inputs 'Y' to proceed or 'Q' to quit."""
    while True:
        user_input = input(f"\n{prompt_msg} [Type 'Y' to proceed, 'Q' to quit]: ").strip().upper()
        if user_input == 'Y':
            return 'PROCEED'
        elif user_input == 'Q':
            print("\n[ABORT] User initiated program exit.")
            raise KeyboardInterrupt("User aborted session.")
        print("Invalid choice. Please type 'Y' to proceed or 'Q' to quit.")


def run_loadcell_warmup(ser):
    """Executes a 60s minimum thermal warm-up and drift evaluation sequence."""
    print("\n" + "=" * 75)
    print("PHASE 1: LOAD CELL SELF-HEATING & THERMAL STABILIZATION (60s)")
    print("=" * 75)
    print(f"Running minimum warmup of {WARMUP_MIN_DURATION_SEC}s. Slope stability limit: ±{MAX_STABLE_SLOPE_PCT:.3f}% FS/min.")

    warmup_target_duration = WARMUP_MIN_DURATION_SEC
    drag_window = []
    lift_window = []
    start_warmup = time.time()
    ser.reset_input_buffer()

    while True:
        elapsed = time.time() - start_warmup
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line and ',' in raw_line:
                try:
                    parts = raw_line.split(',')
                    drag_raw = float(parts[0])
                    lift_raw = float(parts[1])

                    drag_window.append(drag_raw)
                    lift_window.append(lift_raw)

                    # Rolling 30s window (300 samples)
                    if len(drag_window) > 300:
                        drag_window.pop(0)
                        lift_window.pop(0)

                    drag_slope_pct = compute_drift_metrics(drag_window)
                    lift_slope_pct = compute_drift_metrics(lift_window)

                    remaining = max(0, int(warmup_target_duration - elapsed))
                    print(f" Timer: {remaining:02d}s | Drag: {drag_raw:9.1f} bits (Slope: {drag_slope_pct:+.3f}%/m) | Lift: {lift_raw:9.1f} bits (Slope: {lift_slope_pct:+.3f}%/m)   ", end='\r')
                except ValueError:
                    pass

        if elapsed >= warmup_target_duration:
            drag_slope_pct = compute_drift_metrics(drag_window)
            lift_slope_pct = compute_drift_metrics(lift_window)

            if abs(drag_slope_pct) <= MAX_STABLE_SLOPE_PCT and abs(lift_slope_pct) <= MAX_STABLE_SLOPE_PCT:
                print(f"\n\n[STABILITY ACHIEVED] Both channels settled under ±{MAX_STABLE_SLOPE_PCT:.3f}%/min threshold.")
                break
            else:
                print(f"\n\n[DRIFT DETECTED] Drag: {drag_slope_pct:+.3f}%/m | Lift: {lift_slope_pct:+.3f}%/m (Limit: ±{MAX_STABLE_SLOPE_PCT:.3f}%/m).")
                choice = wait_for_user_choice(f"Extend warmup by +{WARMUP_EXTEND_SEC}s?")
                if choice == 'PROCEED':
                    warmup_target_duration += WARMUP_EXTEND_SEC

        time.sleep(SAMPLING_INTERVAL)


def determine_channel_polarity(ser, channel_name, base_scale_magnitude, sample_count=DIRECTION_SAMPLE_COUNT):
    """Prompts user to load a channel in the POSITIVE direction and resolves scale factor sign."""
    print("\n" + "=" * 75)
    print(f"PHASE 2: POLARITY CHECK ({channel_name.upper()} CHANNEL)")
    print("=" * 75)

    ser.reset_input_buffer()
    baseline_samples = []
    ch_idx = 0 if channel_name.lower() == "drag" else 1

    print(f"Measuring neutral rest baseline for {channel_name} (3s)...")
    while len(baseline_samples) < sample_count:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line and ',' in raw_line:
                try:
                    parts = raw_line.split(',')
                    baseline_samples.append(float(parts[ch_idx]))
                except ValueError:
                    pass
        time.sleep(SAMPLING_INTERVAL)

    pre_load_baseline = float(np.mean(baseline_samples))

    print(f">> Apply a steady non-zero load in the POSITIVE {channel_name} direction.")
    wait_for_user_choice(f"Press 'Y' once you are applying steady +{channel_name} load")

    ser.reset_input_buffer()
    loaded_samples = []

    print(f"Sampling +{channel_name} load response...")
    while len(loaded_samples) < sample_count:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line and ',' in raw_line:
                try:
                    parts = raw_line.split(',')
                    loaded_samples.append(float(parts[ch_idx]))
                    print(f" Reading: [{len(loaded_samples):02d}/{sample_count}] | Raw: {loaded_samples[-1]:9.1f} bits   ", end='\r')
                except ValueError:
                    pass
        time.sleep(SAMPLING_INTERVAL)

    mean_loaded_raw = float(np.mean(loaded_samples))
    delta_bits = mean_loaded_raw - pre_load_baseline

    if abs(delta_bits) < 50.0:
        print(f"\n\n[WARNING] Measured delta is very small ({delta_bits:+.1f} bits). Polarity check may be ambiguous.")

    sign = 1.0 if delta_bits >= 0 else -1.0
    final_scale = sign * abs(base_scale_magnitude)

    print(f"\n\n[POLARITY RESOLVED] {channel_name.upper()}: Δ = {delta_bits:+.1f} bits | Sign = {sign:+.0f} | Final Scale = {final_scale:.6e} N/bit\n")

    print(f">> Release the {channel_name} load completely.")
    wait_for_user_choice("Press 'Y' once the load is fully released")

    return final_scale


def wait_for_settling(ser, duration_sec=SETTLING_DURATION_SEC):
    """Waits for mechanical ringing, fluid sloshing, and thermal disturbance to settle completely."""
    print("\n" + "=" * 75)
    print(f"PHASE 3: POST-LOAD SETTLING & RINGING DECAY ({duration_sec}s)")
    print("=" * 75)
    print(f"Waiting {duration_sec}s for water sloshing, structural vibrations, and elastic hysteresis to settle...")

    start_settle = time.time()
    ser.reset_input_buffer()

    while True:
        elapsed = time.time() - start_settle
        if elapsed >= duration_sec:
            break

        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line and ',' in raw_line:
                try:
                    parts = raw_line.split(',')
                    d_raw = float(parts[0])
                    l_raw = float(parts[1])
                    remaining = max(0, int(duration_sec - elapsed))
                    print(f" Settling Timer: {remaining:02d}s | Drag: {d_raw:9.1f} bits | Lift: {l_raw:9.1f} bits   ", end='\r')
                except ValueError:
                    pass
        time.sleep(SAMPLING_INTERVAL)

    print("\n\n[SETTLED] Hydrodynamic and structural ringing decayed to quiescent state.\n")


def measure_live_tare(ser, sample_count=TARE_SAMPLE_COUNT):
    """Measures a 30s unloaded baseline to establish live zero-offsets."""
    print("\n" + "=" * 75)
    print(f"PHASE 4: LIVE ZERO TARE CALCULATION ({sample_count * SAMPLING_INTERVAL:.0f}s)")
    print("=" * 75)
    print("Ensure test section/model is fully unloaded and stationary in still water.")
    
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
    print(f"\n\n[TARE COMPUTED] Active Zero Baseline -> Drag: {drag_tare:.2f} bits | Lift: {lift_tare:.2f} bits\n")

    return drag_tare, lift_tare


class LoadCellReader(threading.Thread):
    def __init__(self, ser, cal_params, maxlen=300):
        super().__init__(daemon=True)
        self.ser = ser
        self.cal_params = cal_params
        self.running = False
        self.start_time = None
        
        self.times = deque(maxlen=maxlen)
        self.drag_vals = deque(maxlen=maxlen)
        self.lift_vals = deque(maxlen=maxlen)
        
        self.all_times = []
        self.all_drag = []
        self.all_lift = []
        
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
                                    
                                    drag_n = (raw_drag - self.cal_params["drag_tare"]) * self.cal_params["drag_scale"]
                                    lift_n = (raw_lift - self.cal_params["lift_tare"]) * self.cal_params["lift_scale"]
                                    
                                    elapsed = time.time() - self.start_time
                                    
                                    with self.lock:
                                        self.times.append(elapsed)
                                        self.drag_vals.append(drag_n)
                                        self.lift_vals.append(lift_n)
                                        
                                        self.all_times.append(elapsed)
                                        self.all_drag.append(drag_n)
                                        self.all_lift.append(lift_n)
                                except ValueError:
                                    continue
                time.sleep(0.01)
        except serial.SerialException as e:
            print(f"[LoadCell Error] {e}")

    def get_plot_data(self):
        with self.lock:
            return list(self.times), list(self.drag_vals), list(self.lift_vals)

    def get_full_history(self):
        with self.lock:
            return list(self.all_times), list(self.all_drag), list(self.all_lift)

    def stop(self):
        self.running = False


class MotionTracker:
    """Tracks position and signed directional velocity (+ forward, - backward) reported by Grbl."""
    def __init__(self, maxlen=300):
        self.times = deque(maxlen=maxlen)
        self.pos_vals = deque(maxlen=maxlen)
        self.vel_vals = deque(maxlen=maxlen)
        
        self.all_times = []
        self.all_pos = []
        self.all_vel = []
        self.lock = threading.Lock()

    def record_motion(self, elapsed, pos_mm, signed_feed_mm_s):
        with self.lock:
            self.times.append(elapsed)
            self.pos_vals.append(pos_mm)
            self.vel_vals.append(signed_feed_mm_s)
            
            self.all_times.append(elapsed)
            self.all_pos.append(pos_mm)
            self.all_vel.append(signed_feed_mm_s)

    def get_plot_data(self):
        with self.lock:
            return list(self.times), list(self.pos_vals), list(self.vel_vals)

    def get_full_history(self):
        with self.lock:
            return list(self.all_times), list(self.all_pos), list(self.all_vel)


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


def grbl_worker(grbl, commands, shared_state):
    """Streams G-code sequence and marks precise motion transition timestamps."""
    try:
        shared_state["motion_start_time"] = time.time() - shared_state["global_start_time"]

        for idx, cmd in enumerate(commands, start=1):
            print(f"[{idx}/{len(commands)}] >>> G-CODE: {cmd}")
            ack = grbl.send_command(cmd)
            print(f"[{idx}/{len(commands)}] <<< ACK:    {ack}")
            
        grbl.wait_until_idle()
        shared_state["motion_end_time"] = time.time() - shared_state["global_start_time"]
    except Exception as e:
        print(f"[Grbl Worker Error] {e}")


def save_trimmed_summary_plot(loadcell, motion_tracker, motion_start, motion_end, output_dir=WORKING_DIR):
    """Prompts the user for a plot name or defaults to motion_YYYYMMDD_HHMMSS, then saves."""
    t_lc, d_lc, l_lc = loadcell.get_full_history()
    t_m, p_val, v_val = motion_tracker.get_full_history()

    if not t_lc:
        print("[Plot Save Error] No load cell data collected to plot.")
        return

    if motion_start is None:
        motion_start = t_lc[0]
    if motion_end is None:
        motion_end = t_lc[-1]

    t_window_start = max(0.0, motion_start - 30.0)
    t_window_end = motion_end + 30.0

    # Filter Load Cell Data
    lc_indices = [i for i, t in enumerate(t_lc) if t_window_start <= t <= t_window_end]
    if not lc_indices:
        lc_indices = range(len(t_lc))

    plot_t = [t_lc[i] for i in lc_indices]
    plot_d = [d_lc[i] for i in lc_indices]
    plot_l = [l_lc[i] for i in lc_indices]

    # Filter & Interpolate Position and Velocity across the same timestamp range
    plot_p = []
    plot_v = []
    for t in plot_t:
        if t_m:
            idx = np.searchsorted(t_m, t, side='right') - 1
            idx = max(0, min(idx, len(p_val) - 1))
            plot_p.append(p_val[idx])
            plot_v.append(v_val[idx])
        else:
            plot_p.append(0.0)
            plot_v.append(0.0)

    # Prompt user for filename or fallback to motion_YYYYMMDD_HHMMSS
    default_filename = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    print("\n" + "=" * 75)
    print("SAVING SUMMARY PLOT")
    print("=" * 75)
    user_name = input(f"Enter filename for the plot [press Enter for '{default_filename}']: ").strip()

    if user_name:
        if not user_name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            user_name += ".png"
        final_filename = user_name
    else:
        final_filename = default_filename

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, final_filename)

    fig_summary, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig_summary.suptitle("Tow Tank Test Run - ±30s Motion Summary", fontsize=14, fontweight='bold')

    ax1.plot(plot_t, plot_d, color='#d62728', lw=1.5, label="Drag Force (N)")
    ax1.axvspan(motion_start, motion_end, color='gray', alpha=0.15, label='Active Motion Window')
    ax1.set_ylabel("Drag Force (N)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(plot_t, plot_l, color='#1f77b4', lw=1.5, label="Lift Force (N)")
    ax2.axvspan(motion_start, motion_end, color='gray', alpha=0.15, label='Active Motion Window')
    ax2.set_ylabel("Lift Force (N)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    ax3.plot(plot_t, plot_p, color='#2ca02c', lw=1.8, label="Gantry Position (mm)")
    ax3.axvspan(motion_start, motion_end, color='gray', alpha=0.15, label='Active Motion Window')
    ax3.set_ylabel("Position (mm)")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    ax4.plot(plot_t, plot_v, color='#9467bd', lw=1.8, label="Signed Velocity (+Fwd / -Bwd)")
    ax4.axvspan(motion_start, motion_end, color='gray', alpha=0.15, label='Active Motion Window')
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Velocity (mm/s)")
    ax4.legend(loc="upper right")
    ax4.grid(True)

    ax4.set_xlim(t_window_start, t_window_end)
    fig_summary.tight_layout()

    fig_summary.savefig(output_path, dpi=300)
    plt.close(fig_summary)
    print(f"\n[Saved Plot] Motion window plot (±30s) saved to:\n -> {output_path}")


def main():
    try:
        scale_params = load_scale_factors(CALIBRATION_FILE_PATH)
        print(f"Loaded calibration scale magnitudes from: {CALIBRATION_FILE_PATH}")
    except Exception as e:
        print(f"Calibration File Error: {e}")
        return

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
        "motion_start_time": None,
        "motion_end_time": None,
        "stop_polling": False
    }

    try:
        # 1. Initialize Load Cell Hardware
        print(f"\nConnecting to Load Cell Arduino on {LOADCELL_PORT}...")
        loadcell_ser = serial.Serial(LOADCELL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)  # Reset settling time

        # 2. Phase 1: Execute 60s Warm-Up & Drift Check
        run_loadcell_warmup(loadcell_ser)

        # 3. Phase 2: Determine Channel Polarity & Scale Signs
        final_drag_scale = determine_channel_polarity(
            ser=loadcell_ser,
            channel_name="drag",
            base_scale_magnitude=scale_params["drag_scale"]
        )

        final_lift_scale = determine_channel_polarity(
            ser=loadcell_ser,
            channel_name="lift",
            base_scale_magnitude=scale_params["lift_scale"]
        )

        # 4. Phase 3: Wait 60s for Ringing and Sloshing to Decay
        wait_for_settling(loadcell_ser, duration_sec=SETTLING_DURATION_SEC)

        # 5. Phase 4: Calculate Clean Live Zero Tare Baseline
        live_drag_tare, live_lift_tare = measure_live_tare(loadcell_ser, sample_count=TARE_SAMPLE_COUNT)

        active_params = {
            "drag_tare": live_drag_tare,
            "drag_scale": final_drag_scale,
            "lift_tare": live_lift_tare,
            "lift_scale": final_lift_scale
        }

        # Align global time baseline
        shared_state["global_start_time"] = time.time()

        # 6. Start Live Load Cell Reader Thread
        loadcell = LoadCellReader(loadcell_ser, cal_params=active_params, maxlen=BUFFER_SIZE)
        loadcell.start()

        # 7. Connect Grbl Controller & Start Streaming + Polling Threads
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
            args=(grbl, commands, shared_state), 
            daemon=True
        )
        gcode_thread.start()

        # 8. Setup Matplotlib 4-Subplot Live Telemetry Window
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        fig, (ax_drag, ax_lift, ax_pos, ax_vel) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig.canvas.manager.set_window_title("Tow Tank Live Telemetry (Force, Position & Velocity)")

        line_drag, = ax_drag.plot([], [], color='#d62728', lw=1.8, label="Drag Force (N)")
        line_lift, = ax_lift.plot([], [], color='#1f77b4', lw=1.8, label="Lift Force (N)")
        line_pos, = ax_pos.plot([], [], color='#2ca02c', lw=1.8, label="Gantry Position (mm)")
        line_vel, = ax_vel.plot([], [], color='#9467bd', lw=1.8, label="Signed Velocity (+Fwd / -Bwd)")

        ax_drag.set_ylabel("Drag Force (N)")
        ax_drag.legend(loc="upper right")
        ax_drag.grid(True)

        ax_lift.set_ylabel("Lift Force (N)")
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
                margin_d = (max_d - min_d) * 0.05 or 0.1
                ax_drag.set_ylim(min_d - margin_d, max_d + margin_d)

                min_l, max_l = min(l), max(l)
                margin_l = (max_l - min_l) * 0.05 or 0.1
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
        
        # Emergency stop and disconnect
        if grbl:
            grbl.emergency_stop()
            grbl.close()
        if loadcell:
            loadcell.stop()
        if loadcell_ser and loadcell_ser.is_open:
            loadcell_ser.close()

        # Generate and save the final ±30s summary plot
        if loadcell and motion_tracker:
            save_trimmed_summary_plot(
                loadcell=loadcell,
                motion_tracker=motion_tracker,
                motion_start=shared_state.get("motion_start_time"),
                motion_end=shared_state.get("motion_end_time"),
                output_dir=WORKING_DIR
            )

        print("Test ended. All serial connections closed.")


if __name__ == '__main__':
    main()
