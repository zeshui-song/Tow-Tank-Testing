import serial
import serial.tools.list_ports
import time
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
SERIAL_PORT = 'COM9'          # Update to match your port (e.g. '/dev/ttyACM0' or 'COM9')
BAUD_RATE = 115200
SAMPLING_INTERVAL = 0.1       # 100ms delay = ~10 Hz sampling rate
SAMPLES_30_SEC = 300          # 30 seconds * 10 Hz = 300 samples
WARMUP_MIN_DURATION_SEC = 60  # Initial minimum self-heating duration
WARMUP_EXTEND_SEC = 30        # Increment interval if slope threshold exceeded
MAX_STABLE_SLOPE_PCT = 0.10   # Maximum allowable drift slope (% full-scale/min)
MAX_HX711_BITS = 8388607.0    # 24-bit ADC positive full scale for percentage drift
GRAVITY = 9.80665             # m/s^2 standard gravity
CALIBRATION_FILE = r'C:\Users\zsong\Desktop\Tow Tank Testing\calibration.txt'
DEVIATION_THRESHOLD_PCT = 5.0 # Warn if new factor deviates more than 5% from previous

def open_serial_connection(port, baud):
    """Safely opens or re-opens the serial port with Windows USB latency buffers."""
    ser = serial.Serial(port, baud, timeout=1, write_timeout=1)
    ser.set_buffer_size(rx_size=16384, tx_size=4096)
    time.sleep(2)  # Allow Arduino reboot
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser

def reconnect_serial(ser, port, baud, max_retries=5):
    """Attempts to cleanly re-establish communication if a USB drop occurs."""
    print(f"\n[WARNING] Serial connection glitch detected on {port}. Attempting auto-reconnect...")
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(1.0)
            ser = open_serial_connection(port, baud)
            print(f"[RECONNECTED] Successfully restored communication on {port}.")
            return ser
        except Exception as e:
            print(f" Reconnection attempt {attempt}/{max_retries} failed: {e}")
    raise serial.SerialException(f"Failed to reconnect to {port} after {max_retries} attempts.")

def read_serial_line(ser_container):
    """Safely reads dual raw values (drag, lift) with built-in hardware disconnect recovery."""
    ser = ser_container[0]
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and ',' in line:
                parts = line.split(',')
                return float(parts[0]), float(parts[1])  # (drag_raw, lift_raw)
    except (serial.SerialException, PermissionError, OSError):
        ser_container[0] = reconnect_serial(ser_container[0], SERIAL_PORT, BAUD_RATE)
    except (ValueError, IndexError):
        pass
    return None

def safe_flush_buffer(ser_container):
    """Safely discards backlog bytes without triggering Windows driver errors."""
    try:
        ser_container[0].reset_input_buffer()
    except Exception:
        try:
            ser_container[0] = reconnect_serial(ser_container[0], SERIAL_PORT, BAUD_RATE)
        except Exception:
            pass

def wait_for_user_choice(prompt_msg):
    """Halts execution until user inputs 'Y' to proceed or 'Q' to quit."""
    while True:
        user_input = input(f"\n{prompt_msg} [Type 'Y' to proceed, 'Q' to quit]: ").strip().upper()
        if user_input == 'Y':
            return 'PROCEED'
        elif user_input == 'Q':
            print("\n[ABORT] User initiated program exit.")
            sys.exit(0)
        print("Invalid choice. Please type 'Y' to proceed or 'Q' to quit.")

def compute_drift_metrics(samples, time_step=SAMPLING_INTERVAL):
    """Calculates linear drift rate in % FS/min over current sample buffer."""
    if len(samples) < 30:
        return 0.0
    y = np.array(samples)
    x = np.arange(len(y)) * time_step
    slope_bits_per_sec, _ = np.polyfit(x, y, 1)
    slope_pct_per_min = (slope_bits_per_sec * 60.0 / MAX_HX711_BITS) * 100.0
    return slope_pct_per_min

def plot_dual_axis_data(time_arr, drag_series, lift_series, title_str, y_label, scale_factor_N=None, target_val=None, target_label=None):
    """Renders dual-axis acquisition data in Force (N) and Mass (g)."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    
    ax1.plot(time_arr, drag_series, label='Drag Channel', color='crimson', linewidth=1.5)
    ax1.plot(time_arr, lift_series, label='Lift Channel', color='royalblue', linewidth=1.5)
    
    if target_val is not None:
        ax1.axhline(y=target_val, color='forestgreen', linestyle='--', label=target_label or f'Target Nominal ({target_val:.4f})')

    ax1.set_xlabel('Elapsed Time (s)', fontsize=10)
    ax1.set_ylabel(y_label, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    if scale_factor_N is not None:
        ax2 = ax1.twinx()
        y1_min, y1_max = ax1.get_ylim()
        ax2.set_ylim((y1_min / GRAVITY) * 1000.0, (y1_max / GRAVITY) * 1000.0)
        ax2.set_ylabel('Equivalent Mass (g)', fontsize=10, color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen')

    plt.title(title_str, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

def run_settle_period(ser_container, duration_sec=30, description="Settling"):
    """Allows physical ring-down/settling while continuously draining the buffer."""
    print(f"\n[SETTLING] Allowing load cells to settle ({description}) for {duration_sec}s...")
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        remaining = duration_sec - int(time.time() - start_time)
        safe_flush_buffer(ser_container)
        print(f" Settling... {remaining:02d}s remaining  ", end='\r')
        time.sleep(0.5)
    safe_flush_buffer(ser_container)
    print(f"\n Settle period complete.")

def collect_30s_raw(ser_container, axis_name):
    """Collects 300 raw bit samples, computes averages, and displays a raw bits plot."""
    print(f"\n[DATA CAPTURE] Recording 30s baseline ({axis_name})...")
    safe_flush_buffer(ser_container)
    drag_samples = []
    lift_samples = []
    time_series = []
    
    start_collect = time.time()
    while len(drag_samples) < SAMPLES_30_SEC:
        raw_vals = read_serial_line(ser_container)
        if raw_vals is not None:
            drag_raw, lift_raw = raw_vals
            drag_samples.append(drag_raw)
            lift_samples.append(lift_raw)
            time_series.append(time.time() - start_collect)
            print(f" Progress: {len(drag_samples):03d}/{SAMPLES_30_SEC} | Drag Bits: {drag_raw:.1f} | Lift Bits: {lift_raw:.1f}    ", end='\r')
        time.sleep(SAMPLING_INTERVAL)
        
    avg_drag = sum(drag_samples) / len(drag_samples)
    avg_lift = sum(lift_samples) / len(lift_samples)
    print(f"\n Complete -> Avg Drag: {avg_drag:.2f} bits | Avg Lift: {avg_lift:.2f} bits")

    plot_dual_axis_data(
        time_series, 
        drag_samples, 
        lift_samples, 
        f"30-Second Data Acquisition: {axis_name}", 
        "Raw ADC Bits"
    )

    return avg_drag, avg_lift, drag_samples, lift_samples, time_series

def load_previous_calibration():
    """Loads existing calibration data if available from the designated folder."""
    if not os.path.exists(CALIBRATION_FILE):
        return None
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not parse existing '{CALIBRATION_FILE}': {e}")
        return None

def save_calibration_file(data):
    """Saves all calibration factors and offsets to the designated calibration.txt path."""
    try:
        target_dir = os.path.dirname(CALIBRATION_FILE)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n[SUCCESS] Calibration saved directly to '{CALIBRATION_FILE}'.")
    except Exception as e:
        print(f"\n[ERROR] Failed to save calibration: {e}")

def check_deviation(name, old_val, new_val):
    """Flags deviations between previous and newly calculated calibration factors."""
    if old_val is None or old_val == 0:
        return
    diff_pct = abs((new_val - old_val) / old_val) * 100.0
    print(f" -> {name} Comparison: Previous = {old_val:.8e} | New = {new_val:.8e} (Δ: {diff_pct:.2f}%)")
    if diff_pct > DEVIATION_THRESHOLD_PCT:
        print(f"    [!] WARNING: {name} deviated by more than {DEVIATION_THRESHOLD_PCT}% from previous session!")

def prompt_calibration_mass(axis_name, default_mass):
    """Prompts the user to reuse or specify a new calibration mass in grams."""
    print(f"\n--- {axis_name.upper()} CALIBRATION MASS CONFIGURATION ---")
    if default_mass is not None:
        choice = input(f"Previous mass for {axis_name} was {default_mass:.2f} g. Use same mass? (Y/n): ").strip().lower()
        if choice in ['', 'y', 'yes']:
            return default_mass
            
    while True:
        try:
            val = float(input(f"Enter {axis_name} calibration mass in grams (g): "))
            if val > 0:
                return val
            print("Mass must be greater than zero.")
        except ValueError:
            print("Invalid numeric value.")

def main():
    print("=" * 75)
    print("   DUAL-AXIS LOAD CELL CALIBRATION (LIFT & DRAG)   ")
    print("=" * 75)
    print(f"[PATH] Active Calibration Target: {CALIBRATION_FILE}")
    print("[NOTE] Press Ctrl+C at ANY time to cleanly terminate the program.")

    ser_container = [None]

    try:
        # 0. Check Existing Calibration History
        prev_cal = load_previous_calibration()
        prev_drag_mass = None
        prev_lift_mass = None

        if prev_cal:
            print(f"\n[INFO] Loaded reference history from '{CALIBRATION_FILE}' ({prev_cal.get('timestamp', 'Unknown')}).")
            prev_lift_mass = prev_cal.get('lift_calibration_mass_g')
            prev_drag_mass = prev_cal.get('drag_calibration_mass_g')
        else:
            print(f"\n[INFO] No prior '{CALIBRATION_FILE}' found. A new one will be created.")

        # 1. Connect to Hardware
        print(f"\nConnecting to {SERIAL_PORT} @ {BAUD_RATE} baud...")
        ser_container[0] = open_serial_connection(SERIAL_PORT, BAUD_RATE)

        # 2. Self-Heating & Stabilization Warm-up
        print("\n" + "=" * 75)
        print("PHASE 1: LOAD CELL SELF-HEATING & THERMAL STABILIZATION")
        print("=" * 75)
        print(f"Running minimum warmup of {WARMUP_MIN_DURATION_SEC}s. Slope stability limit: {MAX_STABLE_SLOPE_PCT:.3f}% FS/min.")

        warmup_target_duration = WARMUP_MIN_DURATION_SEC
        drag_window = []
        lift_window = []
        time_series_warmup = []
        start_warmup = time.time()

        while True:
            elapsed = time.time() - start_warmup
            raw_vals = read_serial_line(ser_container)
            
            if raw_vals is not None:
                drag_raw, lift_raw = raw_vals
                drag_window.append(drag_raw)
                lift_window.append(lift_raw)
                time_series_warmup.append(elapsed)
                
                if len(drag_window) > SAMPLES_30_SEC:
                    drag_window.pop(0)
                    lift_window.pop(0)

                drag_slope_pct = compute_drift_metrics(drag_window)
                lift_slope_pct = compute_drift_metrics(lift_window)

                remaining = max(0, int(warmup_target_duration - elapsed))
                print(f" Timer: {remaining:02d}s | Drag: {drag_raw:9.1f} bits (Slope: {drag_slope_pct:+.3f}%/m) | Lift: {lift_raw:9.1f} bits (Slope: {lift_slope_pct:+.3f}%/m)   ", end='\r')

            if elapsed >= warmup_target_duration:
                drag_slope_pct = compute_drift_metrics(drag_window)
                lift_slope_pct = compute_drift_metrics(lift_window)
                
                if abs(drag_slope_pct) <= MAX_STABLE_SLOPE_PCT and abs(lift_slope_pct) <= MAX_STABLE_SLOPE_PCT:
                    print(f"\n\n[STABILITY ACHIEVED] Both channels settled under {MAX_STABLE_SLOPE_PCT:.3f}%/min threshold.")
                    break
                else:
                    print(f"\n\n[DRIFT DETECTED] Drag: {drag_slope_pct:+.3f}%/m | Lift: {lift_slope_pct:+.3f}%/m (Limit: ±{MAX_STABLE_SLOPE_PCT:.3f}%/m).")
                    choice = wait_for_user_choice(f"Extend warmup by +{WARMUP_EXTEND_SEC}s?")
                    if choice == 'PROCEED':
                        warmup_target_duration += WARMUP_EXTEND_SEC

            time.sleep(SAMPLING_INTERVAL)

        # Plot warmup raw bits trace
        if len(drag_window) > 0:
            t_plot = np.arange(len(drag_window)) * SAMPLING_INTERVAL
            plot_dual_axis_data(
                t_plot, 
                drag_window, 
                lift_window, 
                "Stabilization Phase: Raw Bits Over Final 30s Window", 
                "Raw ADC Bits"
            )

        # 3. Lift Axis Calibration Sequence (Run First)
        print("\n" + "=" * 75)
        print("PHASE 2: LIFT AXIS CALIBRATION")
        print("=" * 75)
        wait_for_user_choice("Reconfigure setup orientation for LIFT calibration (Ensure axis is completely unloaded).")
        
        run_settle_period(ser_container, duration_sec=30, description="Lift Unloaded")
        _, lift_tare_bits, _, _, _ = collect_30s_raw(ser_container, "Lift Empty Tare (Zero Baseline)")
        print(f"Stored -> lift_tare_bits: {lift_tare_bits:.2f} bits")

        lift_cal_mass_g = prompt_calibration_mass("Lift", prev_lift_mass)
        lift_cal_mass_kg = lift_cal_mass_g / 1000.0
        lift_cal_weight_N = lift_cal_mass_kg * GRAVITY

        wait_for_user_choice(f"Attach the {lift_cal_mass_g:.2f}g ({lift_cal_weight_N:.4f} N) calibration weight to the LIFT axis.")
        
        run_settle_period(ser_container, duration_sec=30, description="Lift Loaded")
        _, lift_loaded_bits, drag_during_lift_raw, lift_loaded_raw, lift_time_series = collect_30s_raw(ser_container, "Lift Loaded Raw Bits")

        lift_delta_bits = lift_loaded_bits - lift_tare_bits
        if lift_delta_bits == 0:
            print("[ERROR] Zero bit deflection detected on Lift axis. Scale factors set to 1.0.")
            lift_scale_factor_N = 1.0
            lift_scale_factor_g = 1.0
        else:
            lift_scale_factor_N = lift_cal_weight_N / lift_delta_bits
            lift_scale_factor_g = lift_cal_mass_g / lift_delta_bits

        print(f"\n[LIFT RESULT] Delta: {lift_delta_bits:.2f} bits | Scale Factor: {lift_scale_factor_N:.8e} N/bit ({lift_scale_factor_g:.8e} g/bit)")

        lift_force_samples = (np.array(lift_loaded_raw) - lift_tare_bits) * lift_scale_factor_N
        plot_dual_axis_data(
            lift_time_series,
            np.zeros_like(lift_force_samples),
            lift_force_samples,
            "Lift Loaded Verification (Calibrated Force & Mass)",
            "Measured Lift Force (N)",
            scale_factor_N=lift_scale_factor_N,
            target_val=lift_cal_weight_N,
            target_label=f"Applied Mass: {lift_cal_mass_g:.2f} g ({lift_cal_weight_N:.4f} N)"
        )

        # 4. Drag Axis Calibration Sequence (Run Second)
        print("\n" + "=" * 75)
        print("PHASE 3: DRAG AXIS CALIBRATION")
        print("=" * 75)
        wait_for_user_choice("Reconfigure setup orientation for DRAG calibration (Ensure axis is completely unloaded).")
        
        run_settle_period(ser_container, duration_sec=30, description="Drag Unloaded")
        drag_tare_bits, _, _, _, _ = collect_30s_raw(ser_container, "Drag Empty Tare (Zero Baseline)")
        print(f"Stored -> drag_tare_bits: {drag_tare_bits:.2f} bits")

        drag_cal_mass_g = prompt_calibration_mass("Drag", prev_drag_mass)
        drag_cal_mass_kg = drag_cal_mass_g / 1000.0
        drag_cal_weight_N = drag_cal_mass_kg * GRAVITY

        wait_for_user_choice(f"Attach the {drag_cal_mass_g:.2f}g ({drag_cal_weight_N:.4f} N) calibration weight to the DRAG axis.")
        
        run_settle_period(ser_container, duration_sec=30, description="Drag Loaded")
        drag_loaded_bits, _, drag_loaded_raw, lift_during_drag_raw, drag_time_series = collect_30s_raw(ser_container, "Drag Loaded Raw Bits")

        drag_delta_bits = drag_loaded_bits - drag_tare_bits
        if drag_delta_bits == 0:
            print("[ERROR] Zero bit deflection detected on Drag axis. Scale factors set to 1.0.")
            drag_scale_factor_N = 1.0
            drag_scale_factor_g = 1.0
        else:
            drag_scale_factor_N = drag_cal_weight_N / drag_delta_bits
            drag_scale_factor_g = drag_cal_mass_g / drag_delta_bits

        print(f"\n[DRAG RESULT] Delta: {drag_delta_bits:.2f} bits | Scale Factor: {drag_scale_factor_N:.8e} N/bit ({drag_scale_factor_g:.8e} g/bit)")

        final_drag_force = (np.array(drag_loaded_raw) - drag_tare_bits) * drag_scale_factor_N
        final_lift_force = (np.array(lift_during_drag_raw) - lift_tare_bits) * lift_scale_factor_N
        plot_dual_axis_data(
            drag_time_series,
            final_drag_force,
            final_lift_force,
            "Drag Loaded Verification (Calibrated Force & Mass)",
            "Measured Force (N)",
            scale_factor_N=drag_scale_factor_N,
            target_val=drag_cal_weight_N,
            target_label=f"Applied Mass: {drag_cal_mass_g:.2f} g ({drag_cal_weight_N:.4f} N)"
        )

        # 5. Deviation Check & Final Packaging
        print("\n" + "=" * 75)
        print("PHASE 4: VERIFICATION & SUMMARY")
        print("=" * 75)
        if prev_cal:
            check_deviation("Lift Scale Factor (N/bit)", prev_cal.get('lift_scale_factor_N'), lift_scale_factor_N)
            check_deviation("Drag Scale Factor (N/bit)", prev_cal.get('drag_scale_factor_N'), drag_scale_factor_N)

        cal_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gravity_used": GRAVITY,
            "lift_tare_bits": lift_tare_bits,
            "lift_loaded_bits": lift_loaded_bits,
            "lift_calibration_mass_g": lift_cal_mass_g,
            "lift_calibration_weight_N": lift_cal_weight_N,
            "lift_scale_factor_N": lift_scale_factor_N,
            "lift_scale_factor_g": lift_scale_factor_g,
            "drag_tare_bits": drag_tare_bits,
            "drag_loaded_bits": drag_loaded_bits,
            "drag_calibration_mass_g": drag_cal_mass_g,
            "drag_calibration_weight_N": drag_cal_weight_N,
            "drag_scale_factor_N": drag_scale_factor_N,
            "drag_scale_factor_g": drag_scale_factor_g
        }

        save_calibration_file(cal_results)
        print("\nCalibration session finished successfully.")

    except KeyboardInterrupt:
        print("\n\n[ABORT] Ctrl+C interrupt detected. Halting execution immediately.")

    except Exception as e:
        print(f"\n\n[ERROR] An unexpected runtime error occurred: {e}")

    finally:
        if ser_container[0] is not None and ser_container[0].is_open:
            ser_container[0].close()
            print("Serial port closed cleanly.")
        print("Session ended.\n")

if __name__ == "__main__":
    main()