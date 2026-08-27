import serial
import time

# --- Configuration ---
SERIAL_PORT = 'COM9'  # Update to your actual port (e.g., '/dev/ttyACM0' on Mac/Linux)
BAUD_RATE = 115200
SAMPLING_INTERVAL = 0.1  # 100ms delay = ~10 Hz sampling rate
SAMPLES_30_SEC = 300     # 30 seconds * 10 samples/sec = 300 samples per block

def read_serial_line(ser):
    """Safely reads and parses the clean dual raw values (drag, lift)."""
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and ',' in line:
                parts = line.split(',')
                return float(parts[0]), float(parts[1])  # Returns (drag_raw, lift_raw)
    except (ValueError, IndexError):
        pass
    return None

def run_30s_tare(ser, description="Gathering baseline"):
    """Runs a 30-second average block and returns average raw values for both axes."""
    print(f"Starting 30-second block: {description}...")
    ser.reset_input_buffer()
    drag_samples = []
    lift_samples = []
    
    while len(drag_samples) < SAMPLES_30_SEC:
        raw_vals = read_serial_line(ser)
        if raw_vals is not None:
            drag_raw, lift_raw = raw_vals
            drag_samples.append(drag_raw)
            with_lift = lift_samples.append(lift_raw)
            print(f"Progress: {len(drag_samples)}/{SAMPLES_30_SEC} samples | Drag: {drag_raw:.1f} | Lift: {lift_raw:.1f}      ", end='\r')
        time.sleep(SAMPLING_INTERVAL)
        
    print("\n" + "-"*60)
    avg_drag = sum(drag_samples) / len(drag_samples)
    avg_lift = sum(lift_samples) / len(lift_samples)
    return avg_drag, avg_lift

def main():
    print("--- Dual-Axis Load Cell Verification Suite (Drag & Lift) ---")
    
    print(f"\nConnecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow Arduino time to reset
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Error connecting to serial port: {e}")
        return

    # =========================================================================
    # STEP 1: LIFT CALIBRATION AT 0 DEGREES
    # =========================================================================
    print("\n=== PHASE 1: LIFT AXIS CALIBRATION (0 DEGREES) ===")
    input("Set up the hardware at 0 DEGREES (completely empty). Press [ENTER] to tare zero baseline...")
    zero_drag_0, zero_lift_0 = run_30s_tare(ser, "0° Empty Baseline")
    print(f"Stored 0° Baselines -> Drag Zero: {zero_drag_0:.2f} | Lift Zero: {zero_lift_0:.2f}")

    try:
        cal_weight_lift = float(input("\nEnter known LIFT calibration mass (grams): "))
    except ValueError:
        print("Invalid input. Defaulting to 50g.")
        cal_weight_lift = 50.0

    input(f"--> PLACE YOUR {cal_weight_lift}g WEIGHT ON THE LIFT AXIS NOW. Press [ENTER] to record factor...")
    loaded_drag_0, loaded_lift_0 = run_30s_tare(ser, "0° Lift Loaded Baseline")
    
    try:
        lift_delta = loaded_lift_0 - zero_lift_0
        if lift_delta == 0: raise ZeroDivisionError
        lift_cal_factor = cal_weight_lift / lift_delta
        print(f"\n[SUCCESS] Lift Calibration Factor Locked: {lift_cal_factor:.8f}")
    except ZeroDivisionError:
        print("\n[ERROR] No variance on Lift axis. Scaling forced to 1.0")
        lift_cal_factor = 1.0

    # =========================================================================
    # STEP 2: DRAG CALIBRATION AT 90 DEGREES
    # =========================================================================
    print("\n=== PHASE 2: DRAG AXIS CALIBRATION (90 DEGREES) ===")
    input("Rotate setup to 90 DEGREES (completely empty). Press [ENTER] to tare new zero baseline...")
    zero_drag_90, zero_lift_90 = run_30s_tare(ser, "90° Empty Baseline")
    print(f"Stored 90° Baselines -> Drag Zero: {zero_drag_90:.2f} | Lift Zero: {zero_lift_90:.2f}")

    try:
        cal_weight_drag = float(input("\nEnter known DRAG calibration mass (grams): "))
    except ValueError:
        print("Invalid input. Defaulting to 50g.")
        cal_weight_drag = 50.0

    input(f"--> PLACE YOUR {cal_weight_drag}g WEIGHT ON THE DRAG AXIS NOW. Press [ENTER] to record factor...")
    loaded_drag_90, loaded_lift_90 = run_30s_tare(ser, "90° Drag Loaded Baseline")
    
    try:
        drag_delta = loaded_drag_90 - zero_drag_90
        if drag_delta == 0: raise ZeroDivisionError
        drag_cal_factor = cal_weight_drag / drag_delta
        print(f"\n[SUCCESS] Drag Calibration Factor Locked: {drag_cal_factor:.8f}")
    except ZeroDivisionError:
        print("\n[ERROR] No variance on Drag axis. Scaling forced to 1.0")
        drag_cal_factor = 1.0

    print("="*60)
    print(f"System Calibration Multipliers Locked:\n -> Drag Scale: {drag_cal_factor:.8f}\n -> Lift Scale: {lift_cal_factor:.8f}")
    input("\nRemove calibration weights. Press [ENTER] to begin custom orientation sweeps...")

    # =========================================================================
    # STEP 3: REUSABLE ORIENTATION SWEEP LOOP
    # =========================================================================
    orientation_count = 1
    
    while True:
        print(f"\n=== ORIENTATION POSITION #{orientation_count} ===")
        
        input(f"Move/rotate hardware to your desired position #{orientation_count} (Keep Empty). Press [ENTER] to run a local zero tare...")
        current_drag_offset, current_lift_offset = run_30s_tare(ser, f"Position #{orientation_count} Local Tare")
        print(f"Saved Position Zero Offsets -> Drag: {current_drag_offset:.2f} | Lift: {current_lift_offset:.2f}")
        print("-"*60)

        # PAUSE POINT: Wait for user to apply the verification force load
        print(f"\n[PAUSED] Local zero offsets are locked for Position #{orientation_count}.")
        input("--> ATTACH YOUR LOAD BRUH. Press [ENTER] to begin continuous dual-axis capture...")

        print(f"\nStarting continuous verification logging for Position #{orientation_count}...")
        print("Press Ctrl+C when you are finished recording data for this position.")
        
        ser.reset_input_buffer()

        block_drag_samples = []
        block_lift_samples = []
        block_count = 1

        try:
            while True:
                raw_vals = read_serial_line(ser)
                if raw_vals is not None:
                    drag_raw, lift_raw = raw_vals
                    
                    # Apply local zero offsets and respective global scaling calibration factors
                    cal_drag = (drag_raw - current_drag_offset) * drag_cal_factor
                    cal_lift = (lift_raw - current_lift_offset) * lift_cal_factor
                    
                    block_drag_samples.append(cal_drag)
                    block_lift_samples.append(cal_lift)
                    
                    if len(block_drag_samples) == SAMPLES_30_SEC:
                        avg_drag_val = sum(block_drag_samples) / SAMPLES_30_SEC
                        avg_lift_val = sum(block_lift_samples) / SAMPLES_30_SEC
                        
                        print(f"[Pos {orientation_count} - Block {block_count:02d}] 30s Avg -> DRAG: {avg_drag_val:.4f}g | LIFT: {avg_lift_val:.4f}g")
                        
                        block_drag_samples.clear()
                        block_lift_samples.clear()
                        block_count += 1
                    else:
                        filled_samples = len(block_drag_samples)
                        print(f"Collecting Data Block {block_count:02d}... Progress: {filled_samples}/{SAMPLES_30_SEC} samples", end='\r')
                
                time.sleep(SAMPLING_INTERVAL)
                
        except KeyboardInterrupt:
            print(f"\nFinished data capture for Position #{orientation_count}.")
            
            next_action = input("\nType 'y' to move to next orientation, or 'q' to quit application: ").strip().lower()
            if next_action == 'q':
                print("Exiting verification session.")
                break
            else:
                orientation_count += 1

    ser.close()
    print("Serial port closed cleanly.")

if __name__ == "__main__":
    main()