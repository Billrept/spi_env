import serial
import threading
import time

PORT = "COM3"   # Replace with your CM-550 USB COM port
BAUD = 57600

# --- Function to send Remocon packets ---
def send_remocon(ser, value):
    lb =  value        & 0xFF
    hb = (value >> 8)  & 0xFF
    pkt = bytes([0xFF, 0x55, lb, (~lb) & 0xFF, hb, (~hb) & 0xFF])
    ser.write(pkt)
    print(f"[TX] Sent Remocon packet: {value}")

# --- Function to continuously read incoming data ---
def read_from_cm550(ser):
    print("[RX] Listening for messages from CM-550...")
    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"[RX] {line}")
        except serial.SerialException:
            print("[RX] Serial port error, stopping reader thread.")
            break
        except Exception as e:
            print("[RX] Error:", e)

# --- Main program ---
def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        print(f"Port opened: {ser.name}")

        # Start a thread to keep reading data
        reader_thread = threading.Thread(target=read_from_cm550, args=(ser,), daemon=True)
        reader_thread.start()

        # Main loop: send test values every few seconds
        while True:
            cmd = input("Enter a number to send (or 'q' to quit): ").strip()
            if cmd.lower() == 'q':
                break
            if cmd.isdigit():
                send_remocon(ser, int(cmd))
            else:
                print("Please enter a valid integer value.")

            time.sleep(0.2)

    except serial.SerialException:
        print("❌ Could not open port. Make sure CM-550 is connected and not in use.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Port closed.")

if __name__ == "__main__":
    main()
