import serial

ser = serial.Serial("COM3", 57600, timeout=1)

print("Port opened:", ser.name)
print("Reading...")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print(">>", line)
