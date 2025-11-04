
"""
Diagnose LATCH timing issues with the CM-550 controller.
Tests different send rates and timing patterns.
"""
import serial
import time
import numpy as np

# Remocon packet format
def _checksum_byte(x): 
    return (~x) & 0xFF

def _pack_remocon(value16: int) -> bytes:
    lb = value16 & 0xFF
    hb = (value16 >> 8) & 0xFF
    return bytes([0xFF, 0x55, lb, _checksum_byte(lb), hb, _checksum_byte(hb)])

def test_latch_timing(port="COM3", baud=57600):
    """
    Test different timing patterns for sending 12 joint commands + LATCH
    """
    print("=" * 70)
    print("CM-550 LATCH TIMING DIAGNOSTIC")
    print("=" * 70)
    
    ser = serial.Serial(port, baudrate=baud, timeout=0.05)
    time.sleep(0.5)  # Let connection stabilize
    
    # Motor mapping
    CH_MAP_12 = [3, 1, 7, 5, 12, 9, 4, 2, 8, 6, 11, 10]
    
    # Test configurations
    test_configs = [
        ("FAST (0.0008s/packet)", 0.0008),
        ("MEDIUM (0.002s/packet)", 0.002),
        ("SLOW (0.005s/packet)", 0.005),
        ("VERY SLOW (0.01s/packet)", 0.01),
    ]
    
    for test_name, delay in test_configs:
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")
        
        # Send 3 cycles
        for cycle in range(3):
            print(f"\nCycle {cycle + 1}:")
            
            # Target position: oscillate between 2048±200 ticks
            offset = 200 if cycle % 2 == 0 else -200
            
            # Measure total transmission time
            start_time = time.time()
            
            # Send 12 motor commands
            for joint_idx in range(12):
                motor_id = CH_MAP_12[joint_idx] & 0x0F
                ticks = (2048 + offset) & 0x0FFF
                value = (motor_id << 12) | ticks
                
                packet = _pack_remocon(value)
                ser.write(packet)
                
                if joint_idx == 0:
                    print(f"  First packet sent: Motor {motor_id}, Ticks {ticks}")
                
                time.sleep(delay)
            
            # Send LATCH
            latch_value = (15 << 12) | 0
            ser.write(_pack_remocon(latch_value))
            
            transmission_time = time.time() - start_time
            
            print(f"  LATCH sent after {transmission_time*1000:.1f}ms")
            print(f"  Total: 13 packets in {transmission_time*1000:.1f}ms")
            print(f"  Rate: {transmission_time/13*1000:.2f}ms per packet")
            
            # Wait to see if movement happens
            time.sleep(0.1)
        
        print(f"\nWaiting 1 second before next test...")
        time.sleep(1.0)
    
    # Test burst mode (no delays between packets)
    print(f"\n{'='*70}")
    print("TEST: BURST MODE (no delays)")
    print("='*70}")
    
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        offset = 200 if cycle % 2 == 0 else -200
        
        start_time = time.time()
        
        # Send all 12 packets as fast as possible
        for joint_idx in range(12):
            motor_id = CH_MAP_12[joint_idx] & 0x0F
            ticks = (2048 + offset) & 0x0FFF
            value = (motor_id << 12) | ticks
            ser.write(_pack_remocon(value))
        
        # LATCH
        ser.write(_pack_remocon((15 << 12) | 0))
        
        transmission_time = time.time() - start_time
        print(f"  Burst sent in {transmission_time*1000:.2f}ms")
        time.sleep(0.1)
    
    # Test with explicit flush
    print(f"\n{'='*70}")
    print("TEST: WITH SERIAL FLUSH")
    print("=" * 70)
    
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")
        offset = 200 if cycle % 2 == 0 else -200
        
        for joint_idx in range(12):
            motor_id = CH_MAP_12[joint_idx] & 0x0F
            ticks = (2048 + offset) & 0x0FFF
            value = (motor_id << 12) | ticks
            ser.write(_pack_remocon(value))
            time.sleep(0.0008)
        
        # LATCH
        ser.write(_pack_remocon((15 << 12) | 0))
        ser.flush()  # Force send immediately
        
        print(f"  Commands sent with flush()")
        time.sleep(0.1)
    
    ser.close()
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nOBSERVATIONS:")
    print("  - Did the robot move during any test?")
    print("  - Which timing worked best?")
    print("  - Did burst mode work?")
    print("  - Did flush() help?")

if __name__ == "__main__":
    import sys
    
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    
    print(f"\nConnecting to {port}...")
    print("Make sure the robot is powered on and ready!\n")
    
    try:
        test_latch_timing(port)
    except serial.SerialException as e:
        print(f"\n❌ ERROR: {e}")
        print(f"   Check that {port} is correct and the robot is connected.")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
