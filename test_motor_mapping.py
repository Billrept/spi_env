"""
Test script to verify motor ID to physical leg mapping.
This tests the ACTUAL motor IDs (1-12) to see which physical leg moves.
"""
import time
from textimu_link import Remocon12Link, CH_MAP_12

PORT = "COM3"

def test_motor_ids_directly():
    """Test each motor ID (1-12) directly to verify physical wiring."""
    link = Remocon12Link(port=PORT)
    print("\n" + "="*70)
    print("MOTOR ID TO PHYSICAL LEG MAPPING TEST")
    print("="*70)
    print("\nThis will move each Motor ID (1-12) one at a time.")
    print("For each motor, observe which physical leg moves and in what direction.")
    print("\nPhysical robot layout (looking from above):")
    print("         FRONT ↑")
    print("    FL         FR")
    print("    ML         MR")
    print("    RL         RR")
    print("\nExpected mapping (based on CH_MAP):")
    print("  Motor 1  → FRH (Front-Right Horizontal)")
    print("  Motor 2  → FRV (Front-Right Vertical)")
    print("  Motor 3  → FLH (Front-Left Horizontal)")
    print("  Motor 4  → FLV (Front-Left Vertical)")
    print("  Motor 5  → MRH (Middle-Right Horizontal)")
    print("  Motor 6  → MRV (Middle-Right Vertical)")
    print("  Motor 7  → MLH (Middle-Left Horizontal)")
    print("  Motor 8  → MLV (Middle-Left Vertical)")
    print("  Motor 9  → RRH (Rear-Right Horizontal)")
    print("  Motor 10 → RRV (Rear-Right Vertical)")
    print("  Motor 11 → RLH (Rear-Left Horizontal)")
    print("  Motor 12 → RLV (Rear-Left Vertical)")
    print("\n" + "="*70)
    
    time.sleep(2)  # Wait for IMU to stabilize
    
    # Test each motor ID directly
    for motor_id in range(1, 13):
        print(f"\n{'='*70}")
        print(f"Testing Motor ID: {motor_id}")
        print(f"{'='*70}")
        
        # Find which joint_idx maps to this motor_id
        try:
            joint_idx = CH_MAP_12.index(motor_id)
            joint_names = ["FLH", "FRH", "MLH", "MRH", "RLH", "RRH", 
                          "FLV", "FRV", "MLV", "MRV", "RLV", "RRV"]
            print(f"According to CH_MAP: Motor {motor_id} = {joint_names[joint_idx]}")
        except:
            print(f"Motor {motor_id} not in CH_MAP!")
            continue
        
        # Create command with only this joint at +0.4 rad
        q_rad12 = [0.0] * 12
        q_rad12[joint_idx] = 0.4
        
        print(f"\nSending command: q_rad12[{joint_idx}] = +0.4 rad")
        print(f"This should move: {joint_names[joint_idx]}")
        link.send_joint_targets_rad12(q_rad12, debug=False)
        
        response = input("\nWhich physical leg moved? (e.g., 'FL', 'FR', 'ML', etc.) or 'none': ")
        direction = input("Direction? (H=horizontal, V=vertical): ")
        
        actual = f"{response.upper()}{direction.upper()}"
        expected = joint_names[joint_idx]
        
        if actual == expected:
            print(f"✅ CORRECT: Motor {motor_id} = {actual}")
        else:
            print(f"❌ MISMATCH: Motor {motor_id} expected {expected}, but moved {actual}")
        
        # Return to center
        q_rad12[joint_idx] = 0.0
        link.send_joint_targets_rad12(q_rad12)
        time.sleep(0.5)
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    link.close()

if __name__ == "__main__":
    test_motor_ids_directly()
