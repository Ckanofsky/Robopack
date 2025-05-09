# follow_person_main.py

import cv2
import time # For potential FPS calculation or delays

# Import from our custom modules
try:
    import detect_objects_library as detector
    import robot_motors
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'detect_objects_library.py' and 'robot_motors.py' are in the same directory or Python path.")
    exit()

# --- Configuration Constants for Robot Control ---
# Proportional gain for turning. Adjust this to change how sharply the robot turns.
KP_TURN = 0.5  # Lower for smoother, higher for more aggressive turning.
# Maximum power for individual motors (e.g., if motors take values from -100 to 100)
MAX_MOTOR_POWER = 70 # Max power for turning/driving
# Base forward speed when the target is centered or when moving generally.
# Set to 0 if you only want the robot to turn and not move forward.
BASE_FORWARD_SPEED = 25 # Speed to move forward when person is centered or while turning
# Horizontal deadzone for the target (percentage of frame width).
# If the person's center is within this % of the frame center, consider it centered.
FRAME_CENTER_DEADZONE_PERCENT_X = 0.10 # 10% deadzone

def main():
    """
    Main function to run the person following robot simulation.
    """
    if detector.model is None:
        print("Exiting application as YOLO model failed to load in the library.")
        return

    # Initialize robot motors
    motors = robot_motors.RobotMotors(max_power=MAX_MOTOR_POWER)

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open video device.")
        motors.stop()
        return

    # Get frame dimensions (once)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x = frame_width / 2
    # frame_center_y = frame_height / 2 # Not used for horizontal following

    # Calculate deadzone in pixels
    deadzone_pixels_x = frame_width * FRAME_CENTER_DEADZONE_PERCENT_X

    print("Starting person following simulation...")
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print(f"Targeting 'person1'. Robot will try to keep 'person1' in the center.")
    print(f"Frame center X: {frame_center_x}, Deadzone: +/- {deadzone_pixels_x:.2f} pixels.")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Process the frame to detect objects
            # This updates detector.current_persons and detector.current_cars
            detector.process_frame_detections(frame, frame_width, frame_height)

            # Try to get "person1" (the person closest to the frame center)
            person1 = detector.get_object_by_id("person1")

            left_motor_power = 0
            right_motor_power = 0

            if person1:
                person_center_x, _ = person1.get_center()
                error_x = person_center_x - frame_center_x

                # --- Robot Control Logic ---
                if abs(error_x) > deadzone_pixels_x:
                    # Person is off-center, need to turn
                    turn_signal = KP_TURN * error_x
                    
                    # Apply turn signal to motors.
                    # This logic makes the robot turn towards the error.
                    # If error_x is positive (person to the right), turn_signal is positive.
                    # We want to increase right motor or decrease left motor (or both) to turn right.
                    # Let's try: left = base - turn, right = base + turn.
                    # Clamping will be handled by motor class.
                    left_motor_power = BASE_FORWARD_SPEED - turn_signal
                    right_motor_power = BASE_FORWARD_SPEED + turn_signal
                    
                    print(f"  Targeting person1: CenterX={person_center_x:.2f}, ErrorX={error_x:.2f}, TurnSignal={turn_signal:.2f}")

                else:
                    # Person is centered, move forward (if base speed is > 0)
                    left_motor_power = BASE_FORWARD_SPEED
                    right_motor_power = BASE_FORWARD_SPEED
                    print(f"  Targeting person1: Centered (ErrorX={error_x:.2f}). Moving forward.")
                
                motors.set_motors(left_motor_power, right_motor_power)

            else:
                # No "person1" detected, stop the robot
                print("  No 'person1' detected. Stopping.")
                motors.stop()

            # --- Drawing Detections on Frame ---
            # Draw all detected persons
            for p_obj in detector.current_persons:
                x1 = int(p_obj.center_x - p_obj.width / 2)
                y1 = int(p_obj.center_y - p_obj.height / 2)
                x2 = int(p_obj.center_x + p_obj.width / 2)
                y2 = int(p_obj.center_y + p_obj.height / 2)
                label = f"{p_obj.id} ({p_obj.confidence*100:.0f}%)"
                color = (0, 255, 0) # Green for persons
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw all detected cars (optional, but good for visualization)
            for c_obj in detector.current_cars:
                x1 = int(c_obj.center_x - c_obj.width / 2)
                y1 = int(c_obj.center_y - c_obj.height / 2)
                x2 = int(c_obj.center_x + c_obj.width / 2)
                y2 = int(c_obj.center_y + c_obj.height / 2)
                label = f"{c_obj.id} ({c_obj.confidence*100:.0f}%)"
                color = (255, 0, 0) # Blue for cars
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center line and deadzone for visualization
            cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), frame_height), (255, 255, 0), 1) # Cyan center line
            dz_left = int(frame_center_x - deadzone_pixels_x)
            dz_right = int(frame_center_x + deadzone_pixels_x)
            cv2.line(frame, (dz_left, 0), (dz_left, frame_height), (0, 255, 255), 1) # Yellow deadzone line
            cv2.line(frame, (dz_right, 0), (dz_right, frame_height), (0, 255, 255), 1) # Yellow deadzone line


            # Display the frame
            cv2.imshow('Person Follower Simulation', frame)

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # time.sleep(0.01) # Optional small delay 

    finally:
        # Release resources
        print("Releasing resources...")
        motors.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Application finished.")

if __name__ == "__main__":
    main()
