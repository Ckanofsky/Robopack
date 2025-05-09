# robot_motors.py

class RobotMotors:
    """
    A simple class to simulate controlling two motors for a tank drive robot.
    For now, it just prints the power values assigned to each motor.
    """
    def __init__(self, max_power=100):
        """
        Initializes the motor controller.

        Args:
            max_power (int): The maximum absolute power value for the motors.
                             Power values will be clamped to [-max_power, max_power].
        """
        self.max_power = abs(max_power)
        self.left_motor_power = 0
        self.right_motor_power = 0
        print(f"RobotMotors initialized. Max power: +/- {self.max_power}. Motors are currently STOPPED.")

    def _clamp_power(self, power):
        """Helper function to clamp power to the defined min/max range."""
        return max(-self.max_power, min(self.max_power, power))

    def set_motors(self, left_power, right_power):
        """
        Sets the power for the left and right motors.

        Args:
            left_power (float): Power for the left motor.
            right_power (float): Power for the right motor.
        """
        self.left_motor_power = self._clamp_power(left_power)
        self.right_motor_power = self._clamp_power(right_power)

        print(f"MOTOR CMD -> Left: {self.left_motor_power:.2f}, Right: {self.right_motor_power:.2f}")
        # In a real robot, you would send these commands to the motor hardware here.

    def stop(self):
        """Stops both motors."""
        self.set_motors(0, 0)
        # print("MOTORS STOPPED") # Covered by set_motors print

# Example usage (if you run this file directly)
if __name__ == "__main__":
    motors = RobotMotors(max_power=100)

    print("\nTesting motor commands:")
    motors.set_motors(50, 50)    # Move forward
    motors.set_motors(-50, -50)  # Move backward
    motors.set_motors(70, -70)   # Turn (e.g., right pivot)
    motors.set_motors(120, -120) # Test clamping (should be 100, -100)
    motors.stop()                # Stop
