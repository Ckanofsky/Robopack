from gpiozero import PWMOutputDevice
import time

class RobotMotors:
    """
    A class to control two motors for a tank drive robot using BTS7960 Motor Drivers.
    Each motor has its own control via gpiozero on the Raspberry Pi.
    """
    def __init__(self, left_rpwm_pin=12, left_lpwm_pin=18, right_rpwm_pin=13, right_lpwm_pin=19, max_power=100):
        """
        Initializes the motor controller.

        Args:
            left_rpwm_pin (int): GPIO pin for left motor RPWM (forward)
            left_lpwm_pin (int): GPIO pin for left motor LPWM (reverse)
            right_rpwm_pin (int): GPIO pin for right motor RPWM (forward)
            right_lpwm_pin (int): GPIO pin for right motor LPWM (reverse)
            max_power (int): The maximum absolute power value for the motors.
                             Power values will be clamped to [-max_power, max_power].
        """
        self.max_power = abs(max_power)
        self.left_motor_power = 0
        self.right_motor_power = 0
        
        # Pin assignments
        self.left_rpwm_pin = left_rpwm_pin
        self.left_lpwm_pin = left_lpwm_pin
        self.right_rpwm_pin = right_rpwm_pin
        self.right_lpwm_pin = right_lpwm_pin
        
        # Initialize PWM pins for motors
        self.left_rpwm = PWMOutputDevice(self.left_rpwm_pin)
        self.left_lpwm = PWMOutputDevice(self.left_lpwm_pin)
        self.right_rpwm = PWMOutputDevice(self.right_rpwm_pin)
        self.right_lpwm = PWMOutputDevice(self.right_lpwm_pin)
        
        # Set initial power to 0 (stop)
        self.left_rpwm.value = 0
        self.left_lpwm.value = 0
        self.right_rpwm.value = 0
        self.right_lpwm.value = 0
        
        print(f"RobotMotors initialized. Max power: +/- {self.max_power}. Motors are currently STOPPED.")
        print(f"Left motor pins - RPWM: {self.left_rpwm_pin}, LPWM: {self.left_lpwm_pin}")
        print(f"Right motor pins - RPWM: {self.right_rpwm_pin}, LPWM: {self.right_lpwm_pin}")

    def _clamp_power(self, power):
        """Helper function to clamp power to the defined min/max range."""
        return max(-self.max_power, min(self.max_power, power))

    def _set_motor_pwm(self, rpwm, lpwm, power):
        """
        Helper function to set PWM values for a single motor based on power.
        
        Args:
            rpwm: PWMOutputDevice for forward direction
            lpwm: PWMOutputDevice for reverse direction
            power: Power value (0-100)
        """
        duty_cycle = abs(power) / 100
        
        if power >= 0:
            # Forward direction
            rpwm.value = duty_cycle
            lpwm.value = 0
        else:
            # Reverse direction
            rpwm.value = 0
            lpwm.value = duty_cycle

    def set_motors(self, left_power, right_power):
        """
        Sets the power for the left and right motors.

        Args:
            left_power (float): Power for the left motor (-max_power to +max_power).
            right_power (float): Power for the right motor (-max_power to +max_power).
        """
        self.left_motor_power = self._clamp_power(left_power)
        self.right_motor_power = self._clamp_power(right_power)
        
        # Set PWM values for left motor
        self._set_motor_pwm(self.left_rpwm, self.left_lpwm, self.left_motor_power)
        
        # Set PWM values for right motor
        self._set_motor_pwm(self.right_rpwm, self.right_lpwm, self.right_motor_power)
        
        print(f"MOTOR CMD -> Left: {self.left_motor_power:.2f}, Right: {self.right_motor_power:.2f}")

    def stop(self):
        """Stops both motors."""
        self.set_motors(0, 0)

    def cleanup(self):
        """Cleanup GPIO resources."""
        self.stop()
        print("GPIO cleaned up")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors in destructor


# Example usage (if you run this file directly)
if __name__ == "__main__":
    try:
        motors = RobotMotors(max_power=100)
        
        print("\nTesting motor commands:")
        
        motors.set_motors(50, 50)    # Move forward
        time.sleep(2)
        
        motors.set_motors(-50, -50)  # Move backward
        time.sleep(2)
        
        motors.set_motors(70, -70)   # Turn (e.g., right pivot)
        time.sleep(2)
        
        motors.set_motors(120, -120) # Test clamping (should be 100, -100)
        time.sleep(2)
        
        motors.stop()                # Stop
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if 'motors' in locals():
            motors.cleanup()
        print("Test completed")
