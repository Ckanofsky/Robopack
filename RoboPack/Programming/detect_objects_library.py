# detect_objects_library.py
# Ensure you have the required packages installed:
# pip install opencv-python ultralytics numpy

import cv2
from ultralytics import YOLO
import numpy as np
import math

# --- YOLO Model Loading ---
# Load the YOLOv8 model. This happens once when the module is imported.
# "yolov8n.pt" is a small and fast model. You can use other versions like "yolov8s.pt", etc.
try:
    model = YOLO("yolov8n.pt")
    MODEL_NAMES = model.names # Class names from the model
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'yolov8n.pt' is accessible or try a different model path.")
    model = None
    MODEL_NAMES = {}

# --- Configuration for print update frequency ---
PROCESS_FRAME_INTERVAL = 15  # Print detection details every N frames
frame_processor_counter = 0  # Counter to track frames for printing

# --- Global lists to store detected objects for the current frame ---
# These lists will be updated by handle_detection_data
current_persons = []
current_cars = []

class DetectedObject:
    """
    Base class for detected objects (Person, Car).
    """
    def __init__(self, obj_id, center_x, center_y, width, height, confidence):
        self.id = obj_id
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.width = float(width)
        self.height = float(height)
        self.area = float(width * height)
        self.confidence = float(confidence)

    def get_center(self):
        """Returns the center coordinates (x, y) of the object."""
        return (self.center_x, self.center_y)

    def get_dimensions(self):
        """Returns the dimensions (width, height) of the object."""
        return (self.width, self.height)

    def get_area(self):
        """Returns the area of the object's bounding box."""
        return self.area

    def __str__(self):
        return (f"ID: {self.id} (Conf: {self.confidence:.2f})\n"
                f"  Center (x, y): ({self.center_x:.2f}, {self.center_y:.2f})\n"
                f"  Dimensions (w, h): ({self.width:.2f}, {self.height:.2f})\n"
                f"  Area: {self.area:.2f} pixels^2")

class Person(DetectedObject):
    """
    Represents a detected person.
    """
    def __init__(self, obj_id_num, center_x, center_y, width, height, confidence):
        super().__init__(f"person{obj_id_num}", center_x, center_y, width, height, confidence)

class Car(DetectedObject):
    """
    Represents a detected car.
    """
    def __init__(self, obj_id_num, center_x, center_y, width, height, confidence):
        super().__init__(f"car{obj_id_num}", center_x, center_y, width, height, confidence)

def list_model_vocabulary():
    """
    Prints the list of class names that the loaded YOLO model can detect.
    """
    if not model:
        print("YOLO model not loaded. Cannot list vocabulary.")
        return
    print("YOLO Model Vocabulary (Class ID: Class Name):")
    for class_id, class_name in MODEL_NAMES.items():
        print(f"  {class_id}: {class_name}")
    print("-" * 30)

def get_object_by_id(object_id_str):
    """
    Retrieves a detected person or car object by its ID from the current frame's detections.

    Args:
        object_id_str (str): The ID of the object (e.g., "person1", "car2").

    Returns:
        Person or Car object if found, None otherwise.
    """
    global current_persons, current_cars
    for p_obj in current_persons:
        if p_obj.id == object_id_str:
            return p_obj
    for c_obj in current_cars:
        if c_obj.id == object_id_str:
            return c_obj
    # Optional: print(f"Object with ID '{object_id_str}' not found in current detections.")
    return None

def process_frame_detections(frame, frame_width, frame_height):
    """
    Processes a single frame for object detection, updates global lists of detected persons and cars,
    and prints their details at a controlled frequency.

    Args:
        frame: The video frame (NumPy array) to process.
        frame_width: Width of the video frame.
        frame_height: Height of the video frame.
    
    Returns:
        results: The raw results from the YOLO model for this frame.
                 This can be useful if the calling script wants more details.
    """
    global frame_processor_counter, current_persons, current_cars, model

    if model is None:
        print("YOLO model is not loaded. Skipping detection.")
        current_persons.clear()
        current_cars.clear()
        return None

    # Clear previous frame's objects
    current_persons.clear()
    current_cars.clear()

    # Run object detection
    try:
        results = model(frame, verbose=False) # verbose=False to reduce console spam from YOLO
    except Exception as e:
        print(f"Error during YOLO model prediction: {e}")
        return None


    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        class_ids_tensor = boxes.cls
        bboxes_xywh_tensor = boxes.xywh  # [center_x, center_y, width, height]
        confidences_tensor = boxes.conf

        # Move tensors to CPU and convert to NumPy arrays
        class_ids = class_ids_tensor.cpu().numpy().astype(int)
        bboxes_xywh = bboxes_xywh_tensor.cpu().numpy()
        confidences = confidences_tensor.cpu().numpy()

        detected_persons_data = [] # To store (distance_to_center, data_tuple)
        detected_cars_data = []    # data_tuple = (center_x, center_y, width, height, confidence)

        for i in range(len(class_ids)):
            class_id = class_ids[i]
            
            # Ensure MODEL_NAMES is populated and class_id is valid
            if not MODEL_NAMES or class_id not in MODEL_NAMES:
                # print(f"Warning: Unknown class_id {class_id} detected.")
                continue # Skip if class_id is not in our model's vocabulary

            class_name = MODEL_NAMES[class_id].lower() # Use lower case for consistent checking

            center_x, center_y, width, height = bboxes_xywh[i]
            confidence = confidences[i]
            
            distance_to_center = math.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)

            if class_name == 'person':
                detected_persons_data.append({
                    'distance': distance_to_center,
                    'data': (center_x, center_y, width, height, confidence)
                })
            elif class_name == 'car':
                detected_cars_data.append({
                    'distance': distance_to_center,
                    'data': (center_x, center_y, width, height, confidence)
                })

        # Sort by distance to center and assign IDs
        detected_persons_data.sort(key=lambda x: x['distance'])
        detected_cars_data.sort(key=lambda x: x['distance'])

        for i, p_entry in enumerate(detected_persons_data):
            cx, cy, w, h, conf = p_entry['data']
            person_obj = Person(obj_id_num=i + 1, center_x=cx, center_y=cy, width=w, height=h, confidence=conf)
            current_persons.append(person_obj)

        for i, c_entry in enumerate(detected_cars_data):
            cx, cy, w, h, conf = c_entry['data']
            car_obj = Car(obj_id_num=i + 1, center_x=cx, center_y=cy, width=w, height=h, confidence=conf)
            current_cars.append(car_obj)

    # --- Console Printout (Throttled) ---
    frame_processor_counter += 1
    if frame_processor_counter >= PROCESS_FRAME_INTERVAL:
        frame_processor_counter = 0 # Reset counter
        print("\n--- Library: Frame Detections (Console Update) ---")
        found_target_for_print = False

        if current_persons:
            found_target_for_print = True
            print(f"Detected Persons ({len(current_persons)}):")
            for p_obj in current_persons:
                print(p_obj) # Uses the __str__ method
                print("-" * 10)

        if current_cars:
            found_target_for_print = True
            print(f"Detected Cars ({len(current_cars)}):")
            for c_obj in current_cars:
                print(c_obj) # Uses the __str__ method
                print("-" * 10)

        if not found_target_for_print:
            print("No persons or cars detected in this update cycle by the library.")
        print("--- End Library Update ---\n")
    
    return results # Return raw results for potential further use

# Example of listing vocabulary if script is run directly (optional)
if __name__ == "__main__":
    print("Object Detection Library Module")
    if model:
        list_model_vocabulary()
        print("\nThis module is intended to be imported by another script.")
        print("It provides `process_frame_detections` to analyze frames,")
        print("and `current_persons` / `current_cars` lists are populated.")
        print("Use `get_object_by_id('person1')` to retrieve a specific person.")
    else:
        print("Failed to load YOLO model. Library functionality is limited.")
