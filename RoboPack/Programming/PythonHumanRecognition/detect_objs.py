# Ensure you have the required packages installed:
# pip install opencv-python ultralytics

import cv2
from ultralytics import YOLO
import numpy as np # Import numpy for array operations

# Load the YOLOv8 model
# "yolov8n.pt" is a small and fast model. You can use other versions like "yolov8s.pt", "yolov8m.pt", etc.
model = YOLO("yolov8n.pt")

# --- Configuration for print update frequency ---
PROCESS_FRAME_INTERVAL = 15  # Print detection details every N frames
frame_processor_counter = 0  # Counter to track frames for printing

def list_vocab():
    """
    Prints the list of class names that the loaded YOLO model can detect.
    """
    print("YOLO Model Vocabulary (Class ID: Class Name):")
    class_names = model.names
    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")
    print("-" * 30)

def handle_detection_data(results):
    """
    Processes detection results to identify humans and cars,
    and prints their bounding box center coordinates and area at a controlled frequency.

    Args:
        results: The output from the YOLO model after processing a frame.
    """
    global frame_processor_counter
    frame_processor_counter += 1

    if frame_processor_counter < PROCESS_FRAME_INTERVAL:
        return # Skip printing for this frame

    # Reset counter after reaching the interval
    frame_processor_counter = 0
    
    boxes = results[0].boxes 
    class_ids = boxes.cls.cpu().numpy().astype(int) 
    bboxes_xywh = boxes.xywh.cpu().numpy() 
    names = model.names

    print("\n--- Frame Detections (Console Update) ---")
    found_target_for_print = False
    for i in range(len(class_ids)):
        class_id = class_ids[i]
        class_name = names[class_id]

        # Check if the detected object is a 'person' or 'car'
        # Note: YOLO COCO dataset calls humans 'person'
        if class_name.lower() == 'person' or class_name.lower() == 'car':
            found_target_for_print = True
            center_x, center_y, width, height = bboxes_xywh[i]
            area = width * height
            
            print(f"Detected: {class_name}")
            print(f"  Center (x, y): ({center_x:.2f}, {center_y:.2f})")
            print(f"  Dimensions (width, height): ({width:.2f}, {height:.2f})")
            print(f"  Area: {area:.2f} pixels^2")
            print("-" * 20)
            
    if not found_target_for_print:
        print("No humans or cars detected in this update cycle.")
    print("--- End Frame Detections (Console Update) ---\n")


def main():
    """
    Main function to initialize webcam, run object detection, 
    and display results focusing only on humans and cars.
    """
    global frame_processor_counter # Ensure main can access the global counter if needed

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Starting webcam feed for object detection...")
    print("Displaying ONLY humans and cars. Console updates are throttled.")
    print(f"Console position updates will occur every {PROCESS_FRAME_INTERVAL} frames.")
    print("Press 'q' to quit.")

    # Get class names from the model
    model_class_names = model.names
    # Get the class IDs for 'person' and 'car' for efficient lookup
    # This assumes 'person' and 'car' are in model_class_names. Add error handling if needed.
    try:
        person_class_id = [k for k, v in model_class_names.items() if v.lower() == 'person'][0]
        car_class_id = [k for k, v in model_class_names.items() if v.lower() == 'car'][0]
        target_class_ids = {person_class_id, car_class_id}
    except IndexError:
        print("Error: 'person' or 'car' class not found in model vocabulary. Check list_vocab().")
        cap.release()
        cv2.destroyAllWindows()
        return


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Run object detection on the current frame
        results = model(frame, verbose=False) 

        # Process detection data for console printouts (throttled)
        handle_detection_data(results)

        # --- Manual drawing of bounding boxes for ONLY humans and cars ---
        # Make a copy of the frame to draw on, or draw directly on 'frame'
        annotated_frame = frame.copy() 
        
        if results[0].boxes is not None:
            detected_boxes = results[0].boxes
            # Get bounding boxes in (x1, y1, x2, y2) format for drawing
            bboxes_xyxy = detected_boxes.xyxy.cpu().numpy()
            # Get class IDs
            class_ids = detected_boxes.cls.cpu().numpy().astype(int)
            # Get confidence scores
            confidences = detected_boxes.conf.cpu().numpy()

            for i in range(len(class_ids)):
                current_class_id = class_ids[i]
                
                # Check if the detected object is a person or a car
                if current_class_id in target_class_ids:
                    x1, y1, x2, y2 = map(int, bboxes_xyxy[i]) # Convert to int for drawing
                    confidence = confidences[i]
                    label = f"{model_class_names[current_class_id]}: {confidence:.2f}"
                    
                    # Define color based on class (optional, e.g., green for person, blue for car)
                    color = (0, 255, 0) # Default Green
                    if model_class_names[current_class_id].lower() == 'car':
                        color = (255, 0, 0) # Blue for cars
                    elif model_class_names[current_class_id].lower() == 'person':
                        color = (0, 255, 0) # Green for persons

                    # Draw the bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Put the label above the bounding box
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the frame
        cv2.imshow('Object Detection (Humans and Cars Only)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

if __name__ == "__main__":
    list_vocab()
    main()
