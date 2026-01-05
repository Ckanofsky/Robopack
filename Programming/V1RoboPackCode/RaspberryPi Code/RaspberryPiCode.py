#!/usr/bin/env python3
"""
Raspberry Pi AI Camera Object Detection
Displays camera feed with bounding boxes around detected objects
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
import time

def setup_camera():
    """Initialize the Raspberry Pi AI Camera with IMX500"""
    picam2 = Picamera2(IMX500.get_camera())
    
    # Get intrinsics for the MobileNet SSD model (default on IMX500)
    intrinsics = IMX500.load_network_intrinsics(
        "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    )
    
    # Configure camera with AI processing
    config = picam2.create_preview_configuration(
        controls={"FrameRate": 30},
        buffer_count=12
    )
    
    # Set the neural network
    IMX500.set_network(picam2, intrinsics=intrinsics)
    picam2.configure(config)
    
    return picam2, intrinsics

def draw_detections(frame, detections, labels, threshold=0.5):
    """Draw bounding boxes and labels on detected objects"""
    h, w = frame.shape[:2]
    
    for detection in detections:
        # Extract detection info
        class_id = int(detection[0])
        score = float(detection[1])
        
        # Only draw if confidence is above threshold
        if score < threshold:
            continue
            
        # Get bounding box coordinates (normalized 0-1)
        box = detection[2:6]
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        
        # Get label
        label = labels[class_id] if class_id < len(labels) else f"Class {class_id}"
        
        # Choose color based on class
        color = (0, 255, 0)  # Green for most objects
        if "person" in label.lower():
            color = (255, 0, 0)  # Blue for people
        elif "car" in label.lower() or "vehicle" in label.lower():
            color = (0, 165, 255)  # Orange for vehicles
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = f"{label}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def main():
    """Main function to run object detection"""
    print("Initializing Raspberry Pi AI Camera...")
    
    try:
        picam2, intrinsics = setup_camera()
        
        # Get COCO labels (default for MobileNet SSD)
        labels = intrinsics.labels
        
        # Start camera
        picam2.start()
        print("Camera started. Press 'q' to quit, 's' to save a snapshot.")
        
        fps_start = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            # Capture frame and metadata
            frame = picam2.capture_array()
            metadata = picam2.capture_metadata()
            
            # Get detections from IMX500
            detections = []
            if "imx500_results" in metadata:
                detections = metadata["imx500_results"]
            
            # Convert from RGB to BGR for OpenCV display
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw detections on frame
            if detections:
                frame = draw_detections(frame, detections, labels, threshold=0.5)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start)
                fps_start = time.time()
                fps_counter = 0
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Raspberry Pi AI Camera - Object Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved snapshot: {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'picam2' in locals():
            picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    main()