import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Class name mapping
CLASS_NAMES = [
    'squat', 'bowl', 'pushup', 'pullup', 'tennis_serve', 
    'baseball_swing', 'baseball_pitch', 'golf_swing', 
    'tennis_forehand', 'bench_press', 'jumping_jacks', 
    'situp', 'strum_guitar', 'clean_and_jerk', 'jump_rope'
]

def process_video(video_path, model_path):
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Track the best detection
        best_confidence = 0
        best_frame = None
        best_class_idx = None
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO prediction
            results = model(frame_rgb)
            
            # Process detections
            for result in results:
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Update best detection
                    if conf > best_confidence:
                        best_confidence = conf
                        best_frame = result.plot()
                        best_class_idx = cls_idx
        
        # Get class name
        best_class_name = CLASS_NAMES[best_class_idx] if best_class_idx is not None and best_class_idx < len(CLASS_NAMES) else "unknown"
        
        # Release video capture
        cap.release()
        
        return best_frame, best_class_name, best_confidence
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None, 0.0

def main():
    st.title("YOLO Pose Detection on Video")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            
        try:
            with st.spinner("Processing video..."):
                # Process the video
                best_frame, best_class, confidence = process_video(video_path, "best.pt")
                
                # Display results
                if best_frame is not None:
                    st.success(f"Best Detection - Class: {best_class.capitalize()}, Confidence: {confidence*100:.2f}%")
                    st.image(best_frame, use_container_width=True)
                else:
                    st.warning("No detections found in the video.")
                    
        finally:
            # Clean up the temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)

if __name__ == "__main__":
    main()
