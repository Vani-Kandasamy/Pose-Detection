import streamlit as st

try:
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

from ultralytics import YOLO
import tempfile
import os
import numpy as np

# Class name mapping
CLASS_NAMES = [
    'squat', 'bowl', 'pushup', 'pullup', 'tennis_serve', 
    'baseball_swing', 'baseball_pitch', 'golf_swing', 
    'tennis_forehand', 'bench_press', 'jumping_jacks', 
    'situp', 'strum_guitar', 'clean_and_jerk', 'jump_rope'
]

def process_video_with_annotations(video_path, model_path, output_path):
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        best_confidence = 0
        best_class_idx = None
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            
            for result in results:
                annotated_frame = result.plot()
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > best_confidence:
                        best_confidence = conf
                        best_class_idx = cls_idx
                
                # Display the current frame
                if frame_count % 5 == 0:
                    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        best_class_name = CLASS_NAMES[best_class_idx] if best_class_idx is not None and best_class_idx < len(CLASS_NAMES) else "unknown"
        
        return output_path, best_class_name, best_confidence
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None, 0.0

def run_app():
    st.title("YOLO Pose Detection on Video")
    IMAGE_ADDRESS = "https://thinkpalm.com/wp-content/uploads/2023/01/image_03.jpg"

    # UI
    st.title("Pose Prediction")
    st.image(IMAGE_ADDRESS)
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        output_path = "output_annotated.mp4"
        
        try:
            with st.spinner("Processing video..."):
                output_video, best_class, confidence = process_video_with_annotations(
                    video_path, "best.pt", output_path
                )
                
                if output_video is not None and os.path.exists(output_path):
                    st.success("Video processing complete!")
                    
                    # Read the output video file
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    # Display best class and confidence
                    st.subheader("Detection Results")
                    st.write(f"Best Class: {best_class.capitalize()}")
                    st.write(f"Confidence: {confidence*100:.2f}%")
                    
                    # Download button
                    st.download_button(
                        label="Download Annotated Video",
                        data=video_bytes,
                        file_name="annotated_video.mp4",
                        mime="video/mp4"
                    )
                    
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

run_app()
