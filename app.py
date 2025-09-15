import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
from collections import Counter
from io import BytesIO

def extract_frames_from_video(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def make_predictions_on_frames(frames, model_path):
    try:
        # Load the YOLO model
        yolo_model = YOLO(model_path)

        # Store predictions and detection images
        all_predictions = []  # To store the class IDs or names
        detection_images = []  # To store the processed images

        for frame in frames:
            # Perform prediction
            results = yolo_model.predict(frame)
            if results:
                # Assume results is list-like with a confidence attribute
                for result in results:
                    # Assuming that result has attributes `boxes` and `cls`
                    for detection in result.boxes:
                        all_predictions.append(detection.cls)

                    # Assuming that `result.plot()` returns an image with annotations
                    detection_images.append(result.plot())

        return all_predictions, detection_images
    except Exception as e:
        st.error(f"Failed to make predictions: {str(e)}")
        return [], []

def run_app():
    MODEL_PATH = "best (2).pt"

    st.title("Pose Detection on Video")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Read video file bytes
        video_bytes = uploaded_file.read()

        # Save and read the video file
        with open('temp_video.mp4', 'wb') as f:
            f.write(video_bytes)

        # Extract frames from the video
        frames = extract_frames_from_video('temp_video.mp4')

        # Get predictions
        with st.spinner("Processing Video..."):
            all_predictions, detection_images = make_predictions_on_frames(frames, MODEL_PATH)

            # Determine class with max votes
            if all_predictions:
                class_counter = Counter(all_predictions)
                most_common_class, count = class_counter.most_common(1)[0]
                st.success(f"Predicted Class: {most_common_class} with {count} detections.")

            # Display some of the processed frames
            for idx, img in enumerate(detection_images[:5]):  # Display first 5 detection images
                st.image(img, caption=f'Detection {idx+1}', use_column_width=True)

if __name__ == "__main__":
    run_app()
