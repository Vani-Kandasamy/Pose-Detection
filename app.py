
import streamlit as st
from PIL import Image
from ultralytics import YOLO

def make_predictions(image_path, model_path):
    try:
        # Load the YOLO model
        yolo_model = YOLO(model_path)

        # Perform prediction
        results = yolo_model.predict(image_path)

        # Extract the first detection result for this image
        if results:
            # The results object should have a method or property to output the plotted image
            # Check if 'plot' or similar is provided to illustrate detections
            for result in results:
                detection_image = result.plot()  # Adjust if function calls are different
                return detection_image
        else:
            return None
    except Exception as e:
        st.error(f"Failed to make predictions: {str(e)}")
        return None

def run_app():
    # Constants
    IMAGE_NAME = "uploaded.png"
    MODEL_PATH = "best.pt"
    IMAGE_ADDRESS = "https://thinkpalm.com/wp-content/uploads/2023/01/image_03.jpg"

    # UI
    st.title("Pose Prediction")
    st.image(IMAGE_ADDRESS, caption="Pose Prediction Example")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and save the uploaded file
        image = Image.open(uploaded_file)
        image.save(IMAGE_NAME)
        #st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Get predictions with a spinner
        with st.spinner("Getting Predictions..."):
            detection_image = make_predictions(IMAGE_NAME, MODEL_PATH)

            # Display original image and detection results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Prediction Result")
                if detection_image is not None:
                    st.image(detection_image, caption='Predicted Pose', use_container_width=True)
                else:
                    st.error("No prediction results.", icon="🚨")

if __name__ == "__main__":
    run_app()
