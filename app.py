import io

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import tempfile
import os


model = YOLO("best.pt")

st.title("Furniture Detection App")
st.write("Uplpoad an image to detect furniture!")

upload_file = st.file_uploader("Choose an image... ", type=["jpg", "jpeg", "png"])
print(upload_file)

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.write("Processing Image...")

    # Save the uploaded image temporarily to a file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        image.save(temp_file_path)

    # Run inference on the temporary image file
    result = model.predict(source=temp_file_path, save=False, conf=0.5)

    # Plot the result
    result_img = result[0].plot()

    # Display the detected furniture
    st.image(result_img, caption="Detected Furniture", use_container_width=True)

    # Clean up the temporary file after processing
    os.remove(temp_file_path)