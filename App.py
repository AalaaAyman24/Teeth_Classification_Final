import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


model = load_model(
    r'D:\Cellula Technologies\Week2\teeth__classification__model.h5')

st.title('🦷 Teeth Disease Classifier 🦷')
st.write('Upload an image to classify it.')

Illness = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

uploaded_file = st.file_uploader(
    "Choose an image..", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='The image that was uploaded',
             use_column_width=True)

    image = image.resize((150, 150))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = Illness[predicted_class_index]
    probability = predictions[0][predicted_class_index]

    st.write(f'Prediction: {predicted_class}')
    st.write(f'Probability: {probability:.2f}')


else:
    st.write("Please upload an image to classify!")
