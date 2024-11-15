import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model('trained_model/face_mask_detector.h5')  # Update the model path

def predict_image(image, model):
    # Resize and normalize the image
    image = image.resize((128, 128))  # Resize the image to match the model's expected input
    image_array = np.array(image) / 255.0  # Convert the image to numpy array and normalize it
    
    # Reshape for the model input
    image_reshaped = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_reshaped)
    return predictions

# Set up Streamlit interface
st.title('Face Mask Detection')
st.write("Upload an image and the model will predict whether a face mask is present.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # Open and convert the image to RGB
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict_image(image, model)
    prediction_label = np.argmax(prediction)
    
    if prediction_label == 1:
        result = 'The person in the image is wearing a mask.'
    else:
        result = 'The person in the image is not wearing a mask.'
        
    st.write(result)

if __name__ == '__main__':
    st.run()
