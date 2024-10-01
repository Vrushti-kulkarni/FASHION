import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

model = tf.keras.models.load_model('trained_model_mnist.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#IMAGE PRE PROCESSING TO ACCEPT ANY INTERVIEW
def process_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L') #convert to gray scale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1,28,28,1)) #ALLOW ONLY ONE IMAGE
    return img_array


#STREAMLIT CODE
st.title('Fashion<3')

uploaded_img = st.file_uploader("Upload image ", type= ['jpg','jpeg','png'])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    #structure of webapp
    col1 , col2 = st.columns(2) #st.columns(number of cols)

    with col1:
        resized_img = image.resize((100,100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            img_array = process_image(uploaded_img)

            result = model.predict(img_array)

            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f"prediction is {prediction}")
