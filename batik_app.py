import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (80,80)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('my_model.hdf5')

st.write("""
         # Deteksi Model Batik
         """
         )

st.write("Untuk mendeteksi model batik berdasarkan gambar")

file = st.file_uploader("Silahkan upload gambar", type=["jpg", "png"])

if file is None:
    st.text("Belum ada gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    #print("prediction : ",prediction)
    if np.argmax(prediction) == 0:
        st.write("Hasil : Batik Geblek Renteng")
    elif np.argmax(prediction) == 1:
        st.write("Hasil : Batik Kawung")
    elif np.argmax(prediction) == 2:
        st.write("Hasil : Batik Lereng")
    elif np.argmax(prediction) == 3:
        st.write("Hasil : Batik Nitik")
    elif np.argmax(prediction) == 4:
        st.write("Hasil : Batik Parang")
    #st.write(np.argmax(prediction))
    #st.text("Probability (0: Geblek Renteng, 1: Kawung, 2: Lereng, 3: Nitik, 4: Parang)")
    #st.write(prediction)
