import numpy as np
import pandas as pd
from ultralytics import YOLO
import streamlit as st
import cv2
import base64
import time
import shutil
import os
from PIL import Image
import base64
import random
from utils import main_model,get_random_image,message,upload,process_line,process_image_with_yolo,process_image_with_yolo_pic1,process_image_with_yolo_pic2

st.set_page_config(layout="wide",initial_sidebar_state="expanded",
                   page_icon='🔎',page_title='Poth-Hole Detector')

# Define custom style for the glowing text
glowing_text_style = '''
    <style>
        .glowing-text {
            font-family: 'Arial Black', sans-serif;
            font-size: 48px;
            text-align: center;
            animation: glowing 2s infinite;
        }
        
        @keyframes glowing {
            0% { color: #FF9933; } /* Saffron color */
            25% { color: #FFFFFF; } /* White color */
            50% { color: #128807; } /* Green color */
            
        }
    </style>
'''

# Display the glowing text using st.markdown
st.markdown(glowing_text_style, unsafe_allow_html=True)
st.markdown(f'<p class="glowing-text">🕳️ PothHole Detector 🕳️</p>', unsafe_allow_html=True)



       

 
sidebar_option = st.sidebar.radio("Select an option", ("Take picture for prediction", "Upload file"))

def main():
    
    
    
    
   
    if sidebar_option == "Take picture for prediction":
        if st.checkbox('Take a picture for prediction'):
    
        
            image, original_image,image_filename= upload()
            if original_image is not None and image_filename is not None and len(image_filename)!=0 and st.button('Prediction'):  # Check if original_image is not None
                st.info('Wait for the results...!')
                #image1=cv2.imread(image)
                pic0=image
                process_image_with_yolo(pic0)

    elif sidebar_option == "Upload file":  
          
        
        fileimage=st.file_uploader('Upload the file for detection 📁',type=['jpg','jpeg','png'])
        st.info("If you haven't filed, our system will employ a default image for prediction 📁. Simply press the 'Predict' button and directly upload your file for analysis 🧐.")
        
        
        if st.button('Predict'):

                    
            if True:
                    
                if fileimage is None:
                    default_image,image_name=get_random_image()
                    st.warning('⚠️ We are using random image from our backend!.')
                    
                    
                    pic1=Image.open(default_image)
                    image_np1 = np.array(pic1)
                    if pic1 is not None and image_np1 is not None:
                        
                        process_image_with_yolo_pic1(pic1,image_np1,image_name)
                    
                else:
                    st.info('Wait for the results...!')
                    
                    
                    pic2=Image.open(fileimage)
                    image_np2 = np.array(pic2)
                    if pic2 is not None and image_np2 is not None:
                        process_image_with_yolo_pic2(fileimage,image_np2)
                   
                    

        
        
            

   
if __name__ == '__main__':
    
   
    main()
