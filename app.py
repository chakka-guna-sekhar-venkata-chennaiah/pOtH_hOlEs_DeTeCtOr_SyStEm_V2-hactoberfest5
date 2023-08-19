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


st.set_page_config(layout="wide",initial_sidebar_state="expanded",
                   page_icon='🔎',page_title='Poth-Hole Detector')

image_directory = "val"  # Assuming "val" is the directory name

# Get a list of image filenames in the directory
image_filenames = [filename for filename in os.listdir(image_directory) if filename.endswith(".jpg")]

# Function to generate a random image from the list of filenames
def get_random_image():
    if not image_filenames:
        return None
    random_image_filename = random.choice(image_filenames)
    random_image_path = os.path.join(image_directory, random_image_filename)
    return random_image_path      

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
            75% { color: #0000FF; } /* Blue color */
            100% { color: #FF9933; } /* Saffron color */
        }
    </style>
'''

# Display the glowing text using st.markdown
st.markdown(glowing_text_style, unsafe_allow_html=True)
st.markdown(f'<p class="glowing-text">🕳️ PothHole Detector 🕳️</p>', unsafe_allow_html=True)

def upload():
    image=None
    image_filename=None
    initial_image = st.camera_input('Take a picture')
    original_image = initial_image
    temp_path = None
    if initial_image is not None:
        image_filename = f"{int(time.time())}.jpg"
        bytes_data = initial_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    return image, original_image,image_filename

def process_line(line, image_np,counter):
    # Process a single line from the labels.txt file
    bresults = line.split()
    if len(bresults) >=5:
        names={0:'POTH_HOLE'}
        xc, yc, nw, nh = map(float, bresults[1:5])
        h, w = image_np.shape[0], image_np.shape[1]

        xc *= w
        yc *= h
        nw *= w
        nh *= h
        top_left = (int(xc - nw / 2), int(yc - nh / 2))
        bottom_right = (int(xc + nw / 2), int(yc + nh / 2))

        # Draw bounding box
        cv2.rectangle(image_np, top_left, bottom_right, (4, 29, 255), 3, cv2.LINE_4)

        # Draw label text
        #label = names[int(bresults[0])]
        label = f'{names[int(bresults[0])]}-{counter}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_width, text_height = text_size
        text_x = (top_left[0] + bottom_right[0] - text_width) // 2 + 100
        text_y = top_left[1] - 10
        cv2.putText(image_np, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        

 
sidebar_option = st.sidebar.radio("Select an option", ("Take picture for prediction", "Upload file"))

def main():
    
    
    
    
   
    if sidebar_option == "Take picture for prediction":
        if st.checkbox('Take a picture for prediction'):
    
        
            image, original_image,image_filename= upload()
            if original_image is not None and original_image is not None and len(image_filename)!=0 and st.button('Prediction'):  # Check if original_image is not None
                st.info('Wait for the results...!')
                #image1=cv2.imread(image)
                
                counter=1
                names={0:'POTH_HOLE'
                    }
               
                model=YOLO('best.pt')
                result = model.predict(image,save=True,save_txt=True)
                txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))
                if txt_files_exist:
                    lis=open('runs/detect/predict/labels/image0.txt','r').readlines()
                    for line in lis:
                        process_line(line, image,counter)
                        counter+=1
                    with st.spinner('Wait for the results...!'):
                        time.sleep(5)
                    
                    st.image(image,use_column_width=True)
                    st.balloons()
                    
                    try:
                        if os.path.exists('runs'):
                            shutil.rmtree('runs')
                            st.session_state.original_image = None  # Clear the original_image variable
                           
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning('⚠️Please check your image')
                    st.info("📷✨ **Encountering the 'Please check your image' error?**")
                    st.write(
                            """
                            Our algorithm may not have been able to predict the content of your image. To improve results, consider the following:
                            👉 **Verify image quality and resolution.**
                            👉 **Ensure the image is clear and well-lit.**
                            👉 **Check if the image meets our specified format requirements.**
                            👉 **Consider alternative images for better results.**
                            Our aim is to provide accurate predictions, and addressing these aspects can make a significant difference. If the issue persists, please reach out to our support team. We're here to help! 🤝🔧
                            """
                        )
                    try:
                        if os.path.exists('runs'):
                            shutil.rmtree('runs')
                            st.session_state.original_image = None  # Clear the original_image variable
                           
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    elif sidebar_option == "Upload file":  
          
        
        fileimage=st.file_uploader('Upload the file for detection 📁',type=['jpg','jpeg','png'])
        st.info("If you haven't filed, our system will employ a default image for prediction 📁. Simply press the 'Predict' button and directly upload your file for analysis 🧐.")
        
        
        if st.button('Predict'):

                    
            if True:
                    
                if fileimage is None:
                    default_image=get_random_image()
                    st.warning('⚠️ We are using random image from our backend!.')
                    st.info('Wait for the results...!')
                    counter=1
                    pic=Image.open(default_image)
                    image_np = np.array(pic)
                    names={0:'POTH_HOLE'
                        }
                    mod1=YOLO('best.pt')
                    mod1.predict(image_np,save=True,save_txt=True)
                    txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))
                    if txt_files_exist:
                        lis=open('runs/detect/predict/labels/image0.txt','r').readlines()
                        with st.spinner('Wait for the results...!'):
                            time.sleep(5)
    
                                    
                        for line in lis:
                            process_line(line, image_np,counter)
                            counter+=1
                        col1,col2=st.columns(2)
                        with col1:
                                
                            st.info('Original Image!')
                            st.image(default_image,use_column_width=True)
                        with col2:
                            st.info('Detected Image!')
                            st.image(image_np,use_column_width=True)
                        st.balloons()
                        
                        try:
                            if os.path.exists('runs'):
                                shutil.rmtree('runs')
                                st.session_state.original_image = None  # Clear the original_image variable
                                
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    else:
                        st.warning('⚠️Please check your image')
                        st.info("📷✨ **Encountering the 'Please check your image' error?**")
                        st.write(
                            """
                            Our algorithm may not have been able to predict the content of your image. To improve results, consider the following:
                            👉 **Verify image quality and resolution.**
                            👉 **Ensure the image is clear and well-lit.**
                            👉 **Check if the image meets our specified format requirements.**
                            👉 **Consider alternative images for better results.**
                            Our aim is to provide accurate predictions, and addressing these aspects can make a significant difference. If the issue persists, please reach out to our support team. We're here to help! 🤝🔧
                            """
                        )
                        try:
                            if os.path.exists('runs'):
                                shutil.rmtree('runs')
                                st.session_state.original_image = None  # Clear the original_image variable
                            
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                else:
                    st.info('Wait for the results...!')
                    counter=1
                    pic=Image.open(fileimage)
                    image_np = np.array(pic)
                    names={0:'POTH_HOLE'
                        }
                    mod1=YOLO('best.pt')
                    mod1.predict(image_np,save=True,save_txt=True)
                    txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))
                    if txt_files_exist:
                        lis=open('runs/detect/predict/labels/image0.txt','r').readlines()
                        with st.spinner('Wait for the results...!'):
                            time.sleep(5)
    
                                    
                        for line in lis:
                            process_line(line, image_np,counter)
                            counter+=1
                        col1,col2=st.columns(2)
                        with col1:
                                
                            st.info('Original Image!')
                            st.image(fileimage,use_column_width=True)
                        with col2:
                            st.info('Detected Image!')
                            st.image(image_np,use_column_width=True)
                        st.balloons()
                        
                        try:
                            if os.path.exists('runs'):
                                shutil.rmtree('runs')
                                st.session_state.original_image = None  # Clear the original_image variable
                                
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    else:
                        st.warning('⚠️Please check your image')
                        st.info("📷✨ **Encountering the 'Please check your image' error?**")
                        st.write(
                            """
                            Our algorithm may not have been able to predict the content of your image. To improve results, consider the following:
                            👉 **Verify image quality and resolution.**
                            👉 **Ensure the image is clear and well-lit.**
                            👉 **Check if the image meets our specified format requirements.**
                            👉 **Consider alternative images for better results.**
                            Our aim is to provide accurate predictions, and addressing these aspects can make a significant difference. If the issue persists, please reach out to our support team. We're here to help! 🤝🔧
                            """
                        )
                        try:
                            if os.path.exists('runs'):
                                shutil.rmtree('runs')
                                st.session_state.original_image = None  # Clear the original_image variable
                            
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    

        
        
            

   
if __name__ == '__main__':
    
   
    main()