import os
import random
import streamlit as st 
from ultralytics import YOLO
import cv2
import time
import numpy as np
import shutil

image_directory = "val"  # Assuming "val" is the directory name

# Get a list of image filenames in the directory
image_filenames = [filename for filename in os.listdir(image_directory) if filename.endswith(".jpg")]

# Function to generate a random image from the list of filenames
def get_random_image():
    if not image_filenames:
        return None
    random_image_filename = random.choice(image_filenames)
    random_image_filename1=random_image_filename.split('.')[0]
    random_image_path = os.path.join(image_directory, random_image_filename)
    return random_image_path,random_image_filename1 

@st.cache_resource()    
def main_model():
    model=YOLO('best.pt')
    return model

        
def message():
    st.warning('⚠️Please check your image')
    st.info("📷✨ **Encountering the 'Please check your image' error?**")
    st.write("""
            Our algorithm may not have been able to predict the content of your image. To improve results, consider the following:
            👉 **Verify image quality and resolution.**
            👉 **Ensure the image is clear and well-lit.**
            👉 **Check if the image meets our specified format requirements.**
            👉 **Consider alternative images for better results.**
            Our aim is to provide accurate predictions, and addressing these aspects can make a significant difference. If the issue persists, please reach out to our support team. We're here to help! 🤝🔧
            """)

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
 
 
def process_image_with_yolo(pic0):
    

    # Load your YOLO model
    
    if pic0 is not None:
        # Perform YOLO prediction on the image
        
        model = main_model()
        counter = 1
       
        pic0=pic0
        result = model.predict(pic0, save=True, save_txt=True)
        
        txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))

        if txt_files_exist:
            lis = open('runs/detect/predict/labels/image0.txt', 'r').readlines()
            for line in lis:
                process_line(line, pic0, counter)
                counter += 1
          
            
                
            if pic0 is not None:
                st.image(pic0, use_column_width=True)
                st.balloons()
            try:
                if os.path.exists('runs'):
                    shutil.rmtree('runs')
                    st.session_state.original_image = None  # Clear the original_image variable
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            message()
            try:
                if os.path.exists('runs'):
                    shutil.rmtree('runs')
                    st.session_state.original_image = None  # Clear the original_image variable
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
def process_image_with_yolo_pic1(pic1,image_np1,image_name):
    if pic1 is not None and image_np1 is not None and len(image_name)!=0:
        image_name=image_name
        model = main_model()
        counter = 1
      
        pic1=pic1
        image_np1=image_np1
        
        result = model.predict(pic1, save=True, save_txt=True)

        txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))
        main_str='runs/detect/predict/labels/'+image_name+'.txt'
        if txt_files_exist:
            lis = open(main_str, 'r').readlines()
            for line in lis:
                process_line(line, image_np1, counter)
                counter += 1
          
            col1,col2=st.columns(2)
            with col1:
                     
                st.info('Original Image!')
               
                st.image(pic1,use_column_width=True)
            with col2:
                
                    
                st.info('Detected Image!')
               
                st.image(image_np1,use_column_width=True)
            st.balloons()
            try:
                if os.path.exists('runs'):
                    shutil.rmtree('runs')
                    st.session_state.original_image = None  # Clear the original_image variable
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
                    
           
        else:
            message()
            try:
                if os.path.exists('runs'):
                    shutil.rmtree('runs')
                    st.session_state.original_image = None  # Clear the original_image variable
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
def process_image_with_yolo_pic2(pic2,image_np2):
    if pic2 is not None and image_np2 is not None:
        model=main_model()
        counter = 1
       
        pic2=pic2
        image_np2=image_np2
        result = model.predict(image_np2, save=True, save_txt=True)

        txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))

        if txt_files_exist:
            
            lis = open('runs/detect/predict/labels/image0.txt', 'r').readlines()
            for line in lis:
                process_line(line, image_np2, counter)
                counter += 1
          
            
            col1,col2=st.columns(2)
            with col1:
                
                    
                                    
                st.info('Original Image!')
          
                st.image(pic2,use_column_width=True)
            with col2:
                
                st.info('Detected Image!')
            
                st.image(image_np2,use_column_width=True)
            st.balloons()
            
            try:
                if os.path.exists('runs'):
                    shutil.rmtree('runs')
                    st.session_state.original_image = None  # Clear the original_image variable
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
              
        else:
            
            message()
            try:
                if os.path.exists('runs'):
                    shutil.rmtree('runs')
                    st.session_state.original_image = None  # Clear the original_image variable
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
            

            
