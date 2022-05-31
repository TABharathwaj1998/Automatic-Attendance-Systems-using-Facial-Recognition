#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import face_recognition
from PIL import Image
import streamlit as st
import pandas as pd
import smtplib, ssl

st.image('image.png')

AIDI1003Students = pd.read_csv("C:/Users/BHARATHWAJ T A/Pictures/AIDI1003.csv")
student=AIDI1003Students.set_index("Names",drop=False)
studentsList=len(AIDI1003Students)

list=0
studentNames=[]
for initialise in range(studentsList):
    studentNames.append(AIDI1003Students["Names"].iloc[initialise])
    student.loc[studentNames[initialise],"Status"]="Absent"


# In[2]:


def faceCompare():    
    indices=0
    flag=0
    cascPath = "C:/Users/BHARATHWAJ T A/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    while True:
        video_capture = cv2.VideoCapture(0)
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=4)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if ((cv2.waitKey(1) & 0xFF == ord('p')) and (flag<studentsList)):
            while indices < studentsList:
                known_image = face_recognition.load_image_file("C:/Users/BHARATHWAJ T A/Pictures/"+studentNames[indices]+".jpeg")
                biden_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(frame)[0]
                matches = face_recognition.compare_faces([biden_encoding], unknown_encoding)

                if True in matches:
                    student.loc[studentNames[indices],"Status"]="Present"
                    flag+=1
                    student.to_csv("C:/Users/BHARATHWAJ T A/Pictures/AIDI1003.csv",index=False)
                    indices=0
                    break
                else:
                    indices+=1
        elif ((cv2.waitKey(1) & 0xFF == ord('a')) or (flag == studentsList)):
            student.to_csv("C:/Users/BHARATHWAJ T A/Pictures/AIDI1003.csv",index=False)
            # When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()
            break


# In[3]:


st.title("Automatic Attendance system")
st.header("AIDI1003")

st.warning(
    """
    ✏️ **NOTE:** Please ensure everyone in the class are made aware that they are being monitored through a video camera
"""
)

if st.button('Scan'):
    faceCompare()
    port = 465  
    smtp_server = "smtp.gmail.com"
    sender_email = "argusteam0921@gmail.com"
    password = "$argusteam_0921"
    message1 = """        Subject: Hi there


    You have been marked Present for your attendance in the class today."""

    message2 = """        Subject: Hi there


    You have been marked Absent for missing the class today. Please contact me to discuss about your attendance and receive lecture material"""
    context = ssl.create_default_context()
    
    currentStatus=pd.read_csv("C:/Users/BHARATHWAJ T A/Pictures/AIDI1003.csv")
    
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        for index in range(studentsList):
            if currentStatus["Status"].iloc[index]=="Present":
                server.sendmail(sender_email, currentStatus["Email ID"].iloc[index], message1)
            elif currentStatus["Status"].iloc[index]=="Absent":
                server.sendmail(sender_email, currentStatus["Email ID"].iloc[index], message2)
            else:
                pass

if st.button('Send Reports'):
    st.write('Reports sent')
    
     
        
st.write('If you would like to receive a report of attendance, enter your email address below and hit receive button')
title = st.text_input('Enter your email address here:')

if st.button('receive'):
    import smtplib, ssl
    port = 465  
    smtp_server = "smtp.gmail.com"
    sender_email = "argusteam0921@gmail.com"
    receiver_id = title
    password = "$argusteam_0921"
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    aidi = pd.read_csv("C:/Users/BHARATHWAJ T A/Pictures/AIDI1003.csv")
    stud=aidi.set_index("Names",drop=False)
    html = stud.to_html()
    msg = MIMEMultipart()
    msg['Subject'] = "Attendance Report"
    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_id, msg.as_string())
    
    

st.write('The project is available for review on our [** git repository**](https://github.com/AlanLozV/attendance-ai-system)!')





