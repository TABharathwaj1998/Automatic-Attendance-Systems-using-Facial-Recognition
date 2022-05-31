import os
from PIL import Image
import numpy as np
import cv2
import pickle
import sys
import face_recognition
from PIL import Image
import pandas as pd
import threading

base_dir = os.getcwd()
image_dir = os.path.join(base_dir, "images")

#'''Import Csv file named AIDI1003, set Names as index and collect the number of rows present'''
aidi = pd.read_csv(base_dir+"/"+"AIDI1003.csv")
student=aidi.set_index("Names",drop=False)
studentsList=len(aidi)

#'''Initialize the variables before functioning'''
list=0
studentNames=[]
for initialise in range(studentsList):
    studentNames.append(aidi["Names"].iloc[initialise])
    student.loc[studentNames[initialise],"Status"]="Absent"

def handler():
  print("Attendance is complete")
  os._exit(1)

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

import webbrowser
webbrowser.open('file.html')

@app.route('/my-link/')
def my_link():
    indices=0
    flag=0
    facecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i=0
    

    alarm = threading.Timer(65, handler)
    alarm.start()
    
    
    
    while True:
        video_capture = cv2.VideoCapture(0)
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=4)
        

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display the resulting frame
            cv2.imshow('Video', frame)     
            while indices < studentsList:
                known_image = face_recognition.load_image_file(image_dir+"/" + studentNames[indices] + ".jpeg")
                new_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(frame)[0]
                matches = face_recognition.compare_faces([new_encoding], unknown_encoding)

                if True in matches:
                    student.loc[studentNames[indices],"Status"]="Present"
                    
                    indices=0
                    break
                else:
                    indices+=1
        student.to_csv(base_dir+"/"+"AIDI1003.csv",index=False)

        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    alarm.cancel()
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
        
print ('I got clicked!')

if __name__ == '__main__':
  app.run(debug=True)


