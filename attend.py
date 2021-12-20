import cv2 #importing opencv-python library
import numpy as np #importing numpy for array and encoding operations
import face_recognition #importing face reccognition library for finding , encoding , comparing faces
import os #importing os for fetching local folder data
from datetime import datetime #importing date time for stamping date time
import pickle # for saving/dumping encodings to local storage
import speech_recognition as sr # for speech recognition and v=conversion to text


path='images' #defining folder that contains images of students
imgs=[] #initialising empty image set for image data in matix form
classnames=[] #inititlaising empty student name set for storing names
mylist=os.listdir(path) #List all files in the path folder

for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    classnames.append(os.path.splitext(cl)[0])
    file4 = open("namer.txt" , "r")
    ttt=file4.read()
    v = ttt.find(str(os.path.splitext(cl)[0]))
    if v<0:
        file3 = open("namer.txt", "a")
        file3.write(os.path.splitext(cl)[0] + ",")
        file3.close()
        imgs.append(curimg)

def findencodings(imgs): # defining function that encodes the existing images in the images folder/it actually uses the matrix made by opencv to encode
    encodelist=[] # empty list to store encodings
    for img in imgs: # cyclying through images

        img=cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # changing image to RGB as facerec uses only that and opencv uses BGR
        encode=face_recognition.face_encodings(img)[0] # encoding the image/ encoding is done depending on face feature distance(128 features)
        encodelist.append(encode) # adding encodes to empty sets

    return encodelist # returning all image encodings

print(classnames)
if len(imgs)>0:
    encoder=findencodings(imgs)
    with open('coder.txt', 'wb') as fp:
        pickle.dump(encoder, fp)
    # file1 = open("code.txt", "a")
    # file1.write(str(encoder))
    # file1.close()

# file5 = open("code.txt", "r")
# encodeknown=file5.read()
# file5.close()

with open('coder.txt' , 'rb') as fp:
    encodeknown=pickle.load(fp)

def markattend(name): # defining function to mark attendance in the CSV file
    with open('database.csv' , 'r+') as f: # opening CSV and giving read write privelage of database
        data=f.readlines() #read line from CSV
        namelist=[] # name of attended students
        for line in data: # cycle through dataset
            entry=line.split(',') # split data based on comma
            namelist.append(entry[0]) # add entry[0] as name so that it is written locally in python and checked here first , before pushing into CSV
        if name not in namelist: # if name not already present , then add to CSV , else not
            now=datetime.now() # capture time
            dtstring=now.strftime('%H:%M:%S') # capture time in this format
            f.writelines(f'\n{name},{dtstring}') # write in the CSV file

def voicecheck(enroll):
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Please Speak Out your Enrollment Number")
            audio=r.listen(source)
            print("Conversion underway ....")
            txt=r.recognize_google(audio)
            txt=txt.replace(" ", "")
    except sr.UnknownValueError:
        print("Enrollment Number Not Heard")
        return False

    if txt==str(enroll):
        return True
    else:
        return False

cap=cv2.VideoCapture(0) # start webcam

while True:
    success , img = cap.read() # success is boolean for successfull webcam operation or not , image is input of webcam

    images=cv2.resize(img , (0,0) , None , 0.25 , 0.25) # resize input from the webcam to 1/4th value
    images=cv2.cvtColor(images , cv2.COLOR_BGR2RGB) # changing image to RGB as facerec uses only that and opencv uses BGR

    facecur=face_recognition.face_locations(images) # for live location of face from webcam
    encodecur = face_recognition.face_encodings(images , facecur) # encode current image from webcam , could have used modularity function

    for encodeFace,faceloc in zip(encodecur , facecur): # running one webcam image face through entire preimages preencoded dataset
        matches = face_recognition.compare_faces(encodeknown , encodeFace) # comparing face data through feature distances , returns boolean
        facedis =face_recognition.face_distance(encodeknown , encodeFace) # for portraying accuracy thorugh means of features distance
        matchIndex= np.argmin(facedis) # using numpy to save distance values

        if matches[matchIndex]: # for printing name on live image
            name=classnames[matchIndex].upper() # matches name using classnames available with  index
            print(name) # print name in terminal
            # y1,x2,y2,x1 = faceloc # location of face live
            # y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4 # compensating for 1/4th resizing
            # cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,0,0) , 2) # rectangle around face
            # cv2.rectangle(img, (x1, y2-30), (x2, y2), (255, 255, 255), cv2.FILLED) # lower filled rectangle for ame writing
            # cv2.putText(img , name, (x1+6 , y2-6) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,0) , 2) # writing name

            ii = name.lower().split("_")[1]# get the enrollment number from image
            vc = voicecheck(ii) # voice enrollment number check
            if vc == True:
                markattend(name) # finally , marking attendance in CSV using function
                print("Attendance Marked for : "+str(name))
            else:
                print("Enrollment Number Did Not Match, Try Again")
                break

    # cv2.imshow('Input' , img) # webcam feed shown
    # cv2.waitKey(1) # awaiting keystroke escape