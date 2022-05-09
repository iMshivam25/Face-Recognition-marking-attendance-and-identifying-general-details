import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

path = 'Images'
# specifying the directory folder
images = []
# Creating an empty List to store all the images
classNames = []
# creating a list, that would contain all the names of person of images[]
myList = os.listdir(path)
# myList would contain list of all names that are in path

# creating a function to do img importing and encoding automatically
for go in myList:
    curImg = cv2.imread(f'{path}/{go}')
    # scrolling through images
    images.append(curImg)
    # appending the current image of the directory to the image list
    classNames.append(os.path.splitext(go)[0])
    # appending the name of that image/person
print(classNames)
# prints all the name of person in the className

# creating a function to encode the image automatically
def findEncodings(images):
    encodedList = []
    # creating an empty list to store the encoded images
    for img in images:
        # scrolling through images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # converting the image to RGB
        encode = face_recognition.face_encodings(img)[0]
        # encoding the image
        encodedList.append(encode)
        # appending to the list
    return encodedList
    # returns the list

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
#         r+, so that we can read and write at the same time
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

_of_img = findEncodings(images)
print(len(_of_img))

cap = cv2.VideoCapture(0)
# initializing the webcam

while True:
    success, img = cap.read()
    # takes a capture
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # resizing/scaling the image. (0,0), None means we aren't defining any pixel size for the captured image
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # converting the image to RGB
    facesCurrFrame = face_recognition.face_locations(imgS)
    # in our webcam frame we may find many faces, so we would recognize each of them using facesCurrFrame
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)
    # encodes each faces in the webcam frame

    # Now, we will find matches.
    # Iterate through all the faces, we've found in the current frame
    # then we will compare all the faces with all encodings
    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame): # one by one it will grab one face loc from faceCurrFrame and encodeFace from encodesCurrFrame
        matches = face_recognition.compare_faces(_of_img, encodeFace)
        faceDis = face_recognition.face_distance(_of_img, encodeFace)
#         zip is used to have them in same loop
#         faceDis will give a list of matches (comparison) to known faces
        print(faceDis)
        matchIndex = np.argmin(faceDis)
#         finding the best match

# since we have matchIndex now we know which person we are talking about

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y2,x2,y1,x1 = faceLoc
    #         taking the face location in terms of coordinates
    #         y2, x2, y1, x1 = y2*4, x2*4, y1*4, x1*4
    #         we have scaled the image 1/4th of size as in line 51 so as to correctly detect
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # creating the rectangle around the detected face
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), 2)
            # rectangular section to display name
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #         displaying the name
            markAttendance(name)
    #         whenever we match a face we put the attendance using markAttendance function

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
