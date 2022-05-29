from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from PIL import Image
from pathlib import Path

app = Flask(__name__)


# generating dataset
def generate_dataset(username):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # starting the webcam
    vc = cv2.VideoCapture(0)
    userName = username
    count = 1

    # To save the images in the dataset folder
    def saveImage(image, userName, imgId):
        Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite("dataset/{}/{}_{}.jpg".format(userName,
                    userName, imgId), image)

    print("Video Camera Starts")

    coords = []
    while True:
        # Capturing the image
        _, img = vc.read()
        originalImg = img.copy()

        # Converting it to grey color
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Coordinate Locations
        faces = faceCascade.detectMultiScale(gray_img,
                                             scaleFactor=1.2,
                                             minNeighbors=5,
                                             minSize=(50, 50))

        for (x, y, w, h) in faces:
            coords = [x, y, w, h]

        while count <= 20:
            captured_image = originalImg[coords[1]:coords[1] +
                                  coords[3], coords[0]: coords[0] + coords[2]]
            saveImage(captured_image, userName, count)
            count += 1
        break
    
    # Stopping video camera
    vc.release()
    print("Ended")


def train():
    names = []
    path = []

    # Get the names of all the users
    for users in os.listdir("dataset"):
        names.append(users)

    # Get the path to all the images
    for name in names:
        for image in os.listdir("dataset/{}".format(name)):
            path_string = os.path.join("dataset/{}".format(name), image)
            path.append(path_string)

    faces = []
    ids = []

    # For each image create a numpy array and add it to faces list
    for img_path in path:
        image = Image.open(img_path).convert("L")
        imgNp = np.array(image, "uint8")
        name = img_path.split("/")[1].split("\\")[1].split("_")[0]

        faces.append(imgNp)

        ids = np.array(ids)
        ids = np.append(ids, name)
    print("ids formed")
    
    # Using label encoder
    le = preprocessing.LabelEncoder()
    le.fit(ids)
    list(le.classes_)
    le.transform(ids)
    
    
    # Calling recognizer
    trainer = cv2.face.LBPHFaceRecognizer_create()   
    trainer.train(faces, le.transform(ids))
    trainer.write("training.yml")

    print("[INFO] Training Done")



@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def option():
    if request.method == 'POST':
        if request.form.get('a1') == 'Take Images':
            return redirect(url_for('generate_ds'))
        elif request.form.get('a2') == 'Recognize Face':
            return redirect(url_for('recog'))
        elif request.form.get('a3') == 'CSV/Excel Sheet':
            return redirect(url_for('download_csv'))


@app.route('/generate')
def generate_ds():
    return render_template("namesubmit.html")


@app.route('/generate', methods=['POST'])
def my_form_post():
    if(request.method == 'POST'):
        if(request.form.get('a4') == 'SUBMIT'):
            submission = request.form['name']
            generate_dataset(submission)
            train()
            return redirect(url_for('facedata'))


@app.route('/facedatarcd')
def facedata():

    return render_template("facedatarecorded.html")


@app.route('/recognize')
def recog():
    # p = detection()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("training.yml")

    path = []
    names = []
    ids = []
   
    for users in os.listdir("dataset"):
        names.append(users)

    # Getting path of all images
    for name in names:
        for image in os.listdir("dataset/{}".format(name)):
            path_string = os.path.join("dataset/{}".format(name), image)
            path.append(path_string)
    for img_path in path:
        image = Image.open(img_path).convert("L")
        name = img_path.split("/")[1].split("\\")[1].split("_")[0]

        ids = np.array(ids)
        ids = np.append(ids, name)

    le = preprocessing.LabelEncoder()
    le.fit(ids)
  
    names = []
    for users in os.listdir("dataset"):
        names.append(users)

    print(names)

    # to mark attendance in excel sheet
    def markAttendance(name):
        s = ''
        with open('attendance.csv', 'r+') as f:
            attendanceList = f.readlines()
            names = []
            for line in attendanceList:
                entry = line.split(',')
                names.append(entry[0])

            if name not in names:
                print("test4")
                now = datetime.now()
                dateString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dateString}')
                print("Attendance Marked")
                s = "Attendance Marked"
            else:
                print("test5")
                s = "Attendance Already Marked"
        return s
    variable = ''
    while True:

        _, img = video_capture.read()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, _ = recognizer.predict(gray_image[y: y + h, x: x + w])

            if id:
                variable = markAttendance(le.inverse_transform([id])[0])

            else:
                variable = markAttendance("unknown")

        video_capture.release()
        break
    return render_template("attendance.html", var=variable)


@app.route('/download')
def download_csv():
    return render_template("download.html")


@app.route('/download', methods=['POST'])
def download_file():
    if(request.method == 'POST'):
        path = "attendance.csv"
        return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(port=8000)
