# Face-Recognition-System
This project is a real-time Face Recognition System built using Python, OpenCV, and NumPy. It captures facial images via webcam, stores them as NumPy arrays, and recognizes faces using a custom K-Nearest Neighbors (KNN) algorithm. Haar Cascade is used for detection, and Euclidean distance enables accurate classification through majority voting.

import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)

# Ask the name
filename = input("Enter the name of the person: ")

# Correct dataset path
dataset_path = r"C:\1st year\data"

# Create folder if not present
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

offset = 20

model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
facedata = []
skip = 0

while True:
    success, img = cam.read()
    if not success:
        print("Failed to capture image")
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3])

    if len(faces) > 0:
        x, y, w, h = faces[-1]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cropped_face = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face,(100, 100))

        skip += 1

        if skip % 10 == 0:
            facedata.append(cropped_face)
            print("Saved so far:", len(facedata))

    cv2.imshow("Image Window", img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# Convert to numpy array
facedata = np.asarray(facedata)

if len(facedata) > 0:
    m = facedata.shape[0]
    facedata = facedata.reshape((m, -1))

    file = os.path.join(dataset_path, filename + ".npy")

    np.save(file, facedata)

    print("Data successfully saved at:")
    print(file)
else:
    print("No face data captured. File not saved.")
