import cv2
import numpy as np
import os

# ================= LOAD DATA =================
dataset_path = r"C:\1st year\data"

facedata = []
labels = []
classId = 0
nameMap = {}

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        file_path = os.path.join(dataset_path, f)

        nameMap[classId] = f[:-4]

        dataItem = np.load(file_path)
        m = dataItem.shape[0]

        facedata.append(dataItem)

        target = classId * np.ones((m,))
        labels.append(target)

        classId += 1

X = np.concatenate(facedata, axis=0)
Y = np.concatenate(labels, axis=0)   # ✅ FIXED (1D)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ================= KNN =================
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))  # y[i] is scalar now

    dlist = sorted(dlist, key=lambda x: x[0])

    dlist = dlist[:k]

    labels = [item[1] for item in dlist]

    labels, cnts = np.unique(labels, return_counts=True)

    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)


# ================= PREDICTION =================
cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier(
    r"C:\1st year\udemy\haarcascade_frontalface_alt.xml"
)

offset = 10   # ✅ defined properly

while True:
    success, img = cam.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(gray, 1.3, 5)

    for f in faces:
        x, y, w, h = f

        # Prevent negative slicing
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = x + w + offset
        y2 = y + h + offset

        cropped_face = img[y1:y2, x1:x2]
        cropped_face = cv2.resize(cropped_face, (100, 100))

        # Flatten before prediction
        classpredicted = knn(X, Y, cropped_face.flatten())

        namePredicted = nameMap[classpredicted]

        cv2.putText(img, namePredicted, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 200, 0), 2, cv2.LINE_AA)

        cv2.rectangle(img, (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)

    cv2.imshow("Prediction Window", img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
