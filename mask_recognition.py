import cv2
from tensorflow.keras.models import load_model

SCALE = (200, 200)

# Importiamo il Modello
model=load_model(r"C:\Users\lafacero\Desktop\mask recognition\mask_recognition.h5")
cap = cv2.VideoCapture(0)

if(not cap.read()[0]):
    print("webcam non disponibile")
    exit(0)

face_cascade = cv2.CascadeClassifier(r"C:\Users\lafacero\Desktop\mask recognition\haarcascade_frontalface_default.xml")
while(cap.isOpened()):
    _, frame = cap.read()
    #il face detection va fatto su immagini in b/n con opencv
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cerco i volti nell'immagine con detectMultiScale
    rects = face_cascade.detectMultiScale(gray, 1.1, 15)

    for rect in rects:
        #leggo l'immagine prendendo y e x di partenza e arrivo, ritagliando l'immagine su queste coordinate per passarle alla rete neurale
        img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        #ridimensiono l'immagine
        small_img = cv2.resize(img, SCALE)
        #normalizzo l'immagine
        x = small_img.astype(float)
        x/=255.

        
        #per passare l'array alla rete neurale la devo ridimensionare come tensore
        x = x.reshape(1, SCALE[0], SCALE[1], 3)
        y = model.predict(x)

        y=y[0][0]
        label = "No maschera" if y>0.5 else "Maschera"

        #stampo il rettangolo, label, etc. intorno all'immagine originale
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
        cv2.rectangle(frame, (rect[0], rect[1]-20), (rect[0]+170, rect[1]), (0,255,0), cv2.FILLED)
        cv2.putText(frame, label, (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 2)

    cv2.imshow("Gender Recognition", frame)
    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
