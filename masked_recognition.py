import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

SCALE = (100, 100)
BASE_DIR= r"C:\Users\lafacero\Documents\GitHub\covid19-mask-recognition"

count_masked=0
count_unmasked=0

# Importiamo il Modello
model=load_model(os.path.join(BASE_DIR, r"masked_recognition.h5"))
cap = cv2.VideoCapture(0)

if(not cap.read()[0]):
    print("webcam non disponibile")
    exit(0)

face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, r"haarcascade_frontalface_default.xml"))
while(cap.isOpened()):
    _, frame = cap.read()
    #il face detection va fatto su immagini in b/n con opencv
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cerco i volti nell'immagine con detectMultiScale
    rects = face_cascade.detectMultiScale(gray, 1.1, 15)

    for rect in rects:
        
        #leggo l'immagine prendendo y e x di partenza e arrivo, ritagliando l'immagine su queste coordinate per passarle alla rete neurale
        img = gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        #ridimensiono l'immagine
        resized=cv2.resize(img, SCALE)

        #normalizzo l'immagine
        normalized=resized/255.0

        #per passare l'array alla rete neurale la devo ridimensionare come tensore
        reshaped=np.reshape(normalized,(1,SCALE[0],SCALE[1],1))

        #predizione
        result=model.predict(reshaped)
        
        y=np.argmax(result,axis=1)[0]
        label = "No maschera" if y>0.5 else "Maschera"
        
        if y>0.5:
            count_unmasked+=1 
        else:
            count_masked+=1

        #stampo il rettangolo, label, etc. intorno all'immagine originale
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
        cv2.rectangle(frame, (rect[0], rect[1]-20), (rect[0]+170, rect[1]), (0,255,0), cv2.FILLED)
        cv2.putText(frame, label, (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, 'Mascherine rilevate: '+str(count_masked), (frame.shape[1]-200, frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 2) 

        #reset contatori per il frame successivo
        count_masked=0
        count_unmasked=0
        
    cv2.imshow("LIVE", frame)
    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()