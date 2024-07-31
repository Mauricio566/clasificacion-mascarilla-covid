# cada frame primero se procesa con el modelo SSD para detectar los
# rostros en la imagen. Una vez que se detectan los rostros, las regiones de interés (ROIs) que contienen los rostros se extraen de la imagen original y se pasan al modelo que entrenaste tú mismo para detectar si llevan mascarilla o no.
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array#se utiliza para convertir una imagen representada como un objeto de la biblioteca Pillow (PIL) en un array de NumPy. Este paso es comúnmente necesario antes de pasar la imagen a un modelo de red neuronal convolucional para su procesamiento
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input#La función preprocess_input es específica de la arquitectura MobileNetV2 y se utiliza para preprocesar las imágenes antes de pasarlas al modelo MobileNetV2 para la inferencia


import numpy as np
#print("Versión de TensorFlow:", tf.__version__)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

#se realiza videocaptura
cap = cv2.VideoCapture(0)


faceNet = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt","res10_300x300_ssd_iter_140000.caffemodel")
modelo = "C:\\Users\\User\\Documents\\nueva_inferencia_covid\\model_mask_covid"
# load the DNN model

mySelf_model = load_model(modelo,compile=False)

# Compilar el modelo manualmente
mySelf_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

locs = []
preds = []


while True:
    faces = []
    
    # Leemos los frames(returna 2 valores)
    ret,frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    h = frame.shape[0]#altura de cada frame
    w = frame.shape[1]#ancho de cada frame
	
    #factor de escala que se aplica a cada píxel de la imagen de entrada. 
    # En este caso, se establece en 1.0, lo que significa que no se aplica 
    # ningún cambio de escala a la imagen.
    #y se resta el valor medio de cada canal de color
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    #pasar el blob(frame procesado) al modelo
    faceNet.setInput(blob)
    #propagacion hacia adelante
    detections = faceNet.forward()
    #print(detections)
    
    umbral_confianza = 0.8
    #for i in range(0,detections.shape[2]):#print("i ",i)#0,1,..199
    #print(detections.shape[2])#200
    #print(detections.shape)#(1,1,200,7)
    #print(type(detections.shape))#<class 'tuple'>
    #print(type(detections))#<class 'numpy.ndarray'>
    print(detections.ndim)#4
    
    
    
    
    for i in range(detections.shape[2]):#print("i ",i)#0,1,..199
        
        confidence = detections[0, 0, i, 2]#0.07739706,0.077314354 etc
        
        if confidence > umbral_confianza:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])#box[142.03503609 232.37307072 326.42650604 468.88217926]
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = mySelf_model.predict(faces, batch_size=32)
                #print("predicciones de mi modelo ",preds)
                #return (locs, preds)
                
                
            for (box, pred) in zip(locs, preds):
                #print(box)#(209, 217, 355, 405)
                #print(type(box))#<class 'tuple'>
                
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred#desempaquetando tupla con 2 valores
                # determine the class label and color we'll use to draw
                # the bounding box and text
                #label = "Con tapabocas" if mask > withoutMask else "Sin tapabocas"
                if mask > withoutMask:
                    label = "con tapabocas"
                else:
                    label = "sin tapabocas"    
                
                if label == "con tapabocas":
                    color = (0,255,0)
                else:
                    color =  (0, 0, 255)
                         
                #color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

  

    
    
    
    
    
    
    cv2.imshow("DETECCION DE ROSTROS", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()    
