from keras.models import load_model
import numpy as np
import cv2

label_dict = {
              "glasses":0, 
              "mask":1, 
              "face":2
              }

face_Quality_model = load_model("model/faceQuality_model.h5")   

def judge_face_Quality(face, model=face_Quality_model):
    # print("Using loaded model to predict...")
    
    img = cv2.resize(face, (32, 32))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = img_gray.reshape(1,32,32,1)
    predicted = model.predict(data)
    species_dict = {v: k for k, v in Class_dict.items()}
    return species_dict[np.argmax(predicted)], np.max(predicted)






