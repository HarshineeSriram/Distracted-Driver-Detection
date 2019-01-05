#import numpy as np
#import cv2
#
#cap = cv2.VideoCapture(0)
#
#while(True):
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#
#    # Our operations on the frame come here
#    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#    # Display the resulting frame
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
## When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

#main_path = 'filtered images/c'
#class_values2=[]
#images2 = [] 
#for i in range(10):
#    for root, dirs, files in os.walk("filtered images/c"+str(i)):  
#        for filename in files[:10]:
#            im_path = 'filtered images/c' + str(i) + '/' + str(filename)
#            img = cv2.imread('filtered images/c' + str(i) + '/' + str(filename))
#            img = cv2.resize(img, (100,100))/256
#            images2.append(img)
#            class_values2.append(i)
#np_images2 = np.asarray(images2)
#targets = np.asarray(class_values2)
#
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, LSTM
#model_base = keras.applications.vgg16.VGG16(include_top=False, input_shape=(100, 100,3), weights='imagenet')
#output = model_base.output
#output = Flatten()(output)
#output = Dense(256, activation = 'sigmoid')(output)
#output = Dropout(0.4)(output)
#output = Dense(64, activation = 'sigmoid')(output)
#output = Dropout(0.4)(output)
#output = Dense(10, activation = 'softmax')(output)
#
#model = Model(model_base.input, output)
#for layer in model_base.layers:
#    layer.trainable = False
#    
#model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#model.fit(np_images2, targets, epochs = 5, validation_split = 0.1, batch_size = 32)

#model.save('model')
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, LSTM

model_lstm = keras.models.Sequential()
model_lstm.add(LSTM(50, input_shape = (100,100), return_sequences=True))
model_lstm.add(Dropout(0.3))
model_lstm.add(LSTM(25, return_sequences=True))
model_lstm.add(Dropout(0.3))
model_lstm.add(Flatten())

model_lstm.add(Dense(256, activation = 'relu'))
model_lstm.add(Dropout(0.4))
model_lstm.add(Dense(64, activation = 'relu'))
model_lstm.add(Dropout(0.4))
model_lstm.add(Dense(10, activation = 'softmax'))

model_lstm.load_weights('final models/lstm_final_model.h5')
def predict(img):
    img = cv2.resize(img, (100,100))
    img = img.reshape((1,100,100))
    class_value = model1.predict(img)
    return class_value

cap = cv2.VideoCapture('1_67Mz9ho1Bx13hbfKkh75pw.gif')
while True:
    ret, frame = cap.read()
    if ret==False:
        continue
    #frame = imutils.resize(frame, width=500)
    #frame1 = frame
    cv2.imshow("Face", frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    class_val = predict(frame)
    cv2.rectangle(frame, (50,50), (100,100), (255,255,255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(str(class_val))
    cv2.putText(frame, str(class_val), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Face',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()