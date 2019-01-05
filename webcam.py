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
#    if cv2.waitKey(1000) & 0xFF == ord('q'):
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
from keras.layers import Dense, Conv2D, Dropout, Flatten
model_base = keras.applications.vgg16.VGG16(include_top=False, input_shape=(100, 100,3), weights='imagenet')
output = model_base.output
output = Flatten()(output)
output = Dense(256, activation = 'sigmoid')(output)
output = Dropout(0.4)(output)
output = Dense(64, activation = 'sigmoid')(output)
output = Dropout(0.4)(output)
output = Dense(10, activation = 'softmax')(output)

model = Model(model_base.input, output)
for layer in model_base.layers:
    layer.trainable = False
class_names = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind', 
               'hair and makeup', 'talking to passenger']
model.load_weights('final models/vgg16_final_model.h5')




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
    img_vgg = cv2.resize(img, (100,100))
    img_vgg = img_vgg.reshape((1,100,100, 3))
    class_value_vgg = model.predict(img_vgg)
    
    
    img_lstm = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_lstm = cv2.resize(img_lstm, (100,100))
    img_lstm = img_lstm.reshape((1,100,100))
    class_value_lstm = model_lstm.predict(img_lstm)
    
    max_vgg = np.max(class_value_vgg)
    max_lstm = np.max(class_value_lstm)
    
    index_vgg = np.argmax(class_value_vgg)
    index_lstm = np.argmax(class_value_lstm)
    
    if(max_vgg > max_lstm):
        return index_vgg
    else: 
        return index_lstm

frame_counter=0
cap = cv2.VideoCapture('1_67Mz9ho1Bx13hbfKkh75pw.gif')
while True:
    ret, frame = cap.read()
    
    
    if ret==False:
        cap = cv2.VideoCapture('1_67Mz9ho1Bx13hbfKkh75pw.gif')
        continue
    #frame = imutils.resize(frame, width=500)
    #frame1 = frame
    #cv2.imshow("Face", frame)
    
    frame_counter += 1
    
    #If the last frame is reached, reset the capture and the frame_counter
    
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    class_val = predict(frame)
    
    cv2.rectangle(frame, (50,360), (520,400), (0,255,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(str(class_names[class_val]))
    status = 'FOCUSED'
    if class_val!=0:
        status = 'DISTRACTED'
        
    cv2.putText(frame, str(status + ' : ' +class_names[class_val]), (75, 385), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Face',frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()