from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

finger_path1 = 'TRAIN_finger_demo02'
categories1 = ['0', '1', '2', '3', '4', '5']
num_class1 = len(categories1)
X = []
Y = []
for idx, category in enumerate(categories1):
    label = [0 for i in range(num_class1)]
    label[idx] = 1
    image_dir = finger_path1 + '/' + category + '/'
    for top, dir, f in os.walk(image_dir):
#        f.remove('.DS_Store')
        for filename in f:
            img = cv2.imread(image_dir + filename)
            X.append(img/128)
            Y.append(label)

finger_path2 = 'TEST_finger_demo02'
categories2 = ['0', '1', '2', '3', '4', '5']
num_class2 = len(categories2)
Xx=[]
Yy=[]
for idx, category in enumerate(categories2):
    label = [0 for i in range(num_class2)]
    label[idx] = 1
    image_dir = finger_path2 + '/' + category + '/'
    for top, dir, f in os.walk(image_dir):
#        f.remove('.DS_Store')
        for filename in f:
            img = cv2.imread(image_dir + filename)
            Xx.append(img/128)
            Yy.append(label)

Xtr = np.array(X)
Ytr = np.array(Y)
Xte = np.array(Xx)
Yte = np.array(Yy)
X_train, Y_train = Xtr, Ytr
X_test, Y_test = Xte, Yte

model = Sequential()
model.add(Conv2D(32, 3, 3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))

model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(6, activation='softmax'))
#print(model.summary())

# 모델 학습과정 설정하기
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# optimizer :
# loss :
# metrics

# 모델 학습시키기
#model.fit(X_train, Y_train, epochs=1)
history = model.fit(X_train, Y_train, batch_size =10, validation_split = 0.2, epochs = 5, verbose = 0)
#print(model.summary())



# 모델 학습과정 살펴보기
print('## training loss and acc ##')
print(history.history['loss'])
#print(history.history['acc'])
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.legend(['training', 'validation'], loc = 'upper left')
#plt.show()


# 모델 평가하기
results = model.evaluate(X_test, Y_test, batch_size=32)
print('Test accuracy: ', results)

# 모델 저장하기
model.save('hand_detect_model2.h5')

# 모델 사용하기
Y_predict = model.predict(X_test)
print(Y_predict.shape)
print(len(Y_test))
correct = 0
for i in range(len(Y_test)):
    #print(" True : ", np.argmax(Y_test[i]), ", Predict : ", np.argmax(Y_predict[i]))
    if (np.argmax(Y_test[i]) == np.argmax(Y_predict[i])) :
        correct += 1
print(len(Y_test), " 중 ", correct, " 개 일치")

