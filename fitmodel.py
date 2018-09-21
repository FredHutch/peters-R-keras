import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
 
x_train = np.loadtxt('x_train.csv',dtype=float,skiprows=1,delimiter=',')
y_train=x_train[:,0]
x_train=x_train[:,1:17000]
x_test = np.loadtxt('x_test.csv',dtype=float,skiprows=1,delimiter=',')
y_test=x_test[:,0]
x_test=x_test[:,1:17000]
 
model = Sequential()
model.add(Dense(64, input_dim=17000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
 
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
numpy.savetxt("score.csv", score)