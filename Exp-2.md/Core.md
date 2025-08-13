Code :

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(128, activation='relu'),
Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

Output :

<img width="1599" height="899" alt="Screenshot 2025-08-13 111516" src="https://github.com/user-attachments/assets/cf91bf5d-ee97-4972-8656-d03a8f2961ce" />

Test case (Code):

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(-1,28,28,1).astype('float32')/255
X_test=X_test.reshape(-1,28,28,1).astype('float32')/255
y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train_cat,epochs=2,batch_size=64,verbose=0)
indices=[0,1,2,3]
preds=model.predict(X_test[indices])
pred_labels=np.argmax(preds,axis=1)
df=pd.DataFrame({
    "Input Digit Image":["Image of "+str(y_test[i]) for i in indices],
    "Expected Label":[y_test[i] for i in indices],
    "Model Output":[p for p in pred_labels],
    "Correct (Y/N)":["Y" if pred_labels[k]==y_test[indices[k]] else "N" for k in range(len(indices))]
})
print(df)

Test case (Output):

<img width="1599" height="899" alt="Screenshot 2025-08-13 114702" src="https://github.com/user-attachments/assets/43f61ba5-b236-4456-ba44-49618744c350" />
<img width="1599" height="899" alt="Screenshot 2025-08-13 114716" src="https://github.com/user-attachments/assets/64792fda-011e-40e5-89f9-4ff4e867d928" />

