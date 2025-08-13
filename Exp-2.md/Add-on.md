Code :

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

Output :

<img width="1599" height="899" alt="Screenshot 2025-08-13 113212" src="https://github.com/user-attachments/assets/ad10cd20-143b-4f35-8a8c-2284c5566790" />

Test case (Code):

from keras.datasets import fashion_mnist, mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def train_and_test(dataset,class_names,test_indices):
    (X_train,y_train),(X_test,y_test)=dataset.load_data()
    X_train=X_train.reshape(-1,28,28,1).astype('float32')/255
    X_test=X_test.reshape(-1,28,28,1).astype('float32')/255
    y_train_cat=to_categorical(y_train)
    y_test_cat=to_categorical(y_test)
    model=build_cnn((28,28,1),len(class_names))
    model.fit(X_train,y_train_cat,epochs=2,batch_size=64,verbose=0)
    preds=model.predict(X_test[test_indices])
    pred_labels=np.argmax(preds,axis=1)
    df=pd.DataFrame({
        "Input Image":[class_names[y_test[i]] for i in test_indices],
        "True Label":[class_names[y_test[i]] for i in test_indices],
        "Predicted Label":[class_names[p] for p in pred_labels],
        "Correct (Y/N)":["Y" if pred_labels[k]==y_test[test_indices[k]] else "N" for k in range(len(test_indices))]
    })
    return df
fashion_classes=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
mnist_classes=[str(i) for i in range(10)]
fashion_results=train_and_test(fashion_mnist,fashion_classes,[0,1,2,3])
mnist_results=train_and_test(mnist,mnist_classes,[0,1,2,3])
print("Fashion-MNIST Results:\n",fashion_results)
print("\nMNIST Results:\n",mnist_results)

Test case (Output):

<img width="1599" height="899" alt="Screenshot 2025-08-13 114730" src="https://github.com/user-attachments/assets/c75813f2-95a8-46dc-8573-e3b1edfbcaeb" />
<img width="1599" height="899" alt="Screenshot 2025-08-13 114740" src="https://github.com/user-attachments/assets/7caa2617-0c5a-41b5-977e-2f101798fd7c" />

