code :

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y)
print("Accuracy:", acc)
print("Predictions:", model.predict(X))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

output :

![IMG-20250807-WA0005](https://github.com/user-attachments/assets/5307dd5f-cdcc-47ca-9205-1a0ec72e307b)
![IMG-20250807-WA0004](https://github.com/user-attachments/assets/c8d32cb1-5c88-4211-ae8b-334e04c76b73)
