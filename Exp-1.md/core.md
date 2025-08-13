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

Test case 1 (Code):

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y, verbose=0)
print(f"Accuracy: {acc*100:.2f}%\n")
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype(int)
df_results = pd.DataFrame({
    "Test Input (X)": [list(x) for x in X],
    "Predicted Output (Raw)": predictions.flatten(),
    "Predicted Class": predicted_classes.flatten(),
    "Expected Output (Y)": Y.flatten()
})
print(df_results.to_string(index=False))

Test case 1 (Output):

<img width="1599" height="899" alt="Screenshot 2025-08-13 102602" src="https://github.com/user-attachments/assets/22b52ed7-bbe9-4371-a228-362ede726eb3" />

Test case 2 (Code):

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
    def activation(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.lr * (target - prediction)
                self.weights[1:] += update * xi
                self.weights[0] += update
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)
print("Perceptron Observations:")
print("Test Input\tActual Output\tExpected\tRemarks")
for xi, target in zip(X, y):
    pred = perceptron.predict(xi)
    remark = "Correct" if pred == target else "May fail"
    print(f"{xi}\t\t{pred}\t\t{target[0]}\t\t{remark}")
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)
print("\nDNN Predictions:")
for xi, target in zip(X, y):
    pred = model.predict(np.array([xi]), verbose=0)
    print(f"Input: {xi} => Predicted: {pred[0][0]:.4f}, Expected: {target[0]}")

Test case 2 (Output):

<img width="1599" height="899" alt="Screenshot 2025-08-13 102731" src="https://github.com/user-attachments/assets/90beb9cd-bf46-49a4-9e3c-40164be50117" />
<img width="1599" height="899" alt="Screenshot 2025-08-13 102753" src="https://github.com/user-attachments/assets/3bf978c9-5ae9-469a-b0ba-0661dd6ad026" />


