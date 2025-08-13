code :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
  if y[i] == 0:
    plt.scatter(X[i][0], X[i][1], color='red')
  else:
    plt.scatter(X[i][0], X[i][1], color='blue')
x_values = [0, 1]
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()

output :

![IMG-20250807-WA0006](https://github.com/user-attachments/assets/e4f8b9d4-0b5f-4f29-b6cd-6e8bc06a01ac)
![IMG-20250807-WA0007](https://github.com/user-attachments/assets/548639c1-23b4-4085-ba03-aa94fb1f64c3)

Test case 1 (Code):

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
predictions = clf.predict(X)
df_results = pd.DataFrame({
    "Test Input (X)": [list(x) for x in X],
    "Predicted Output": predictions,
    "Expected Output (Y)": y,
    "Remarks": ["Correct" if predictions[i] == y[i] else "Incorrect"
                for i in range(len(y))]
})

print("Single-Layer Perceptron on XOR Problem\n")
print(df_results.to_string(index=False))
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i == 1 else "")
x_values = np.array([0, 1])
y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.title('Single-Layer Perceptron Decision Boundary for XOR')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

Test case 1 (Output):

<img width="1599" height="893" alt="Screenshot 2025-08-13 102645" src="https://github.com/user-attachments/assets/b9f79b62-e821-45ca-b757-8a74acd2e06e" />
<img width="1599" height="899" alt="Screenshot 2025-08-13 102704" src="https://github.com/user-attachments/assets/91a9cafd-7628-4bb2-bf7d-1b23205ef88f" />


Test case 2 (Code):

import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])
np.random.seed(42)
weights = np.random.randn(2)
bias = np.random.randn()
def step_function(z):
    return np.where(z >= 0, 1, 0)
learning_rate = 0.1
epochs = 10

for _ in range(epochs):
    for xi, target in zip(X, y):
        output = step_function(np.dot(xi, weights) + bias)
        error = target - output
        weights += learning_rate * error * xi
        bias += learning_rate * error
predictions = step_function(np.dot(X, weights) + bias)
print("Test Input\tPerceptron Output\tExpected\tRemarks")
for xi, pred, exp in zip(X, predictions, y):
    remark = "Correct" if pred == exp else "May fail"
    print(f"{xi}\t\t{pred}\t\t{exp}\t\t{remark}")
colors = ['red' if label == 0 else 'blue' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k', s=100)
x_values = np.linspace(-0.5, 1.5, 50)
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Single-Layer Perceptron on XOR')
plt.legend()
plt.grid(True)
plt.show()

Test case 2 (Output):

<img width="1599" height="897" alt="Screenshot 2025-08-13 102826" src="https://github.com/user-attachments/assets/31eb0076-ed3e-42af-b6c4-98f3440eaf3e" />
<img width="1599" height="899" alt="Screenshot 2025-08-13 102849" src="https://github.com/user-attachments/assets/a2bf531f-41b2-462e-83d0-ebc5d21f8f78" />
